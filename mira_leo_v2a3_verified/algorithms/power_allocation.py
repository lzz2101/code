from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .regrouping import Group


class PowerAllocationError(ValueError):
    """Raised when Algorithm 3 power allocation is infeasible or invalid."""


@dataclass(frozen=True)
class PowerAllocationSettings:
    P_sat: float
    V_c: float
    V_nc: float
    V_phi: float


def validate_power_settings(
    P_sat: float, V_c: float, V_nc: float, V_phi: float, tolerance: float = 1e-9
) -> None:
    if not isfinite(tolerance) or tolerance < 0.0:
        raise PowerAllocationError("tolerance must be finite and non-negative")
    if not isfinite(P_sat) or P_sat <= 0.0:
        raise PowerAllocationError("P_sat must be finite and positive")
    if not isfinite(V_phi) or V_phi <= 0.0:
        raise PowerAllocationError("V_phi must be finite and positive")
    if not isfinite(V_c) or not 0.0 < V_c < 1.0:
        raise PowerAllocationError("V_c must satisfy 0 < V_c < 1")
    if not isfinite(V_nc) or not 0.0 < V_nc < 1.0:
        raise PowerAllocationError("V_nc must satisfy 0 < V_nc < 1")
    if abs((V_c + V_nc) - 1.0) > tolerance:
        raise PowerAllocationError("V_c + V_nc must equal 1 within tolerance")


def _group_order_key(group: Group) -> tuple[float, int, int]:
    return (float(group.group_gain), int(group.message_id), int(group.group_id))


def _scheduled_groups(groups: list[Group]) -> list[Group]:
    return [group for group in groups if group.scheduled]


def _validate_groups(
    groups: list[Group], candidates: list[Group], satellite_id: int | None
) -> int | None:
    group_sat_ids = {int(group.satellite_id) for group in groups}
    if satellite_id is not None:
        group_sat_ids.add(int(satellite_id))
    if len(group_sat_ids) > 1:
        raise PowerAllocationError(
            f"Algorithm 3 expects one satellite per call, got satellite ids {sorted(group_sat_ids)}"
        )

    for group in candidates:
        gain = float(group.group_gain)
        if not isfinite(gain) or gain < 0.0:
            raise PowerAllocationError(
                f"invalid group_gain for satellite {group.satellite_id}, "
                f"message {group.message_id}, group {group.group_id}: {gain}"
            )

    return next(iter(group_sat_ids)) if group_sat_ids else satellite_id


def _infeasible_message(
    *,
    satellite_id: int | None,
    slot: int | None,
    subset_name: str,
    count: int,
    budget: float,
    V_phi: float,
    floor_total: float,
) -> str:
    return (
        "Algorithm 3 floor infeasible: "
        f"satellite={satellite_id}, slot={slot}, subset={subset_name}, "
        f"N={count}, X={budget:.12g}, V_phi={V_phi:.12g}, "
        f"floor_total={floor_total:.12g}"
    )


def _allocate_subset(
    subset: list[Group],
    *,
    budget: float,
    V_phi: float,
    satellite_id: int | None,
    slot: int | None,
    subset_name: str,
    tolerance: float,
) -> dict[str, object]:
    ordered = sorted(subset, key=_group_order_key)
    count = len(ordered)
    if count == 0:
        return {
            "subset": subset_name,
            "count": 0,
            "budget": float(budget),
            "floor_total": 0.0,
            "allocated_power": 0.0,
            "residual": 0.0,
            "residual_share": 0.0,
            "floor_feasible": True,
            "ordered_group_ids": [],
        }

    floor_total = float(V_phi * count * (count + 1) / 2.0)
    if floor_total > budget + tolerance:
        raise PowerAllocationError(
            _infeasible_message(
                satellite_id=satellite_id,
                slot=slot,
                subset_name=subset_name,
                count=count,
                budget=budget,
                V_phi=V_phi,
                floor_total=floor_total,
            )
        )

    residual = float(budget - floor_total)
    if residual < 0.0 and abs(residual) <= tolerance:
        residual = 0.0
    if residual < 0.0:
        raise PowerAllocationError(
            _infeasible_message(
                satellite_id=satellite_id,
                slot=slot,
                subset_name=subset_name,
                count=count,
                budget=budget,
                V_phi=V_phi,
                floor_total=floor_total,
            )
        )

    residual_share = residual / count
    allocated_power = 0.0
    previous_power = None
    ordered_group_ids = []

    for rank, group in enumerate(ordered, start=1):
        floor_power = float(V_phi * (count - rank + 1))
        final_power = float(floor_power + residual_share)
        if not isfinite(final_power) or final_power <= 0.0:
            raise PowerAllocationError(
                f"invalid allocated power for satellite={satellite_id}, slot={slot}, "
                f"subset={subset_name}, rank={rank}: {final_power}"
            )
        if previous_power is not None and previous_power + tolerance < final_power:
            raise PowerAllocationError(
                f"subset power order violation inside {subset_name}: "
                f"previous={previous_power}, current={final_power}"
            )

        group.power = final_power
        group.metadata["power_subset"] = subset_name
        group.metadata["sic_rank_in_subset"] = float(rank)
        group.metadata["subset_size"] = float(count)
        group.metadata["power_group_gain"] = float(group.group_gain)
        group.metadata["floor_power"] = floor_power
        group.metadata["residual_share"] = float(residual_share)
        group.metadata["subset_budget"] = float(budget)
        group.metadata["final_power"] = final_power

        ordered_group_ids.append(
            {
                "message_id": int(group.message_id),
                "group_id": int(group.group_id),
                "group_gain": float(group.group_gain),
                "power": final_power,
            }
        )
        allocated_power += final_power
        previous_power = final_power

    if abs(allocated_power - budget) > tolerance:
        raise PowerAllocationError(
            f"subset allocation sum mismatch: satellite={satellite_id}, slot={slot}, "
            f"subset={subset_name}, allocated={allocated_power}, budget={budget}"
        )

    return {
        "subset": subset_name,
        "count": count,
        "budget": float(budget),
        "floor_total": floor_total,
        "allocated_power": float(allocated_power),
        "residual": residual,
        "residual_share": float(residual_share),
        "floor_feasible": True,
        "ordered_group_ids": ordered_group_ids,
    }


def _global_sic_power_order_diagnostics(
    candidates: list[Group], tolerance: float
) -> dict[str, object]:
    ordered = sorted(candidates, key=_group_order_key)
    violation_count = 0
    max_violation = 0.0
    violation_details = []

    for rank, group in enumerate(ordered, start=1):
        group.metadata["global_sic_rank"] = rank

    for i, weaker in enumerate(ordered):
        for stronger in ordered[i + 1 :]:
            magnitude = float(stronger.power - weaker.power)
            if magnitude > tolerance:
                violation_count += 1
                max_violation = max(max_violation, magnitude)
                violation_details.append(
                    {
                        "weaker_message_id": int(weaker.message_id),
                        "weaker_group_id": int(weaker.group_id),
                        "weaker_gain": float(weaker.group_gain),
                        "weaker_power": float(weaker.power),
                        "stronger_message_id": int(stronger.message_id),
                        "stronger_group_id": int(stronger.group_id),
                        "stronger_gain": float(stronger.group_gain),
                        "stronger_power": float(stronger.power),
                        "violation": magnitude,
                    }
                )

    return {
        "sic_power_order_violations": violation_count,
        "max_sic_power_order_violation": float(max_violation),
        "sic_power_order_violation_details": violation_details,
    }


def _budget_decision(
    edge_count: int, nonedge_count: int, P_sat: float, V_c: float, V_nc: float
) -> tuple[float, float, str]:
    if edge_count > 0 and nonedge_count > 0:
        return float(V_c * P_sat), float(V_nc * P_sat), "both_subsets"
    if edge_count > 0:
        return float(P_sat), 0.0, "edge_only"
    if nonedge_count > 0:
        return 0.0, float(P_sat), "nonedge_only"
    return 0.0, 0.0, "empty"


def mira_leo_power_allocation(
    groups: list[Group],
    *,
    P_sat: float,
    V_c: float,
    V_nc: float,
    V_phi: float,
    slot: int | None = None,
    satellite_id: int | None = None,
    tolerance: float = 1e-9,
    enforce_global_sic_order: bool = True,
) -> dict[str, object]:
    """
    Algorithm 3: MIRA-LEO Power Allocation for one satellite and one slot.

    The function resets every input group's power to zero, allocates only
    scheduled groups, and returns diagnostics suitable for per-slot logging.
    """
    validate_power_settings(P_sat, V_c, V_nc, V_phi, tolerance)

    for group in groups:
        group.power = 0.0

    candidates = _scheduled_groups(groups)
    resolved_satellite_id = _validate_groups(groups, candidates, satellite_id)

    if not candidates:
        return {
            "satellite_id": resolved_satellite_id,
            "slot": slot,
            "P_sat": float(P_sat),
            "V_c": float(V_c),
            "V_nc": float(V_nc),
            "V_phi": float(V_phi),
            "budget_reason": "empty",
            "edge_count": 0,
            "nonedge_count": 0,
            "edge_budget": 0.0,
            "nonedge_budget": 0.0,
            "edge_power": 0.0,
            "nonedge_power": 0.0,
            "total_allocated_power": 0.0,
            "floor_feasible": True,
            "power_floor_infeasible_count": 0,
            "sic_power_order_violations": 0.0,
            "max_sic_power_order_violation": 0.0,
            "sic_power_order_violation_details": [],
            "subsets": [],
        }

    edge_groups = [group for group in candidates if group.is_edge]
    nonedge_groups = [group for group in candidates if not group.is_edge]
    edge_budget, nonedge_budget, budget_reason = _budget_decision(
        len(edge_groups), len(nonedge_groups), P_sat, V_c, V_nc
    )

    edge_summary = _allocate_subset(
        edge_groups,
        budget=edge_budget,
        V_phi=V_phi,
        satellite_id=resolved_satellite_id,
        slot=slot,
        subset_name="edge",
        tolerance=tolerance,
    )
    nonedge_summary = _allocate_subset(
        nonedge_groups,
        budget=nonedge_budget,
        V_phi=V_phi,
        satellite_id=resolved_satellite_id,
        slot=slot,
        subset_name="nonedge",
        tolerance=tolerance,
    )

    for group in groups:
        if not group.scheduled and group.power != 0.0:
            raise PowerAllocationError(
                f"unscheduled group has nonzero power: satellite={group.satellite_id}, "
                f"message={group.message_id}, group={group.group_id}, power={group.power}"
            )
    for group in candidates:
        if not isfinite(group.power) or group.power <= 0.0:
            raise PowerAllocationError(
                f"scheduled group has invalid power: satellite={group.satellite_id}, "
                f"message={group.message_id}, group={group.group_id}, power={group.power}"
            )

    edge_power = float(edge_summary["allocated_power"])
    nonedge_power = float(nonedge_summary["allocated_power"])
    total_allocated_power = edge_power + nonedge_power

    if abs(edge_power - edge_budget) > tolerance and edge_groups:
        raise PowerAllocationError("edge power budget mismatch")
    if abs(nonedge_power - nonedge_budget) > tolerance and nonedge_groups:
        raise PowerAllocationError("non-edge power budget mismatch")
    if abs(total_allocated_power - P_sat) > tolerance:
        raise PowerAllocationError(
            f"total allocated power must equal P_sat when scheduled groups exist: "
            f"allocated={total_allocated_power}, P_sat={P_sat}"
        )
    if total_allocated_power > P_sat + tolerance:
        raise PowerAllocationError(
            f"total allocated power exceeds P_sat: allocated={total_allocated_power}, P_sat={P_sat}"
        )

    diagnostics = _global_sic_power_order_diagnostics(candidates, tolerance)
    if enforce_global_sic_order and diagnostics["sic_power_order_violations"]:
        first = diagnostics["sic_power_order_violation_details"][0]
        raise PowerAllocationError(
            "Algorithm 3 violates global SIC power order (15c): "
            f"satellite={resolved_satellite_id}, slot={slot}, "
            f"weaker=(message={first['weaker_message_id']}, "
            f"group={first['weaker_group_id']}, gain={first['weaker_gain']}, "
            f"power={first['weaker_power']}), "
            f"stronger=(message={first['stronger_message_id']}, "
            f"group={first['stronger_group_id']}, gain={first['stronger_gain']}, "
            f"power={first['stronger_power']})"
        )
    return {
        "satellite_id": resolved_satellite_id,
        "slot": slot,
        "P_sat": float(P_sat),
        "V_c": float(V_c),
        "V_nc": float(V_nc),
        "V_phi": float(V_phi),
        "budget_reason": budget_reason,
        "edge_count": len(edge_groups),
        "nonedge_count": len(nonedge_groups),
        "edge_budget": edge_budget,
        "nonedge_budget": nonedge_budget,
        "edge_power": edge_power,
        "nonedge_power": nonedge_power,
        "total_allocated_power": float(total_allocated_power),
        "floor_feasible": True,
        "power_floor_infeasible_count": 0,
        "sic_power_order_violations": diagnostics["sic_power_order_violations"],
        "max_sic_power_order_violation": diagnostics[
            "max_sic_power_order_violation"
        ],
        "sic_power_order_violation_details": diagnostics[
            "sic_power_order_violation_details"
        ],
        "subsets": [edge_summary, nonedge_summary],
    }


def power_allocation_placeholder(
    scheduled_groups: list[Group], P_sat: float = 1.0
) -> None:
    """Legacy equal-power baseline retained only for explicit comparisons."""
    if not scheduled_groups:
        return

    power = float(P_sat / len(scheduled_groups))
    for group in scheduled_groups:
        group.power = power
