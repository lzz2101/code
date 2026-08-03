from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Iterable

from mira_leo_v2a3_verified.algorithms.regrouping import Group
from mira_leo_v2a3_verified.config import V2Config
from mira_leo_v2a3_verified.noma.transmission import evaluate_noma_decoding
from mira_leo_v2a3_verified.time_units import milliseconds_to_slots, seconds_to_slots


class ValidationError(ValueError):
    """Raised when strict validation finds a substantive violation."""


@dataclass(frozen=True)
class Violation:
    code: str
    message: str
    slot: int | None = None
    satellite_id: int | None = None
    message_id: int | None = None
    group_id: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _finish(violations: list[Violation], strict: bool) -> list[Violation]:
    if strict and violations:
        first = violations[0]
        raise ValidationError(f"{first.code}: {first.message}")
    return violations


def validate_reward_record(
    record: dict[str, object], *, tolerance: float = 1e-9, strict: bool = True
) -> list[Violation]:
    violations: list[Violation] = []
    reward = float(record["normalized_reward"])
    slot = int(record["slot"])
    satellite_id = int(record["satellite_id"])

    if not isfinite(reward):
        violations.append(
            Violation(
                "nonfinite_reward",
                f"Reward violation: satellite={satellite_id}, slot={slot}, "
                f"action={record['action']}, raw_weighted_sum_aoi="
                f"{record['raw_weighted_sum_aoi']}, raw_reward={record['raw_reward']}, "
                f"normalized_reward={reward}, normalization_parameters="
                f"{record['normalization_parameters']}",
                slot,
                satellite_id,
            )
        )
    elif reward < -tolerance or reward > 1.0 + tolerance:
        violations.append(
            Violation(
                "reward_out_of_bounds",
                f"Reward violation: satellite={satellite_id}, slot={slot}, "
                f"action={record['action']}, raw_weighted_sum_aoi="
                f"{record['raw_weighted_sum_aoi']}, raw_reward={record['raw_reward']}, "
                f"normalized_reward={reward}, normalization_parameters="
                f"{record['normalization_parameters']}",
                slot,
                satellite_id,
            )
        )
    return _finish(violations, strict)


def validate_power_constraints(
    groups: Iterable[Group],
    *,
    P_sat: float,
    slot: int | None = None,
    satellite_id: int | None = None,
    tolerance: float = 1e-9,
    strict: bool = True,
) -> list[Violation]:
    grouped: dict[int, list[Group]] = {}
    for group in groups:
        grouped.setdefault(int(group.satellite_id), []).append(group)
    if satellite_id is not None:
        grouped.setdefault(int(satellite_id), [])

    violations: list[Violation] = []
    for sat_id, satellite_groups in grouped.items():
        scheduled_power = 0.0
        for group in satellite_groups:
            power = float(group.power)
            identity = {
                "slot": slot,
                "satellite_id": sat_id,
                "message_id": int(group.message_id),
                "group_id": int(group.group_id),
            }
            if not isfinite(power):
                violations.append(
                    Violation(
                        "nonfinite_power",
                        f"non-finite power={power}, gain={group.group_gain}",
                        **identity,
                    )
                )
                continue
            if power < -tolerance:
                violations.append(
                    Violation(
                        "negative_power",
                        f"negative power={power}, gain={group.group_gain}",
                        **identity,
                    )
                )
            if group.scheduled:
                scheduled_power += power
                if power <= 0.0:
                    violations.append(
                        Violation(
                            "scheduled_nonpositive_power",
                            f"scheduled group has power={power}, gain={group.group_gain}",
                            **identity,
                        )
                    )
            elif power > tolerance:
                violations.append(
                    Violation(
                        "unscheduled_positive_power",
                        f"unscheduled group has power={power}, gain={group.group_gain}",
                        **identity,
                    )
                )

        if scheduled_power > P_sat + tolerance:
            violations.append(
                Violation(
                    "satellite_power_overflow",
                    f"satellite={sat_id}, slot={slot}, allocated={scheduled_power}, "
                    f"budget={P_sat}",
                    slot,
                    sat_id,
                )
            )
    return _finish(violations, strict)


def validate_allocation_summary(
    summary: dict[str, object], *, tolerance: float = 1e-9, strict: bool = True
) -> list[Violation]:
    violations: list[Violation] = []
    sat_id = summary.get("satellite_id")
    slot = summary.get("slot")
    total = float(summary["total_allocated_power"])
    budget = float(summary["P_sat"])

    if total > budget + tolerance:
        violations.append(
            Violation(
                "satellite_power_overflow",
                f"allocated={total}, budget={budget}",
                int(slot) if slot is not None else None,
                int(sat_id) if sat_id is not None else None,
            )
        )
    if int(summary["edge_count"]) + int(summary["nonedge_count"]) > 0:
        if abs(total - budget) > tolerance:
            violations.append(
                Violation(
                    "allocated_power_not_equal_budget",
                    f"allocated={total}, expected={budget}",
                    int(slot) if slot is not None else None,
                    int(sat_id) if sat_id is not None else None,
                )
            )
    if int(summary["sic_power_order_violations"]) > 0:
        violations.append(
            Violation(
                "sic_power_order",
                f"violations={summary['sic_power_order_violations']}, "
                f"max={summary['max_sic_power_order_violation']}",
                int(slot) if slot is not None else None,
                int(sat_id) if sat_id is not None else None,
            )
        )
    return _finish(violations, strict)


def validate_sic_power_order(
    groups: Iterable[Group], *, tolerance: float = 1e-9, strict: bool = True
) -> list[Violation]:
    ordered = sorted(
        (group for group in groups if group.scheduled),
        key=lambda group: (
            float(group.group_gain),
            int(group.message_id),
            int(group.group_id),
        ),
    )
    violations: list[Violation] = []
    for weaker, stronger in zip(ordered, ordered[1:]):
        if stronger.power > weaker.power + tolerance:
            violations.append(
                Violation(
                    "sic_power_order",
                    f"weaker=(message={weaker.message_id}, group={weaker.group_id}, "
                    f"gain={weaker.group_gain}, power={weaker.power}), "
                    f"stronger=(message={stronger.message_id}, group={stronger.group_id}, "
                    f"gain={stronger.group_gain}, power={stronger.power})",
                    satellite_id=int(weaker.satellite_id),
                    message_id=int(weaker.message_id),
                    group_id=int(weaker.group_id),
                )
            )
    return _finish(violations, strict)


def validate_sic_success(
    groups: list[Group],
    *,
    noise_power: float,
    rate_threshold: float,
    slot: int | None = None,
    strict: bool = True,
) -> list[Violation]:
    scheduled = [group for group in groups if group.scheduled]
    expected = evaluate_noma_decoding(scheduled, noise_power, rate_threshold)
    expected_results = expected["group_results"]
    violations: list[Violation] = []

    for group in scheduled:
        expected_success = bool(
            expected_results[(int(group.message_id), int(group.group_id))]["success"]
        )
        if group.success and not expected_success:
            violations.append(
                Violation(
                    "sic_decoding",
                    f"group marked successful but SIC recomputation failed; "
                    f"gain={group.group_gain}, power={group.power}, "
                    f"noise={noise_power}, threshold={rate_threshold}",
                    slot,
                    int(group.satellite_id),
                    int(group.message_id),
                    int(group.group_id),
                )
            )
    return _finish(violations, strict)


def validate_time_config(cfg: V2Config, *, strict: bool = True) -> list[Violation]:
    violations: list[Violation] = []
    expected_handover = seconds_to_slots(cfg.handover_delay_s, cfg.slot_time_s)
    expected_isl = seconds_to_slots(cfg.isl_delay_s, cfg.slot_time_s)
    if abs(cfg.handover_delay_slots - expected_handover) > 1e-12:
        violations.append(Violation("time_unit", "handover delay conversion mismatch"))
    if abs(cfg.isl_delay_slots - expected_isl) > 1e-12:
        violations.append(Violation("time_unit", "ISL delay conversion mismatch"))
    if abs(milliseconds_to_slots(25.0, 0.01) - 2.5) > 1e-12:
        violations.append(Violation("time_unit", "25 ms / 10 ms must equal 2.5 slots"))
    return _finish(violations, strict)
