from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .errors import ModelConsistencyError, OptionalDependencyError
from .models import Group, GroupPartitionEntry
from .protected_power import POWER_MODE, classify_power_pools, allocate_csi_expanded_protected_power


def legal_actions(subscribers_per_message: int) -> tuple[int, ...]:
    return tuple(
        value
        for value in range(1, subscribers_per_message + 1)
        if subscribers_per_message % value == 0
    )


def build_group_partition(
    config, satellite_index: int, action: int,
    message_members: dict[int, np.ndarray], channel_amplitude: np.ndarray,
) -> tuple[GroupPartitionEntry, ...]:
    """Sort subscribers once at an epoch boundary and freeze only membership."""
    subscribers = int(config.get("services.subscribers_per_message"))
    messages = int(config.get("services.messages_per_active_satellite"))
    if action <= 0 or subscribers % action != 0:
        raise ValueError("action is a group count and must evenly divide subscriber count")
    partition = []
    for message in range(messages):
        members = np.asarray(message_members[message], dtype=np.int32)
        if len(members) != subscribers:
            raise ValueError("subscription count does not match configured group cardinality")
        order = np.lexsort((members, channel_amplitude[satellite_index, members]))
        ordered = members[order]
        size = subscribers // action
        for group_id in range(action):
            partition.append(GroupPartitionEntry(
                message, group_id,
                tuple(int(d) for d in ordered[group_id * size:(group_id + 1) * size]),
            ))
    return tuple(partition)


def materialize_groups_from_partition(
    config, satellite_index: int, partition,
    channel_amplitude: np.ndarray, elevation_deg: np.ndarray,
) -> list[Group]:
    """Recompute worst-user CSI and elevation from this slot's arrays only."""
    phi = float(config.get("algorithm.phi_theta_deg"))
    groups = []
    for entry in partition:
        members = np.asarray(entry.member_indices, dtype=np.int32)
        amplitude = channel_amplitude[satellite_index, members]
        elevation = float(np.max(elevation_deg[satellite_index, members]))
        groups.append(Group(
            satellite_index=satellite_index,
            message_id=entry.message_id, group_id=entry.group_id,
            member_indices=members, gain=float(np.min(amplitude) ** 2),
            max_elevation_deg=elevation, is_edge=bool(elevation < phi),
        ))
    return groups


def make_groups(
    config, satellite_index: int, action: int,
    message_members: dict[int, np.ndarray], channel_amplitude: np.ndarray,
    elevation_deg: np.ndarray,
) -> list[Group]:
    """Legacy immediate-build API; the formal simulator holds the partition."""
    partition = build_group_partition(config, satellite_index, action,
                                      message_members, channel_amplitude)
    return materialize_groups_from_partition(config, satellite_index,
                                             partition, channel_amplitude, elevation_deg)


def schedule_capacity(power_budget: float, v_phi: float) -> int:
    if power_budget <= 0.0 or v_phi <= 0.0:
        raise ValueError("power_budget and v_phi must be positive")
    return math.floor(-0.5 + math.sqrt(0.25 + 2.0 * power_budget / v_phi))


def schedule_capacity_continuous(power_budget: float, v_phi: float) -> float:
    if power_budget <= 0.0 or v_phi <= 0.0:
        raise ValueError("power_budget and v_phi must be positive")
    return -0.5 + math.sqrt(0.25 + 2.0 * power_budget / v_phi)


def schedule_groups(
    config,
    groups: list[Group],
    ages_slots: np.ndarray,
    distance_km: np.ndarray,
    handover: bool,
    isl_delay_slots: float,
) -> list[Group]:
    weights = [float(value) for value in config.get("services.weights")]
    c_km_s = float(config.get("geometry.light_speed_km_s"))
    dt_s = float(config.get("experiment.slot_dt_s"))
    kh_slots = float(config.get("derived.handover_delay_slots"))
    delta_c = float(config.get("services.delta_c_slots"))
    for group in groups:
        members = group.member_indices
        current = float(np.sum(ages_slots[members]))
        unsuccessful = current + len(members) * (1.0 + kh_slots * float(handover))
        propagation = float(np.sum(distance_km[group.satellite_index, members] / (c_km_s * dt_s)))
        successful = len(members) * (delta_c + isl_delay_slots) + propagation
        gamma = unsuccessful - successful
        group.priority = weights[group.message_id] * gamma
        group.metadata.update(
            gamma=float(gamma),
            pre_group_aoi_sum_slots=current,
            predicted_unsuccessful_next_aoi_sum_slots=unsuccessful,
            predicted_successful_next_aoi_sum_slots=successful,
            gamma_slots=float(gamma),
            message_weight=weights[group.message_id],
            weighted_pre_group_aoi_slots=weights[group.message_id] * current,
            weighted_predicted_unsuccessful_aoi_slots=weights[group.message_id] * unsuccessful,
            weighted_predicted_successful_aoi_slots=weights[group.message_id] * successful,
            propagation_delay_slots=propagation,
            isl_delay_slots=float(isl_delay_slots),
            handover=bool(handover),
        )
        group.scheduled = False
        group.power = 0.0
        group.metadata.update(
            scheduler_candidate=bool(group.is_edge or group.priority > 0.0),
            scheduler_rejection_reason="NONPOSITIVE_PRIORITY" if not group.is_edge else "ALGORITHM2_CAP",
            scheduler_algorithm="Algorithm2_strict",
        )

    cap = int(config.get("derived.schedule_capacity_single_class_floor_only"))
    order_key = lambda item: (-item.priority, item.message_id, item.group_id)
    edge = [group for group in groups if group.is_edge]
    nonedge = [group for group in groups if not group.is_edge]
    if len(edge) > cap:
        selected = sorted(edge, key=order_key)[:cap]
    else:
        selected = list(edge)
        ranked = sorted((group for group in nonedge if group.priority > 0.0), key=order_key)
        selected.extend(ranked[:max(0, cap - len(edge))])
    selected_ids = {id(group) for group in selected}
    for group in groups:
        if id(group) in selected_ids:
            group.scheduled = True
            group.metadata["scheduler_rejection_reason"] = "SCHEDULED"
        elif group.is_edge or group.priority > 0.0:
            group.metadata["scheduler_rejection_reason"] = "ALGORITHM2_CAP"
        group.metadata["algorithm2_U"] = cap
    return selected


def _group_order(group: Group) -> tuple[float, int, int]:
    return group.gain, group.message_id, group.group_id


def _budgets(groups: list[Group], p_sat: float, v_c: float, v_nc: float) -> tuple[float, float]:
    edge_count = sum(group.is_edge for group in groups)
    nonedge_count = len(groups) - edge_count
    if edge_count and nonedge_count:
        return v_c * p_sat, v_nc * p_sat
    if edge_count:
        return p_sat, 0.0
    if nonedge_count:
        return 0.0, p_sat
    return 0.0, 0.0


def _floor_by_group(subset: list[Group], v_phi: float) -> dict[int, float]:
    ordered = sorted(subset, key=_group_order)
    return {
        id(group): float(v_phi * (len(ordered) - rank))
        for rank, group in enumerate(ordered)
    }


def _literal_target(subset: list[Group], budget: float, v_phi: float) -> tuple[dict[int, float], float]:
    floors = _floor_by_group(subset, v_phi)
    floor_total = float(sum(floors.values()))
    if not subset:
        return {}, floor_total
    if floor_total > budget + 1e-12:
        return {}, floor_total
    residual_share = (budget - floor_total) / len(subset)
    return {id(group): floors[id(group)] + residual_share for group in subset}, floor_total


def _sic_order_violations(
    groups: list[Group], tolerance: float = 1e-12, powers: dict[int, float] | None = None
) -> list[dict]:
    ordered = sorted(groups, key=_group_order)
    violations = []
    for weaker, stronger in zip(ordered, ordered[1:]):
        weaker_power = weaker.power if powers is None else powers.get(id(weaker), 0.0)
        stronger_power = stronger.power if powers is None else powers.get(id(stronger), 0.0)
        magnitude = float(stronger_power - weaker_power)
        if magnitude > tolerance:
            violations.append(
                {
                    "weaker": weaker.key,
                    "stronger": stronger.key,
                    "weaker_gain": weaker.gain,
                    "stronger_gain": stronger.gain,
                    "weaker_power": weaker_power,
                    "stronger_power": stronger_power,
                    "magnitude": magnitude,
                }
            )
    return violations


def twc_aim_class_budget(groups, p_sat, v_c, v_nc):
    return _budgets(groups, p_sat, v_c, v_nc)


def twc_aim_staircase_target(subset, budget, v_phi):
    """Paper Algorithm 3: current weak-to-strong floors plus float residual."""
    return _literal_target(subset, budget, v_phi)


def _evaluate_staircase_plan(config, scheduled_groups, *, mode, apply=False):
    p_sat = float(config.get("channel.normalized_satellite_power"))
    v_c, v_nc = float(config.get("algorithm.V_c")), float(config.get("algorithm.V_nc"))
    v_phi = float(config.get("algorithm.V_phi"))
    scheduled = list(scheduled_groups)
    edge = [g for g in scheduled if g.is_edge]
    nonedge = [g for g in scheduled if not g.is_edge]
    edge_budget, nonedge_budget = twc_aim_class_budget(scheduled, p_sat, v_c, v_nc)
    edge_target, edge_floor = twc_aim_staircase_target(edge, edge_budget, v_phi)
    nonedge_target, nonedge_floor = twc_aim_staircase_target(nonedge, nonedge_budget, v_phi)
    floor_violations = []
    for label, subset, target, total, budget in (
        ("edge", edge, edge_target, edge_floor, edge_budget),
        ("nonedge", nonedge, nonedge_target, nonedge_floor, nonedge_budget),
    ):
        if subset and not target:
            floor_violations.append({"class": label, "floor_total": total, "budget": budget})
    target = {id(g): float((edge_target | nonedge_target).get(id(g), 0.0)) for g in scheduled}
    powers = dict(target)
    floors = _floor_by_group(edge, v_phi) | _floor_by_group(nonedge, v_phi)
    before = _sic_order_violations(scheduled, powers=target) if not floor_violations else []
    after = list(before)
    projection_used = bool(before and not floor_violations and mode != "literal_with_checks")
    projection_infeasible = False
    solver_diagnostic = {}
    if projection_used:
        projected = _projection_lp(scheduled, target, floors, (edge_budget, nonedge_budget), diagnostic=solver_diagnostic)
        projection_infeasible = projected is None
        if projected is not None:
            powers = {id(g): float(p) for g, p in zip(scheduled, projected)}
            after = _sic_order_violations(scheduled, tolerance=1e-10, powers=powers)
    total_power = float(sum(powers.values()))
    if not floor_violations:
        for group in scheduled:
            if powers[id(group)] < floors[id(group)] - 1e-10:
                floor_violations.append({"class": "edge" if group.is_edge else "nonedge",
                                         "group": group.key, "power": powers[id(group)], "floor": floors[id(group)]})
    budget_violations = []
    if not floor_violations:
        for label, subset, budget in (("edge", edge, edge_budget), ("nonedge", nonedge, nonedge_budget)):
            allocated = sum(powers[id(group)] for group in subset)
            if abs(allocated - budget) > 1e-10:
                budget_violations.append({"class": label, "power": allocated, "budget": budget})
    valid = bool(not floor_violations and not after and not budget_violations
                 and not projection_infeasible and total_power <= p_sat + 1e-10
                 and all(math.isfinite(p) and p >= 0 for p in powers.values()))
    if apply and (valid or mode == "literal_with_checks"):
        for group in scheduled:
            group.power = powers[id(group)]
    status = ("SIC_POWER_PROJECTED" if projection_used else "OK") if valid else (
        "MODEL_CONSISTENCY_ERROR" if mode == "twc_aim_staircase" else "INVALID_POWER_OR_SIC")
    return {
        "mode": mode, "valid": valid, "status": status,
        "edge_count": len(edge), "nonedge_count": len(nonedge),
        "edge_budget": edge_budget, "nonedge_budget": nonedge_budget,
        "edge_floor_total": edge_floor, "nonedge_floor_total": nonedge_floor,
        "edge_residual": edge_budget - edge_floor, "nonedge_residual": nonedge_budget - nonedge_floor,
        "target_powers": target, "final_powers": powers, "powers": powers, "floors": floors,
        "sic_projection_used": projection_used, "projection_infeasible": projection_infeasible,
        "projection_solver_status": solver_diagnostic.get("status"),
        "projection_solver_message": solver_diagnostic.get("message"),
        "sic_order_violations_before": before, "sic_order_violations_after": after,
        "sic_order_violations": after, "floor_violations": floor_violations,
        "budget_violations": budget_violations, "total_power": total_power,
        "removed_groups": [],
    }


def evaluate_twc_aim_power_plan(config, scheduled_groups, *, apply=False):
    """TWC/AIM target plus power-only global-SIC repair; schedule is immutable."""
    return _evaluate_staircase_plan(config, scheduled_groups, mode="twc_aim_staircase", apply=apply)


def evaluate_power_plan(config, scheduled_groups: list[Group], *, apply: bool = False) -> dict:
    mode = str(config.get("algorithm.power_mode"))
    if mode == POWER_MODE:
        return allocate_csi_expanded_protected_power(config, scheduled_groups, apply=apply)
    if mode == "twc_aim_staircase":
        return evaluate_twc_aim_power_plan(config, scheduled_groups, apply=apply)
    if mode in {"literal_with_checks", "constraint_projection"}:
        return _evaluate_staircase_plan(config, scheduled_groups, mode=mode, apply=apply)
    raise ValueError(f"unknown power mode: {mode}")


def check_power_feasibility(config, scheduled_groups: list[Group]) -> dict:
    """Read-only allocator diagnostic; never consulted by Algorithm 2."""
    return evaluate_power_plan(config, scheduled_groups)


def _projection_lp(
    scheduled: list[Group], target: dict[int, float], floors: dict[int, float], budgets: tuple[float, float],
    *, diagnostic: dict | None = None,
) -> np.ndarray | None:
    try:
        from scipy.optimize import linprog  # type: ignore
    except ImportError as exc:
        raise OptionalDependencyError(
            "constraint_projection requires scipy; install requirements.txt"
        ) from exc
    count = len(scheduled)
    ordered = sorted(range(count), key=lambda index: _group_order(scheduled[index]))
    objective = np.r_[np.zeros(count), np.ones(count)]
    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []
    for left, right in zip(ordered, ordered[1:]):
        row = np.zeros(2 * count)
        row[right] = 1.0
        row[left] = -1.0
        a_ub.append(row)
        b_ub.append(0.0)
    for index, group in enumerate(scheduled):
        row_positive = np.zeros(2 * count)
        row_positive[index] = 1.0
        row_positive[count + index] = -1.0
        a_ub.append(row_positive)
        b_ub.append(float(target[id(group)]))
        row_negative = np.zeros(2 * count)
        row_negative[index] = -1.0
        row_negative[count + index] = -1.0
        a_ub.append(row_negative)
        b_ub.append(-float(target[id(group)]))
    a_eq: list[np.ndarray] = []
    b_eq: list[float] = []
    for is_edge, budget in ((True, budgets[0]), (False, budgets[1])):
        indices = [index for index, group in enumerate(scheduled) if group.is_edge == is_edge]
        if indices:
            row = np.zeros(2 * count)
            row[indices] = 1.0
            a_eq.append(row)
            b_eq.append(float(budget))
    result = linprog(
        objective,
        A_ub=np.asarray(a_ub) if a_ub else None,
        b_ub=np.asarray(b_ub) if b_ub else None,
        A_eq=np.asarray(a_eq) if a_eq else None,
        b_eq=np.asarray(b_eq) if b_eq else None,
        bounds=[(floors[id(group)], None) for group in scheduled] + [(0.0, None)] * count,
        method="highs",
    )
    if diagnostic is not None:
        diagnostic.update(status=int(result.status), message=str(result.message))
    return np.asarray(result.x[:count], dtype=float) if result.success else None


def allocate_power(config, groups: list[Group]) -> dict:
    """Keep Algorithm 2's schedule; a proven infeasible LP can mute this slot."""
    for group in groups:
        group.power = 0.0
        group.metadata["power_failure_no_transmission"] = False
    scheduled = [group for group in groups if group.scheduled]
    diagnostic = evaluate_power_plan(config, scheduled, apply=True)
    if diagnostic["mode"] == POWER_MODE:
        return diagnostic
    failure = bool(
        not diagnostic["valid"]
        and diagnostic["mode"] == "twc_aim_staircase"
        and config.get("algorithm.infeasible_power_handling", "error") == "transmission_failure"
        and diagnostic["projection_solver_status"] == 2
        and not diagnostic["floor_violations"]
    )
    diagnostic["transmission_failure"] = failure
    diagnostic["transmission_attempted"] = bool(diagnostic["valid"] and scheduled)
    if failure:
        for group in groups:
            group.power = 0.0
            group.success = group.rate_pass = group.sic_pass = False
            group.sinr = group.rate_bps_hz = group.interference = 0.0
            group.metadata["power_failure_no_transmission"] = bool(group.scheduled)
        diagnostic["status"] = "POWER_INFEASIBLE_NO_TRANSMISSION"
    diagnostic["actual_total_power"] = float(sum(g.power for g in scheduled))
    return diagnostic


def rate_passes_threshold(rate_bps_hz: float, threshold_bps_hz: float) -> bool:
    return bool(float(rate_bps_hz) >= float(threshold_bps_hz))


def decode_groups(config, groups: list[Group]) -> dict:
    scheduled = sorted((group for group in groups if group.scheduled), key=_group_order)
    noise = float(config.get("derived.noise_power"))
    threshold = float(config.get("algorithm.R_theta"))
    chain_ok = True
    first_failed_group = None
    for index, group in enumerate(scheduled):
        later_power = float(sum(item.power for item in scheduled[index + 1 :]))
        denominator = group.gain * later_power + group.interference + noise
        group.sinr = group.gain * group.power / denominator if denominator > 0.0 else math.inf
        group.rate_bps_hz = math.log2(1.0 + group.sinr)
        group.rate_pass = rate_passes_threshold(group.rate_bps_hz, threshold)
        group.metadata["sic_first_failed_predecessor_message_id"] = None
        group.metadata["sic_first_failed_predecessor_group_id"] = None
        group.metadata["sic_first_failed_predecessor_gain_db"] = None
        group.metadata["sic_first_failed_predecessor_power"] = None
        group.metadata["sic_first_failed_predecessor_rate"] = None
        if first_failed_group is not None and group.rate_pass:
            source = first_failed_group
            group.metadata.update(
                sic_first_failed_predecessor_message_id=source.message_id,
                sic_first_failed_predecessor_group_id=source.group_id,
                sic_first_failed_predecessor_gain_db=10.0 * math.log10(max(source.gain, np.finfo(float).tiny)),
                sic_first_failed_predecessor_power=source.power,
                sic_first_failed_predecessor_rate=source.rate_bps_hz,
            )
        if first_failed_group is None and not group.rate_pass:
            first_failed_group = group
        chain_ok = chain_ok and group.rate_pass
        group.sic_pass = chain_ok
        group.success = chain_ok
        group.metadata["rate_bps"] = float(config.get("services.bandwidth_hz")) * group.rate_bps_hz
    return {
        "scheduled_count": len(scheduled),
        "success_count": sum(group.success for group in scheduled),
        "rate_failure_count": sum(not group.rate_pass for group in scheduled),
        "sic_failure_count": sum(not group.sic_pass for group in scheduled),
    }


@dataclass(frozen=True)
class BanditObservation:
    slot: int
    arm: int
    reward: float


class SlidingWindowUCB:
    def __init__(
        self, actions: Iterable[int], window_slots: int | None = None,
        xi: float = math.sqrt(0.5), satellite_id: int = 0, *, window_steps: int | None = None,
    ):
        if window_steps is None:
            window_steps = window_slots
        elif window_slots is not None and window_slots != window_steps:
            raise ValueError("window_slots and window_steps disagree")
        self.actions = tuple(int(value) for value in actions)
        if not self.actions or window_steps is None or window_steps <= 0:
            raise ValueError("actions and positive window_slots are required")
        self.window_steps = int(window_steps)
        self.xi = float(xi)
        self.satellite_id = int(satellite_id)
        self.observations: deque[BanditObservation] = deque()

    @property
    def window_slots(self):
        """Legacy alias; formal simulation steps are global regrouping epochs."""
        return self.window_steps

    def _prune(self, slot: int) -> None:
        start = max(0, int(slot) - self.window_steps)
        while self.observations and self.observations[0].slot < start:
            self.observations.popleft()

    def select(self, slot: int) -> int:
        self._prune(slot)
        prior = [record for record in self.observations if record.slot < slot]
        log_term = math.log(max(1, min(int(slot), self.window_steps)))
        best_action = self.actions[0]
        best_index = -math.inf
        for action in self.actions:
            rewards = [record.reward for record in prior if record.arm == action]
            if not rewards:
                return action
            index = float(np.mean(rewards)) + self.xi * math.sqrt(log_term / len(rewards))
            if index > best_index:
                best_index = index
                best_action = action
        return best_action

    def update(self, slot: int, arm: int, reward: float) -> None:
        if arm not in self.actions:
            raise ValueError("unknown arm")
        if self.observations and slot <= self.observations[-1].slot:
            raise ValueError("observations must use increasing global slots")
        if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
            raise ValueError("bandit reward must be finite and in [0, 1]")
        self._prune(slot + 1)
        self.observations.append(BanditObservation(int(slot), int(arm), float(reward)))


class RewardNormalizer:
    def __init__(self, lower: float, upper: float, identity: str):
        if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
            raise ValueError("reward bounds must be finite and increasing")
        self.lower = float(lower)
        self.upper = float(upper)
        self.identity = str(identity)
        self.total = 0
        self.clipped = 0

    def transform(self, raw_reward: float) -> float:
        if not math.isfinite(raw_reward):
            raise ValueError("raw reward must be finite")
        value = (raw_reward - self.lower) / (self.upper - self.lower)
        self.total += 1
        if value < 0.0 or value > 1.0:
            self.clipped += 1
        return float(np.clip(value, 0.0, 1.0))

    @property
    def clipping_rate(self) -> float:
        return self.clipped / self.total if self.total else 0.0


def finite_horizon_reward_bounds(config) -> tuple[float, float]:
    messages = int(config.get("services.messages_per_active_satellite"))
    subscribers = int(config.get("services.subscribers_per_message"))
    weights = np.asarray(config.get("services.weights"), dtype=float)
    maximum_age = float(config.get("services.initial_age_slots")) + config.slot_count * (
        1.0 + float(config.get("derived.handover_delay_slots"))
    )
    weighted_age = subscribers * float(np.sum(weights)) * maximum_age
    maximum_groups = messages * max(int(value) for value in config.get("algorithm.action_set"))
    penalty = float(config.get("algorithm.B_penalty")) * messages * subscribers * maximum_groups
    return -(weighted_age + penalty), float(maximum_groups)


def pilot_fixed_bounds(raw_rewards: np.ndarray, lower_q: float = 0.001, upper_q: float = 0.999, margin: float = 0.25) -> tuple[float, float]:
    values = np.asarray(raw_rewards, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("pilot reward calibration requires finite samples")
    lower = float(np.quantile(values, lower_q))
    upper = float(np.quantile(values, upper_q))
    spread = upper - lower
    if spread <= 0.0:
        spread = max(1.0, abs(lower) * 0.1)
    return lower - margin * spread, upper + margin * spread


def raw_reward(
    success_group_count: int,
    weighted_post_update_aoi: float,
    b_penalty: float,
    messages: int,
    subscribers_per_message: int,
    edge_group_count: int,
) -> float:
    return float(
        success_group_count
        - weighted_post_update_aoi
        - b_penalty * messages * subscribers_per_message * edge_group_count
    )


def update_user_aoi(
    current_age_slots: float,
    *,
    successful: bool,
    delta_c_slots: float,
    propagation_delay_slots: float,
    isl_delay_slots: float,
    handover_event: bool,
    handover_delay_slots: float,
    handover_mode: str = "paper_additive",
) -> float:
    """One-UE AoI recursion with mutually exclusive handover models."""
    if successful:
        return float(delta_c_slots + propagation_delay_slots + isl_delay_slots)
    if handover_mode == "paper_additive":
        return float(current_age_slots + 1.0 + handover_delay_slots * float(handover_event))
    if handover_mode == "physical_blackout":
        return float(current_age_slots + 1.0)
    raise ValueError("unknown handover mode")
