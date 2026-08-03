from __future__ import annotations

from math import floor, sqrt

import numpy as np

from .regrouping import Group
from mira_leo_v2a3_verified.config import V2Config


def _message_weight(cfg: V2Config, message_id: int) -> float:
    """Return message-level weight w_m."""
    if len(cfg.message_weight) == cfg.M:
        return float(cfg.message_weight[message_id])
    return 1.0 / cfg.M


def compute_schedulable_group_upper_bound(cfg: V2Config) -> int:
    """
    Theorem 3 scheduling capacity upper bound.

    U_l <= -1/2 + sqrt(1/4 + 2 P_l / V_phi)
    """
    if cfg.V_phi <= 0.0:
        return int(cfg.max_scheduled_groups)

    theorem_u = floor(-0.5 + sqrt(0.25 + 2.0 * cfg.P_sat / cfg.V_phi))
    theorem_u = max(1, theorem_u)

    if cfg.use_theorem3_capacity:
        return int(min(theorem_u, cfg.max_scheduled_groups))
    return int(cfg.max_scheduled_groups)


def compute_group_handover(group: Group) -> float:
    """Group-level handover indicator: one handover user makes the group sensitive."""
    if not group.users:
        return 0.0
    return float(max(float(user.get("handover", 0.0)) for user in group.users))


def compute_group_current_aoi(group: Group) -> float:
    """Group sum-AoI before current-slot scheduling."""
    return float(sum(float(user.get("aoi", 0.0)) for user in group.users))


def compute_group_propagation_delay_slots(
    group: Group, distance: np.ndarray, cfg: V2Config
) -> float:
    """Sum propagation delays K_p,u in slot units."""
    total_delay = 0.0
    sat_id = int(group.satellite_id)

    for user in group.users:
        user_id = int(user["id"])
        total_delay += float(distance[sat_id, user_id] / cfg.c / cfg.slot_time_s)

    return float(total_delay)


def compute_handover_aware_priority(
    group: Group, distance: np.ndarray, cfg: V2Config
) -> float:
    """
    Compute weighted one-slot marginal AoI reduction for Algorithm 2.

    priority = w_m * (handover-corrected unscheduled AoI - successful-update AoI)
    """
    group_size = len(group.users)
    if group_size == 0:
        group.priority = 0.0
        group.metadata["gamma"] = 0.0
        return 0.0

    group_handover = compute_group_handover(group)
    current_aoi = compute_group_current_aoi(group)
    unscheduled_aoi = current_aoi + group_size * (
        1.0 + cfg.handover_delay_slots * group_handover
    )
    propagation_delay = compute_group_propagation_delay_slots(group, distance, cfg)
    isl_delay = sum(
        cfg.isl_delay_slots * max(0, int(user.get("isl_hops", 0)))
        for user in group.users
    )
    success_aoi = (
        group_size * cfg.processing_delay_slots + propagation_delay + isl_delay
    )

    gamma = unscheduled_aoi - success_aoi
    weight = _message_weight(cfg, int(group.message_id))
    priority = weight * gamma

    group.priority = float(priority)
    group.metadata["current_aoi"] = float(current_aoi)
    group.metadata["group_handover"] = float(group_handover)
    group.metadata["unscheduled_aoi"] = float(unscheduled_aoi)
    group.metadata["success_aoi"] = float(success_aoi)
    group.metadata["propagation_delay"] = float(propagation_delay)
    group.metadata["isl_delay"] = float(isl_delay)
    group.metadata["gamma"] = float(gamma)
    group.metadata["weight"] = float(weight)
    group.metadata["weighted_gamma"] = float(priority)

    return float(priority)


def handover_aware_scheduler(
    groups: list[Group], cfg: V2Config, distance: np.ndarray
) -> list[Group]:
    """
    Algorithm 2: MIRA-LEO Handover-Aware Scheduling.

    Edge groups are protected first. If edge groups do not fill capacity, the
    remaining capacity is assigned to positive-priority non-edge groups.
    """
    U_l = compute_schedulable_group_upper_bound(cfg)

    for group in groups:
        group.scheduled = False
        group.metadata["scheduler"] = "handover_aware"
        group.metadata["U_l"] = float(U_l)
        compute_handover_aware_priority(group, distance, cfg)

    edge_groups = [group for group in groups if group.is_edge]
    nonedge_groups = [group for group in groups if not group.is_edge]

    if len(edge_groups) > U_l:
        scheduled = sorted(edge_groups, key=lambda group: group.priority, reverse=True)[
            :U_l
        ]
    else:
        scheduled = list(edge_groups)
        residual_capacity = U_l - len(edge_groups)
        ranked_nonedge = sorted(
            nonedge_groups, key=lambda group: group.priority, reverse=True
        )
        positive_nonedge = [
            group for group in ranked_nonedge if group.priority > 0.0
        ]
        scheduled.extend(positive_nonedge[:residual_capacity])

    for group in scheduled:
        group.scheduled = True

    return scheduled


def scheduler_placeholder(
    groups: list[Group], max_scheduled_groups: int = 8
) -> list[Group]:
    """Algorithm 2 placeholder: schedule groups with highest current AoI."""
    for group in groups:
        group.priority = float(sum(user["aoi"] for user in group.users))
        group.scheduled = False

    scheduled = sorted(groups, key=lambda group: group.priority, reverse=True)[
        :max_scheduled_groups
    ]
    for group in scheduled:
        group.scheduled = True

    return scheduled
