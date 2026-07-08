from __future__ import annotations

import numpy as np

from mira_leo_v2.algorithms.regrouping import Group
from mira_leo_v2.config import V2Config


def message_weight(cfg: V2Config, message_id: int) -> float:
    if len(cfg.message_weight) == cfg.M:
        return float(cfg.message_weight[message_id])
    return 1.0 / cfg.M


def compute_weighted_sum_aoi(
    users: list[dict], cfg: V2Config, satellite_id: int | None = None
) -> float:
    total = 0.0
    for user in users:
        if satellite_id is not None and user["serving_sat"] != satellite_id:
            continue
        total += message_weight(cfg, user["message_id"]) * float(user["aoi"])
    return float(total)


def compute_edge_penalty(edge_count: int, B: float, M: int, N: int) -> float:
    return float(B * M * N * edge_count)


def compute_regrouping_reward(
    success_count: int,
    weighted_sum_aoi: float,
    edge_count: int,
    total_groups: int,
    num_users: int,
    B: float = 0.2,
    normalize: bool = True,
    aoi_norm_factor: float = 100.0,
) -> float:
    if normalize:
        success_term = success_count / max(1, total_groups)
        aoi_term = weighted_sum_aoi / max(1.0, num_users * aoi_norm_factor)
        edge_term = edge_count / max(1, total_groups)
        return float(success_term - aoi_term - B * edge_term)

    penalty = B * edge_count
    return float(success_count - weighted_sum_aoi - penalty)


def update_user_aoi(
    users: list[dict], groups: list[Group], distance: np.ndarray, cfg: V2Config
) -> None:
    scheduled_user_ids: set[int] = set()
    successful_user_ids: set[int] = set()

    for group in groups:
        if group.scheduled:
            scheduled_user_ids.update(group.user_ids)
        if group.scheduled and group.success:
            successful_user_ids.update(group.user_ids)

    for user in users:
        user_id = int(user["id"])
        user["scheduled"] = user_id in scheduled_user_ids
        user["success"] = user_id in successful_user_ids

        if user["success"] and user["serving_sat"] is not None:
            sat_id = int(user["serving_sat"])
            prop_delay_slot = float(distance[sat_id, user_id] / cfg.c / cfg.slot_time)
            user["aoi"] = float(cfg.Delta_c + prop_delay_slot)
        else:
            user["aoi"] = float(user["aoi"] + 1.0 + cfg.K_h * user["handover"])
