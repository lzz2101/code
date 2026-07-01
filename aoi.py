from __future__ import annotations

import numpy as np

from .config import SimConfig


def update_user_aoi(
    users: list[dict], groups: list[dict], distance: np.ndarray, cfg: SimConfig
) -> None:
    """Update UE AoI using scheduled-success status and handover penalty."""
    scheduled_user_ids: set[int] = set()
    successful_user_ids: set[int] = set()

    for group in groups:
        if group.get("scheduled", False):
            scheduled_user_ids.update(group["user_ids"])
        if group.get("scheduled", False) and group.get("success", False):
            successful_user_ids.update(group["user_ids"])

    for user in users:
        user_id = user["id"]
        user["scheduled"] = user_id in scheduled_user_ids
        user["success"] = user_id in successful_user_ids

        if user["success"] and user["serving_sat"] is not None:
            sat_id = int(user["serving_sat"])
            prop_delay_slot = float(distance[sat_id, user_id] / cfg.c / cfg.slot_time)
            user["aoi"] = float(cfg.Delta_c + prop_delay_slot)
        else:
            user["aoi"] = float(user["aoi"] + 1.0 + cfg.K_h * user["handover"])


def message_weight(cfg: SimConfig, message_id: int) -> float:
    if len(cfg.message_weight) == cfg.M:
        return float(cfg.message_weight[message_id])
    return 1.0 / cfg.M


def weighted_sum_aoi(users: list[dict], cfg: SimConfig) -> float:
    return float(
        sum(message_weight(cfg, user["message_id"]) * user["aoi"] for user in users)
    )
