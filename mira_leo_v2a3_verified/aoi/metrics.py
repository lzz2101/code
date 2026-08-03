from __future__ import annotations

import numpy as np

from math import isfinite

from mira_leo_v2a3_verified.algorithms.regrouping import Group
from mira_leo_v2a3_verified.config import V2Config


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
    reward_min: float | None = None,
    reward_max: float | None = None,
) -> float:
    del total_groups, num_users, aoi_norm_factor
    details = compute_regrouping_reward_details(
        success_count=success_count,
        weighted_sum_aoi=weighted_sum_aoi,
        edge_count=edge_count,
        B=B,
        normalize=normalize,
        reward_min=reward_min,
        reward_max=reward_max,
    )
    return float(details["normalized_reward"])


def normalize_reward_fixed(
    raw_reward: float,
    reward_min: float,
    reward_max: float,
    tolerance: float = 1e-9,
) -> float:
    """Map fixed raw-reward bounds to [0,1] without run-time min-max state."""
    values = (raw_reward, reward_min, reward_max, tolerance)
    if not all(isfinite(float(value)) for value in values):
        raise ValueError("reward and normalization bounds must be finite")
    if reward_max <= reward_min:
        raise ValueError("reward_max must be greater than reward_min")
    if raw_reward < reward_min - tolerance or raw_reward > reward_max + tolerance:
        raise ValueError(
            f"raw reward outside fixed bounds: raw={raw_reward}, "
            f"min={reward_min}, max={reward_max}"
        )

    bounded_raw = min(reward_max, max(reward_min, raw_reward))
    return float((bounded_raw - reward_min) / (reward_max - reward_min))


def compute_regrouping_reward_details(
    *,
    success_count: int,
    weighted_sum_aoi: float,
    edge_count: int,
    B: float,
    normalize: bool,
    reward_min: float | None,
    reward_max: float | None,
) -> dict[str, object]:
    """Return the paper-direction raw objective and its fixed normalization."""
    if success_count < 0 or edge_count < 0:
        raise ValueError("success_count and edge_count must be non-negative")
    if not isfinite(weighted_sum_aoi) or weighted_sum_aoi < 0.0:
        raise ValueError("weighted_sum_aoi must be finite and non-negative")
    if not isfinite(B) or B < 0.0:
        raise ValueError("B must be finite and non-negative")

    raw_reward = float(success_count - weighted_sum_aoi - B * edge_count)
    if normalize:
        if reward_min is None or reward_max is None:
            raise ValueError("fixed reward_min and reward_max are required")
        normalized_reward = normalize_reward_fixed(
            raw_reward, reward_min, reward_max
        )
        parameters = {
            "method": "fixed_linear_implementation_choice",
            "reward_min": float(reward_min),
            "reward_max": float(reward_max),
        }
    else:
        normalized_reward = raw_reward
        parameters = {"method": "none"}

    return {
        "raw_reward": raw_reward,
        "normalized_reward": float(normalized_reward),
        "normalization_parameters": parameters,
    }


def update_user_aoi(
    users: list[dict],
    groups: list[Group],
    distance: np.ndarray,
    cfg: V2Config,
    slot: int | None = None,
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
            prop_delay_slots = float(
                distance[sat_id, user_id] / cfg.c / cfg.slot_time_s
            )
            isl_hops = max(0, int(user.get("isl_hops", 0)))
            user["aoi"] = float(
                cfg.processing_delay_slots
                + prop_delay_slots
                + cfg.isl_delay_slots * isl_hops
            )
        else:
            handover_penalty = 0.0
            if bool(user.get("handover", 0)):
                event_id = user.get("handover_event_id", slot)
                if event_id != user.get("_last_charged_handover_event"):
                    handover_penalty = cfg.handover_delay_slots
                    user["_last_charged_handover_event"] = event_id
            user["aoi"] = float(user["aoi"] + 1.0 + handover_penalty)
