from __future__ import annotations

import numpy as np

from .config import SimConfig


def fixed_grouping(
    users: list[dict],
    sat_id: int,
    message_id: int,
    channel_gain: np.ndarray,
    elevation: np.ndarray,
    cfg: SimConfig,
) -> list[dict]:
    """Group users by satellite, message, and ascending CSI."""
    candidates = [
        user
        for user in users
        if user["serving_sat"] == sat_id and user["message_id"] == message_id
    ]
    if not candidates:
        return []

    ordered_user_ids = [
        user["id"]
        for user in sorted(candidates, key=lambda item: channel_gain[sat_id, item["id"]])
    ]
    num_groups = min(cfg.fixed_group_num, len(ordered_user_ids))
    chunks = np.array_split(np.array(ordered_user_ids, dtype=int), num_groups)

    groups: list[dict] = []
    for group_idx, chunk in enumerate(chunks):
        user_ids = [int(user_id) for user_id in chunk.tolist()]
        gains = [float(channel_gain[sat_id, user_id]) for user_id in user_ids]
        max_elevation = float(np.max(elevation[sat_id, user_ids]))

        groups.append(
            {
                "id": f"s{sat_id}_m{message_id}_g{group_idx}",
                "sat_id": sat_id,
                "message_id": message_id,
                "rank": group_idx,
                "user_ids": user_ids,
                "group_gain": float(min(gains)),
                "max_elevation": max_elevation,
                "is_edge": False,
                "group_aoi": float(sum(users[user_id]["aoi"] for user_id in user_ids)),
                "priority": 0.0,
                "scheduled": False,
                "power": 0.0,
                "sinr": 0.0,
                "rate": 0.0,
                "success": False,
            }
        )

    return groups


def build_groups_for_satellite(
    users: list[dict],
    sat_id: int,
    channel_gain: np.ndarray,
    elevation: np.ndarray,
    cfg: SimConfig,
) -> list[dict]:
    groups: list[dict] = []
    for message_id in range(cfg.M):
        groups.extend(
            fixed_grouping(users, sat_id, message_id, channel_gain, elevation, cfg)
        )
    return groups


def classify_edge_groups(groups: list[dict], elevation: np.ndarray, cfg: SimConfig) -> None:
    """Mark groups whose strongest elevation angle is still below phi_theta."""
    for group in groups:
        user_ids = group["user_ids"]
        sat_id = group["sat_id"]
        max_phi = float(np.max(elevation[sat_id, user_ids])) if user_ids else -np.inf
        group["max_elevation"] = max_phi
        group["is_edge"] = bool(max_phi < cfg.phi_theta)
