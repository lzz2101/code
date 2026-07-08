from __future__ import annotations

import numpy as np

from .config import SimConfig


def initialize_users(cfg: SimConfig, rng: np.random.Generator | None = None) -> list[dict]:
    """Create fixed ground UEs and assign multicast message IDs."""
    rng = rng or np.random.default_rng(cfg.random_seed)
    half_area = cfg.service_area / 2.0
    users: list[dict] = []

    for user_id in range(cfg.N_total):
        message_id = user_id // cfg.N_per_msg
        users.append(
            {
                "id": user_id,
                "x": float(rng.uniform(-half_area, half_area)),
                "y": float(rng.uniform(-half_area, half_area)),
                "message_id": message_id,
                "aoi": 1.0,
                "serving_sat": None,
                "prev_serving_sat": None,
                "handover": 0,
                "scheduled": False,
                "success": False,
            }
        )

    return users


def update_satellite_positions(t: int, cfg: SimConfig) -> np.ndarray:
    """Return satellite positions as an array with columns x, y, z."""
    positions = np.zeros((cfg.L, 3), dtype=float)

    if cfg.L >= 1:
        positions[0] = [-800e3 + cfg.satellite_speed * t, 0.0, cfg.H]
    if cfg.L >= 2:
        positions[1] = [800e3 - cfg.satellite_speed * t, 200e3, cfg.H]

    for sat_id in range(2, cfg.L):
        direction = 1.0 if sat_id % 2 == 0 else -1.0
        start_x = -800e3 if direction > 0 else 800e3
        y_offset = (sat_id - 1) * 200e3
        positions[sat_id] = [
            start_x + direction * cfg.satellite_speed * t,
            y_offset,
            cfg.H,
        ]

    return positions


def compute_distance_and_elevation(
    users: list[dict], sat_positions: np.ndarray, cfg: SimConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute distance, elevation angle, and coverage for each satellite-UE pair."""
    ue_xy = np.array([[user["x"], user["y"]] for user in users], dtype=float)
    sat_xy = sat_positions[:, :2]
    sat_height = sat_positions[:, 2][:, None]

    delta_xy = sat_xy[:, None, :] - ue_xy[None, :, :]
    horizontal_distance = np.linalg.norm(delta_xy, axis=2)
    distance = np.sqrt(horizontal_distance**2 + sat_height**2)

    elevation = np.degrees(np.arcsin(np.clip(sat_height / distance, 0.0, 1.0)))
    coverage = elevation >= cfg.phi_min
    return distance, elevation, coverage


def assign_serving_satellite(users: list[dict], elevation: np.ndarray, cfg: SimConfig) -> int:
    """Assign the highest-elevation covered satellite and detect handovers."""
    handover_count = 0

    for user in users:
        user_id = user["id"]
        candidate_sats = np.flatnonzero(elevation[:, user_id] >= cfg.phi_min)
        new_serving_sat = None
        if candidate_sats.size > 0:
            best_idx = int(np.argmax(elevation[candidate_sats, user_id]))
            new_serving_sat = int(candidate_sats[best_idx])

        previous = user["serving_sat"]
        handover = int(
            previous is not None
            and new_serving_sat is not None
            and previous != new_serving_sat
        )

        user["prev_serving_sat"] = previous
        user["serving_sat"] = new_serving_sat
        user["handover"] = handover
        handover_count += handover

    return handover_count
