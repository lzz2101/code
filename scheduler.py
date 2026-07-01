from __future__ import annotations

from .config import SimConfig


def schedule_top_aoi_groups(groups: list[dict], cfg: SimConfig) -> list[dict]:
    """Schedule the highest-priority groups for one satellite in one slot."""
    for group in groups:
        edge_bonus = cfg.edge_bonus if group["is_edge"] else 0.0
        group["priority"] = float(group["group_aoi"] + edge_bonus)
        group["scheduled"] = False

    ordered = sorted(groups, key=lambda group: group["priority"], reverse=True)
    scheduled_groups = ordered[: cfg.max_scheduled_groups]

    for group in scheduled_groups:
        group["scheduled"] = True

    return scheduled_groups
