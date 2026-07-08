from __future__ import annotations

from .regrouping import Group


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
