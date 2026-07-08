from __future__ import annotations

from .regrouping import Group


def power_allocation_placeholder(
    scheduled_groups: list[Group], P_sat: float = 1.0
) -> None:
    """Algorithm 3 placeholder: equal power allocation."""
    if not scheduled_groups:
        return

    power = float(P_sat / len(scheduled_groups))
    for group in scheduled_groups:
        group.power = power
