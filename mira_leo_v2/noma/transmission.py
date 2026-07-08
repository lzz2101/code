from __future__ import annotations

import numpy as np

from mira_leo_v2.algorithms.regrouping import Group


def transmit_noma(
    scheduled_groups: list[Group], noise_power: float, rate_threshold: float
) -> None:
    """Compute NOMA SINR, rate, and SIC success for one satellite."""
    ordered_groups = sorted(scheduled_groups, key=lambda group: group.group_gain)
    previous_success = True

    for idx, group in enumerate(ordered_groups):
        stronger_power = sum(item.power for item in ordered_groups[idx + 1 :])
        denominator = group.group_gain * stronger_power + noise_power
        sinr = (
            group.group_gain * group.power / denominator
            if denominator > 0.0
            else np.inf
        )
        group.sinr = float(sinr)
        group.rate = float(np.log2(1.0 + sinr))

        if previous_success and group.rate >= rate_threshold:
            group.success = True
        else:
            group.success = False
            previous_success = False
