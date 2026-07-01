from __future__ import annotations

import numpy as np

from .config import SimConfig


def sic_order(groups: list[dict]) -> list[dict]:
    """Return weak-to-strong channel order."""
    return sorted(groups, key=lambda group: group["group_gain"])


def allocate_power(
    scheduled_groups: list[dict], cfg: SimConfig, mode: str | None = None
) -> None:
    """Allocate satellite power to scheduled groups."""
    mode = mode or cfg.power_mode
    ordered_groups = sic_order(scheduled_groups)
    num_groups = len(ordered_groups)
    if num_groups == 0:
        return

    if mode == "equal":
        for group in ordered_groups:
            group["power"] = float(cfg.P_sat / num_groups)
        return

    if mode == "ordered":
        weights = np.arange(num_groups, 0, -1, dtype=float)
        weight_sum = float(np.sum(weights))
        for group, weight in zip(ordered_groups, weights):
            group["power"] = float(cfg.P_sat * weight / weight_sum)
        return

    raise ValueError(f"Unknown power allocation mode: {mode}")


def compute_noma_success(scheduled_groups: list[dict], cfg: SimConfig) -> None:
    """Compute SINR, rate, and SIC failure propagation for scheduled groups."""
    ordered_groups = sic_order(scheduled_groups)
    previous_success = True

    for idx, group in enumerate(ordered_groups):
        group_gain = float(group["group_gain"])
        power = float(group["power"])
        stronger_power = sum(float(item["power"]) for item in ordered_groups[idx + 1 :])

        denominator = group_gain * stronger_power + cfg.noise_power
        sinr = group_gain * power / denominator if denominator > 0.0 else np.inf
        rate = float(np.log2(1.0 + sinr))

        group["sinr"] = float(sinr)
        group["rate"] = rate

        if previous_success and rate >= cfg.R_threshold:
            group["success"] = True
        else:
            group["success"] = False
            previous_success = False
