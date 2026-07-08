from __future__ import annotations

import numpy as np

from .config import SimConfig


def compute_channel_gain(
    distance: np.ndarray,
    cfg: SimConfig,
    rng: np.random.Generator | None = None,
    fading: bool = True,
) -> np.ndarray:
    """Compute free-space path loss multiplied by optional Rayleigh power fading."""
    rng = rng or np.random.default_rng(cfg.random_seed)
    zeta = (cfg.c / (4.0 * np.pi * distance * cfg.fc)) ** 2
    fading_power = rng.exponential(1.0, size=distance.shape) if fading else 1.0
    return zeta * fading_power
