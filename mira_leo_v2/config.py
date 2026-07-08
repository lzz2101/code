from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


ACTION_SET = (1, 2, 4, 5, 8, 10)
TAU = 200
XI = 1.0 / sqrt(2.0)
B_EDGE = 0.2
NORMALIZE_REWARD = True
AOI_NORM_FACTOR = 100.0


@dataclass(frozen=True)
class V2Config:
    """Configuration for Algorithm 1 MIRA-LEO regrouping."""

    T: int = 1000
    slot_time: float = 0.01

    L: int = 2
    M: int = 3
    N_per_msg: int = 40

    H: float = 800e3
    fc: float = 20e9
    c: float = 3e8

    phi_min: float = 10.0
    phi_theta: float = 30.0

    P_sat: float = 1.0
    noise_power: float = 1e-20
    R_threshold: float = 0.1

    K_h: float = 0.025
    K_M: float = 0.0
    Delta_c: float = 1.0

    max_scheduled_groups: int = 8
    service_area: float = 1000e3
    random_seed: int = 42
    satellite_speed: float = 1500.0
    fading_enabled: bool = True
    message_weight: tuple[float, ...] = (0.5, 0.3, 0.2)

    action_set: tuple[int, ...] = ACTION_SET
    tau: int = TAU
    xi: float = XI
    B_edge: float = B_EDGE
    normalize_reward: bool = NORMALIZE_REWARD
    aoi_norm_factor: float = AOI_NORM_FACTOR

    @property
    def N_total(self) -> int:
        return self.M * self.N_per_msg
