from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    """Configuration for the Version 1 synthetic MIRA-LEO simulator."""

    T: int = 1000
    slot_time: float = 0.01

    L: int = 2
    M: int = 3
    N_per_msg: int = 20

    H: float = 800e3
    fc: float = 20e9
    c: float = 3e8

    phi_min: float = 10.0
    phi_theta: float = 30.0

    P_sat: float = 1.0
    noise_power: float = 1e-12
    R_threshold: float = 1.0

    K_h: float = 2.0
    K_M: float = 0.0
    Delta_c: float = 1.0

    fixed_group_num: int = 4
    max_scheduled_groups: int = 6

    service_area: float = 1000e3
    random_seed: int = 42

    satellite_speed: float = 1500.0
    edge_bonus: float = 100.0
    power_mode: str = "equal"
    fading_enabled: bool = True
    message_weight: tuple[float, ...] = (0.5, 0.3, 0.2)

    @property
    def N_total(self) -> int:
        return self.M * self.N_per_msg
