from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import ClassVar


ACTION_SET = (1, 2, 4, 5, 8, 10)
TAU = 200
XI = 1.0 / sqrt(2.0)
B_EDGE = 0.2
NORMALIZE_REWARD = True
AOI_NORM_FACTOR = 100.0


@dataclass(frozen=True)
class V2Config:
    """Configuration for MIRA-LEO Algorithms 1, 2, and 3."""

    POWER_FRACTION_NOTE: ClassVar[str] = (
        "V_c=0.7 and V_nc=0.3 are implementation defaults for smoke tests; "
        "the paper specifies only 0<V_c,V_nc<1 and V_c+V_nc=1."
    )

    T: int = 1000
    slot_time_s: float = 0.01

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

    handover_delay_s: float = 0.025
    isl_delay_s: float = 0.005
    processing_delay_slots: float = 1.0

    max_scheduled_groups: int = 8
    V_phi: float = 5e-4
    use_theorem3_capacity: bool = True
    V_c: float = 0.7
    V_nc: float = 0.3
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

    @property
    def slot_time(self) -> float:
        """Compatibility alias used by the shared geometry/channel helpers."""
        return self.slot_time_s

    @property
    def handover_delay_slots(self) -> float:
        from mira_leo_v2a3_verified.time_units import seconds_to_slots

        return seconds_to_slots(self.handover_delay_s, self.slot_time_s)

    @property
    def isl_delay_slots(self) -> float:
        from mira_leo_v2a3_verified.time_units import seconds_to_slots

        return seconds_to_slots(self.isl_delay_s, self.slot_time_s)

    @property
    def K_h(self) -> float:
        """Compatibility alias whose internal unit is explicitly slots."""
        return self.handover_delay_slots

    @property
    def K_M(self) -> float:
        """Compatibility alias whose internal unit is explicitly slots per hop."""
        return self.isl_delay_slots

    @property
    def Delta_c(self) -> float:
        """Compatibility alias for the successful-update AoI floor in slots."""
        return self.processing_delay_slots

    @property
    def max_groups_per_satellite(self) -> int:
        return int(self.M * max(self.action_set))

    @property
    def reward_raw_bounds(self) -> tuple[float, float]:
        """Conservative fixed bounds used by the implementation normalization."""
        max_aoi_per_user = 1.0 + self.T * (1.0 + self.handover_delay_slots)
        lower = -(
            self.N_total * max_aoi_per_user
            + self.B_edge * self.max_groups_per_satellite
        )
        upper = float(self.max_groups_per_satellite)
        return float(lower), upper

    def __post_init__(self) -> None:
        self.validate_time_settings()
        self.validate_power_settings()

    def validate_time_settings(self) -> None:
        for name, value, allow_zero in [
            ("slot_time_s", self.slot_time_s, False),
            ("handover_delay_s", self.handover_delay_s, True),
            ("isl_delay_s", self.isl_delay_s, True),
            ("processing_delay_slots", self.processing_delay_slots, True),
        ]:
            if not isfinite(value) or value < 0.0 or (not allow_zero and value == 0.0):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be finite and {qualifier}")

    def validate_power_settings(self, tolerance: float = 1e-9) -> None:
        if not isfinite(self.P_sat) or self.P_sat <= 0.0:
            raise ValueError("P_sat must be finite and positive")
        if not isfinite(self.V_phi) or self.V_phi <= 0.0:
            raise ValueError("V_phi must be finite and positive")
        if not isfinite(self.V_c) or not 0.0 < self.V_c < 1.0:
            raise ValueError("V_c must be finite and satisfy 0 < V_c < 1")
        if not isfinite(self.V_nc) or not 0.0 < self.V_nc < 1.0:
            raise ValueError("V_nc must be finite and satisfy 0 < V_nc < 1")
        if abs((self.V_c + self.V_nc) - 1.0) > tolerance:
            raise ValueError("V_c + V_nc must equal 1 within tolerance")
