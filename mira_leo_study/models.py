from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DeviceTable:
    ids: tuple[str, ...]
    east_km: np.ndarray
    north_km: np.ndarray
    latitude_rad: np.ndarray
    longitude_rad: np.ndarray
    ecef_km: np.ndarray
    layout_hash: str


@dataclass(frozen=True)
class SubscriptionTable:
    cohort_by_device: np.ndarray
    message_by_device: np.ndarray
    active_device_ids: np.ndarray
    cohort_members: tuple[np.ndarray, ...]
    message_members: dict[tuple[int, int], np.ndarray]
    subscription_hash: str


@dataclass(frozen=True)
class SatelliteElement:
    satellite_id: str
    plane: int
    raan_rad: float
    u0_rad: float


@dataclass
class Group:
    satellite_index: int
    cohort_id: int
    message_id: int
    group_id: int
    member_indices: np.ndarray
    gain: float
    max_elevation_deg: float
    is_edge: bool
    scheduled: bool = False
    priority: float = 0.0
    power: float = 0.0
    interference: float = 0.0
    sinr: float = 0.0
    rate_bps_hz: float = 0.0
    rate_pass: bool = False
    sic_pass: bool = False
    success: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[int, int, int]:
        return self.cohort_id, self.message_id, self.group_id


@dataclass(frozen=True)
class GroupPartitionEntry:
    """Membership only; channel, elevation and decoding state are never held.

    Tuples keep membership immutable even across pickle/checkpoint round trips.
    """

    message_id: int
    group_id: int
    member_indices: tuple[int, ...]


@dataclass(frozen=True)
class AssociationDecision:
    serving_satellites: tuple[int, ...]
    handover: tuple[bool, ...]
    initial_acquisition: tuple[bool, ...]
    outage_receivers: tuple[int, ...]
    fallback_used: bool
    score_deg: float
    reason: str


@dataclass(frozen=True)
class Route:
    reachable: bool
    path: tuple[int, ...]
    distance_km: float
    hop_count: int
    fixed_delay_s: float
    geometric_delay_s: float
