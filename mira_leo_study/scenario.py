from __future__ import annotations

import hashlib

import numpy as np

from .geometry import generate_devices
from .models import SubscriptionTable
from .orbit import SyntheticConstellation


def generate_subscriptions(config) -> SubscriptionTable:
    device_count = int(config.get("devices.unique_count"))
    messages = int(config.get("services.messages_per_active_satellite"))
    subscribers = int(config.get("services.subscribers_per_message"))
    required = messages * subscribers
    if required != device_count:
        raise ValueError("subscriptions must partition the complete device population")
    rng = np.random.default_rng(int(config.get("devices.subscription_seed")))
    permutation = rng.permutation(device_count)
    message_by_device = np.full(device_count, -1, dtype=np.int16)
    message_members: dict[int, np.ndarray] = {}
    for message in range(messages):
        subset = permutation[message * subscribers : (message + 1) * subscribers]
        message_members[message] = np.asarray(subset, dtype=np.int32)
        message_by_device[subset] = message
    active = np.asarray(permutation[:required], dtype=np.int32)
    digest = hashlib.sha256()
    digest.update(np.asarray(permutation, dtype="<i4").tobytes())
    digest.update(b"120ue_single_serving_message_only_v1")
    digest.update(message_by_device.astype("<i2").tobytes())
    return SubscriptionTable(
        message_by_device,
        active,
        message_members,
        digest.hexdigest(),
    )


class StudyScenario:
    def __init__(self, config, layout_seed: int | None = None):
        self.config = config
        self.devices = generate_devices(config, layout_seed=layout_seed)
        self.subscriptions = generate_subscriptions(config)
        self.constellation = SyntheticConstellation(config)

    @property
    def identity(self) -> dict[str, str]:
        return {
            "layout_hash": self.devices.layout_hash,
            "subscription_hash": self.subscriptions.subscription_hash,
        }
