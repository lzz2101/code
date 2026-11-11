from __future__ import annotations

import hashlib
import math

import numpy as np


def running_mean(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("running_mean requires a finite one-dimensional sequence")
    return np.cumsum(values) / np.arange(1, len(values) + 1)


def regrouping_overhead(events: int, eligible_count: int, duration_s: float) -> dict:
    if events < 0 or eligible_count <= 0 or not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("overhead requires nonnegative events, positive devices and duration")
    duration_min = duration_s / 60.0
    return {
        "total_regrouping_events": int(events),
        "eligible_device_service_count": int(eligible_count),
        "evaluated_duration_s": float(duration_s),
        "evaluated_duration_min": duration_min,
        "events_per_device_per_min": events / (eligible_count * duration_min),
    }


def advance_trace_hash(previous: str, *arrays) -> str:
    """A checkpointable hash chain of actual environment arrays, excluding policy."""
    digest = hashlib.sha256(bytes.fromhex(previous) if previous else b"")
    for value in arrays:
        item = np.ascontiguousarray(value)
        digest.update(str((item.shape, item.dtype.str)).encode("ascii"))
        digest.update(item.tobytes())
    return digest.hexdigest()


class AssignmentTracker:
    """Track persistent logical G-RNTIs for subscribed device-service instances.

    A logical group includes its sorted membership, so rebuilding identical
    Python Group objects does not trigger signalling. A changed partition can
    trigger signalling even when a device retains the same numeric group_id.
    Only consecutive visible assignments on the same serving satellite count.
    """

    def __init__(self):
        self.previous = {}

    def update(self, groups, visible: np.ndarray) -> dict:
        current = {}
        counts = {"regrouping_events": 0, "initial_assignments": 0, "handover_assignments": 0}
        for group in groups:
            identity = (group.group_id, tuple(sorted(int(d) for d in group.member_indices)))
            for device in group.member_indices:
                device = int(device)
                if not visible[group.satellite_index, device]:
                    continue
                key = (device, group.message_id)
                assignment = (group.satellite_index, identity)
                prior = self.previous.get(key)
                if prior is None:
                    counts["initial_assignments"] += 1
                elif prior[0] != assignment[0]:
                    counts["handover_assignments"] += 1
                elif prior[1] != assignment[1]:
                    counts["regrouping_events"] += 1
                current[key] = assignment
        self.previous = current
        return counts
