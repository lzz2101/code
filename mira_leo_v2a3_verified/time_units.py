from __future__ import annotations

from math import ceil, isfinite


def seconds_to_slots(delay_s: float, slot_time_s: float) -> float:
    """Convert a non-negative delay in seconds to continuous AoI slot units."""
    if not isfinite(delay_s) or delay_s < 0.0:
        raise ValueError("delay_s must be finite and non-negative")
    if not isfinite(slot_time_s) or slot_time_s <= 0.0:
        raise ValueError("slot_time_s must be finite and positive")
    return float(delay_s / slot_time_s)


def milliseconds_to_slots(delay_ms: float, slot_time_s: float) -> float:
    """Convert a non-negative delay in milliseconds to continuous slot units."""
    if not isfinite(delay_ms) or delay_ms < 0.0:
        raise ValueError("delay_ms must be finite and non-negative")
    return seconds_to_slots(delay_ms / 1000.0, slot_time_s)


def blackout_slots(delay_s: float, slot_time_s: float) -> int:
    """Return ceil(delay/slot) for models that use integer blackout slots."""
    return int(ceil(seconds_to_slots(delay_s, slot_time_s)))
