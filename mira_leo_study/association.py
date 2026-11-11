from __future__ import annotations

import numpy as np

from .models import AssociationDecision, SubscriptionTable


class AssociationTracker:
    """One serving LEO for the entire population, with coverage-first fallback."""

    def __init__(self, config, subscriptions: SubscriptionTable):
        self.phi_min = float(config.get("geometry.phi_min_deg"))
        self.hysteresis = float(config.get("association.hysteresis_score_deg"))
        self.ttt_slots = int(config.get("derived.handover_ttt_slots"))
        self.members = subscriptions.active_device_ids
        self.current: int | None = None
        self.pending: int | None = None
        self.pending_since: int | None = None

    def _records(self, elevation_deg: np.ndarray) -> list[dict]:
        elevation_deg = np.asarray(elevation_deg, dtype=float)
        if elevation_deg.ndim != 2 or not np.all(np.isfinite(elevation_deg)):
            raise ValueError("elevations must be a finite [satellite, device] matrix")
        if elevation_deg.shape[0] == 0 or len(self.members) == 0:
            raise ValueError("association requires satellites and active devices")
        records = []
        for satellite in range(elevation_deg.shape[0]):
            values = elevation_deg[satellite, self.members]
            coverage = int(np.count_nonzero(values >= self.phi_min))
            records.append(
                {
                    "assignment": satellite,
                    "coverage": coverage,
                    "full": coverage == len(self.members),
                    "score": float(np.min(values)),
                    "outage": len(self.members) - coverage,
                }
            )
        return records

    def update(self, slot: int, elevation_deg: np.ndarray) -> AssociationDecision:
        records = self._records(elevation_deg)
        full = [record for record in records if record["full"]]
        fallback_used = not bool(full)
        candidates = full or records
        candidates.sort(
            key=lambda record: (
                -record["coverage"],
                -record["score"],
                record["assignment"],
            )
        )
        best = candidates[0]
        previous = self.current
        initial = previous is None
        reason = "initial_acquisition" if initial else "maintained"

        current_record = next(
            (record for record in records if record["assignment"] == previous), None
        )
        current_legal = current_record is not None and bool(current_record["full"])
        selected = previous

        if initial:
            selected = best["assignment"]
        elif not current_legal:
            selected = best["assignment"]
            reason = "visibility_loss" if not fallback_used else "fallback_minimum_outage"
            self.pending = None
            self.pending_since = None
        elif (best["assignment"] != previous and not fallback_used
              and best["score"] >= current_record["score"] + self.hysteresis):
            if self.pending != best["assignment"]:
                self.pending = best["assignment"]
                self.pending_since = int(slot)
            elapsed = int(slot) - int(self.pending_since) + 1
            if elapsed >= max(1, self.ttt_slots):
                selected = best["assignment"]
                reason = "hysteresis_ttt"
                self.pending = None
                self.pending_since = None
        else:
            self.pending = None
            self.pending_since = None

        if selected is None:
            selected = best["assignment"]
        selected_record = next(record for record in records if record["assignment"] == selected)
        self.current = int(selected)
        handover = False if initial else previous != selected
        return AssociationDecision(
            serving_satellite=int(selected),
            handover=handover,
            initial_acquisition=initial,
            outage_receivers=int(selected_record["outage"]),
            fallback_used=not bool(selected_record["full"]),
            score_deg=float(selected_record["score"]),
            reason=reason,
        )


def precompute_association(config, scenario) -> list[AssociationDecision]:
    """Build the policy-independent association trace at communication-slot times."""
    from .geometry import link_geometry

    tracker = AssociationTracker(config, scenario.subscriptions)
    decisions = []
    dt_s = float(config.get("experiment.slot_dt_s"))
    for slot in range(config.slot_count):
        _, satellite_ecef = scenario.constellation.state(slot * dt_s)
        _, elevation = link_geometry(satellite_ecef, scenario.devices.ecef_km)
        decisions.append(tracker.update(slot, elevation))
    return decisions
