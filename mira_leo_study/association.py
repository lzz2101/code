from __future__ import annotations

import itertools

import numpy as np

from .models import AssociationDecision, SubscriptionTable


class AssociationTracker:
    """Deterministic two-cohort injective association with hysteresis and TTT."""

    def __init__(self, config, subscriptions: SubscriptionTable):
        self.phi_min = float(config.get("geometry.phi_min_deg"))
        self.hysteresis = float(config.get("association.hysteresis_score_deg"))
        self.ttt_slots = int(config.get("derived.handover_ttt_slots"))
        self.members = subscriptions.cohort_members
        self.current: tuple[int, ...] | None = None
        self.pending: tuple[int, ...] | None = None
        self.pending_since: int | None = None

    def _records(self, elevation_deg: np.ndarray) -> list[dict]:
        satellite_count = elevation_deg.shape[0]
        records = []
        for assignment in itertools.permutations(range(satellite_count), len(self.members)):
            coverage = []
            minimums = []
            for cohort, sat in enumerate(assignment):
                values = elevation_deg[sat, self.members[cohort]]
                coverage.append(int(np.count_nonzero(values >= self.phi_min)))
                minimums.append(float(np.min(values)))
            records.append(
                {
                    "assignment": tuple(int(value) for value in assignment),
                    "coverage": tuple(coverage),
                    "full": all(coverage[index] == len(self.members[index]) for index in range(len(self.members))),
                    "score": float(sum(minimums)),
                    "outage": tuple(len(self.members[index]) - coverage[index] for index in range(len(self.members))),
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
                -sum(record["coverage"]),
                -record["score"],
                0 if record["assignment"] == self.current else 1,
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
        elif not fallback_used and best["score"] >= current_record["score"] + self.hysteresis:
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
        self.current = tuple(selected)
        handover = tuple(
            False if initial else previous[index] != selected[index]
            for index in range(len(selected))
        )
        return AssociationDecision(
            serving_satellites=tuple(selected),
            handover=handover,
            initial_acquisition=tuple(initial for _ in selected),
            outage_receivers=tuple(selected_record["outage"]),
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
