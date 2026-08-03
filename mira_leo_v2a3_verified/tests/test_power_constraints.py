from __future__ import annotations

import unittest

from mira_leo_v2a3_verified.algorithms.power_allocation import (
    PowerAllocationError,
    mira_leo_power_allocation,
)
from mira_leo_v2a3_verified.algorithms.regrouping import Group
from mira_leo_v2a3_verified.validation.validators import (
    ValidationError,
    validate_power_constraints,
)


def group(
    group_id: int,
    gain: float,
    *,
    edge: bool,
    satellite: int = 0,
    scheduled: bool = True,
) -> Group:
    return Group(
        satellite,
        0,
        group_id,
        [{"id": group_id, "message_id": 0, "aoi": 1.0, "handover": 0}],
        group_gain=gain,
        is_edge=edge,
        scheduled=scheduled,
    )


def allocate(groups: list[Group]) -> dict[str, object]:
    return mira_leo_power_allocation(
        groups, P_sat=1.0, V_c=0.7, V_nc=0.3, V_phi=5e-4
    )


class PowerConstraintTests(unittest.TestCase):
    def test_pw_01_empty(self) -> None:
        summary = allocate([])
        self.assertEqual(summary["total_allocated_power"], 0.0)
        self.assertEqual(summary["sic_power_order_violations"], 0.0)

    def test_pw_02_edge_only(self) -> None:
        groups = [group(0, 0.1, edge=True), group(1, 0.2, edge=True)]
        summary = allocate(groups)
        self.assertAlmostEqual(summary["edge_power"], 1.0)
        self.assertEqual(summary["nonedge_power"], 0.0)

    def test_pw_03_nonedge_only(self) -> None:
        groups = [group(0, 0.1, edge=False), group(1, 0.2, edge=False)]
        summary = allocate(groups)
        self.assertAlmostEqual(summary["nonedge_power"], 1.0)
        self.assertEqual(summary["edge_power"], 0.0)

    def test_pw_04_mixed_feasible_and_floors(self) -> None:
        groups = [
            group(0, 0.1, edge=True),
            group(1, 0.2, edge=True),
            group(2, 0.5, edge=False),
            group(3, 0.8, edge=False),
        ]
        summary = allocate(groups)
        self.assertAlmostEqual(summary["edge_power"], 0.7)
        self.assertAlmostEqual(summary["nonedge_power"], 0.3)
        self.assertAlmostEqual(summary["total_allocated_power"], 1.0)
        for item in groups:
            self.assertGreaterEqual(item.power, item.metadata["floor_power"])

    def test_pw_05_infeasible_floor_is_rejected(self) -> None:
        groups = [group(0, 0.1, edge=True), group(1, 0.2, edge=True)]
        with self.assertRaises(PowerAllocationError):
            mira_leo_power_allocation(
                groups, P_sat=1.0, V_c=0.7, V_nc=0.3, V_phi=0.8
            )

    def test_pw_06_negative_power_injection_is_detected(self) -> None:
        item = group(0, 0.1, edge=False)
        item.power = -0.01
        with self.assertRaisesRegex(ValidationError, "negative_power"):
            validate_power_constraints([item], P_sat=1.0, strict=True)

    def test_pw_07_per_satellite_overflow_is_not_hidden_by_total(self) -> None:
        sat0 = group(0, 0.1, edge=False, satellite=0)
        sat1 = group(1, 0.1, edge=False, satellite=1)
        sat0.power = 1.01
        sat1.power = 0.99
        self.assertAlmostEqual(sat0.power + sat1.power, 2.0)
        with self.assertRaisesRegex(ValidationError, "satellite_power_overflow"):
            validate_power_constraints([sat0, sat1], P_sat=1.0, strict=True)

    def test_pw_08_unscheduled_positive_power_is_detected(self) -> None:
        item = group(0, 0.1, edge=False, scheduled=False)
        item.power = 0.1
        with self.assertRaisesRegex(ValidationError, "unscheduled_positive_power"):
            validate_power_constraints([item], P_sat=1.0, strict=True)

    def test_pw_09_scheduled_nonpositive_power_is_detected(self) -> None:
        item = group(0, 0.1, edge=False)
        item.power = 0.0
        with self.assertRaisesRegex(ValidationError, "scheduled_nonpositive_power"):
            validate_power_constraints([item], P_sat=1.0, strict=True)


if __name__ == "__main__":
    unittest.main()
