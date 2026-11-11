from __future__ import annotations

import math
import unittest
import importlib.util
from pathlib import Path

import numpy as np

from mira_leo_study.algorithms import (
    SlidingWindowUCB,
    allocate_power,
    decode_groups,
    legal_actions,
    make_groups,
    rate_passes_threshold,
    schedule_capacity,
    schedule_capacity_continuous,
    update_user_aoi,
)
from mira_leo_study.analysis import parameter_identifiability
from mira_leo_study.config import load_config
from mira_leo_study.errors import OptionalDependencyError
from mira_leo_study.models import Group
from mira_leo_study.scenario import StudyScenario


CONFIG = Path(__file__).parents[1] / "configs" / "base.yaml"


def group(index: int, gain: float, edge: bool, scheduled: bool = True) -> Group:
    return Group(
        satellite_index=0,
        
        message_id=0,
        group_id=index,
        member_indices=np.array([index % 120], dtype=np.int32),
        gain=gain,
        max_elevation_deg=5.0 if edge else 50.0,
        is_edge=edge,
        scheduled=scheduled,
        priority=float(index + 1),
    )


class AlgorithmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG).with_overrides(
            {
                "algorithm.B_penalty": 1.0,
                "algorithm.reward_normalization": "finite_horizon_bound",
                "algorithm.power_mode": "literal_with_checks",
            }
        )

    def test_actions_mean_group_count_and_divide_forty(self):
        self.assertEqual(legal_actions(40), (1, 2, 4, 5, 8, 10, 20, 40))
        scenario = StudyScenario(self.config)
        amplitude = np.ones((8, 120), dtype=float)
        elevation = np.full((8, 120), 45.0)
        groups = make_groups(
            self.config,
            0,
            5,
            scenario.subscriptions.message_members,
            amplitude,
            elevation,
        )
        self.assertEqual(len(groups), 15)
        self.assertTrue(all(len(item.member_indices) == 8 for item in groups))

    def test_edge_definition_uses_group_maximum(self):
        scenario = StudyScenario(self.config)
        amplitude = np.ones((8, 120), dtype=float)
        elevation = np.full((8, 120), 20.0)
        members = scenario.subscriptions.message_members[0]
        elevation[0, members[0]] = 35.0
        groups = make_groups(
            self.config,
            0,
            1,
            scenario.subscriptions.message_members,
            amplitude,
            elevation,
        )
        message_zero = next(item for item in groups if item.message_id == 0)
        self.assertEqual(message_zero.max_elevation_deg, 35.0)
        self.assertFalse(message_zero.is_edge)

    def test_edge_threshold_is_strict(self):
        scenario = StudyScenario(self.config.with_overrides({"algorithm.phi_theta_deg": 20.0}))
        amplitude = np.ones((8, 120), dtype=float)
        members = scenario.subscriptions.message_members[0]
        for values, expected in (([16.0, 17.0, 19.0], True), ([16.0, 17.0, 20.0], False), ([16.0, 25.0, 19.0], False)):
            elevation = np.full((8, 120), 50.0)
            elevation[0, members] = 11.0
            elevation[0, members[:3]] = values
            groups = make_groups(
                scenario.config,
                0,
                1,
                scenario.subscriptions.message_members,
                amplitude,
                elevation,
            )
            target = next(item for item in groups if item.message_id == 0)
            self.assertEqual(target.is_edge, expected)

    def test_theorem_capacity_uses_feasible_integer(self):
        self.assertAlmostEqual(schedule_capacity_continuous(1.0, 0.0005), 62.74752959602454)
        self.assertEqual(schedule_capacity(1.0, 0.0005), 62)
        self.assertLessEqual(0.0005 * 62 * 63 / 2.0, 1.0)
        self.assertGreater(0.0005 * 63 * 64 / 2.0, 1.0)

    def test_literal_power_floor_and_unscheduled_power(self):
        groups = [group(index, 1.0 + index, False) for index in range(62)]
        unscheduled = group(100, 100.0, False, scheduled=False)
        unscheduled.power = 99.0
        result = allocate_power(self.config, groups + [unscheduled])
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(sum(item.power for item in groups), 1.0, places=10)
        self.assertEqual(unscheduled.power, 0.0)
        overflow = [group(index, 1.0 + index, False) for index in range(63)]
        invalid = allocate_power(self.config, overflow)
        self.assertFalse(invalid["valid"])
        self.assertEqual(invalid["status"], "INVALID_POWER_OR_SIC")
        self.assertEqual(len(invalid["floor_violations"]), 1)

    def test_literal_cross_class_sic_violation_is_not_hidden(self):
        weak_nonedge = group(0, 1.0, False)
        strong_edge = group(1, 2.0, True)
        result = allocate_power(self.config, [weak_nonedge, strong_edge])
        self.assertFalse(result["valid"])
        self.assertEqual(len(result["sic_order_violations"]), 1)
        self.assertAlmostEqual(weak_nonedge.power, 0.3)
        self.assertAlmostEqual(strong_edge.power, 0.7)

    def test_constraint_projection_is_explicit_and_dependency_checked(self):
        projected_config = self.config.with_overrides(
            {"algorithm.power_mode": "constraint_projection"}
        )
        groups = [group(0, 1.0, False), group(1, 2.0, True)]
        if importlib.util.find_spec("scipy") is None:
            with self.assertRaises(OptionalDependencyError):
                allocate_power(projected_config, groups)
        else:
            result = allocate_power(projected_config, groups)
            self.assertFalse(result["valid"])
            self.assertTrue(result["projection_infeasible"])
            self.assertEqual(result["removed_groups"], [])
            self.assertTrue(all(g.scheduled for g in groups))

    def test_sic_failure_propagates_even_if_stronger_rate_passes(self):
        weak = group(0, 1e-20, False)
        strong = group(1, 1e-17, False)
        weak.power = 0.6
        strong.power = 0.4
        threshold_config = self.config.with_overrides({"algorithm.R_theta": 0.3})
        decode_groups(threshold_config, [weak, strong])
        self.assertFalse(weak.rate_pass)
        self.assertTrue(strong.rate_pass)
        self.assertFalse(strong.success)

    def test_rate_threshold_has_no_payload_gate(self):
        noise = float(self.config.get("derived.noise_power"))
        threshold = float(self.config.get("algorithm.R_theta"))
        self.assertFalse(rate_passes_threshold(0.9e-5, threshold))
        self.assertTrue(rate_passes_threshold(1.0e-5, threshold))
        self.assertTrue(rate_passes_threshold(1.1e-5, threshold))
        for multiplier, expected in ((0.9, False), (1.1, True)):
            item = group(0, 1.0, False)
            expected_rate = threshold * multiplier
            expected_sinr = 2.0**expected_rate - 1.0
            item.power = expected_sinr * noise
            decode_groups(self.config, [item])
            self.assertAlmostEqual(item.rate_bps_hz, math.log2(1.0 + expected_sinr), places=14)
            self.assertEqual(item.rate_pass, expected)
            self.assertEqual(item.success, expected)
            self.assertFalse(hasattr(item, "payload_pass"))

    def test_aoi_success_failure_handover_and_no_double_charge(self):
        success = update_user_aoi(
            20.0,
            successful=True,
            delta_c_slots=1.0,
            propagation_delay_slots=0.25,
            isl_delay_slots=1.0,
            handover_event=True,
            handover_delay_slots=5.0,
        )
        failure = update_user_aoi(
            20.0,
            successful=False,
            delta_c_slots=1.0,
            propagation_delay_slots=0.25,
            isl_delay_slots=1.0,
            handover_event=True,
            handover_delay_slots=5.0,
        )
        blackout = update_user_aoi(
            20.0,
            successful=False,
            delta_c_slots=1.0,
            propagation_delay_slots=0.0,
            isl_delay_slots=0.0,
            handover_event=True,
            handover_delay_slots=5.0,
            handover_mode="physical_blackout",
        )
        self.assertAlmostEqual(success, 2.25)
        self.assertAlmostEqual(failure, 26.0)
        self.assertAlmostEqual(blackout, 21.0)

    def test_sw_ucb_uses_global_slot_window_and_inactive_gaps(self):
        agent = SlidingWindowUCB([1, 2], window_slots=3, xi=0.5, satellite_id=0)
        agent.update(0, 1, 0.5)
        agent.update(4, 2, 0.5)
        self.assertEqual(len(agent.observations), 1)
        self.assertEqual(agent.observations[0].slot, 4)

    def test_no_edge_is_not_identifiable(self):
        self.assertEqual(parameter_identifiability([0, 0, 0], 3)["status"], "NOT_IDENTIFIABLE")
        self.assertEqual(parameter_identifiability([1, 2], 2)["status"], "IDENTIFIABLE")


if __name__ == "__main__":
    unittest.main()
