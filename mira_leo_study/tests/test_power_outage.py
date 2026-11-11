"""Power-outage component tests with artificial inputs; no simulator/pilot runs."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mira_leo_study.algorithms import allocate_power, evaluate_power_plan, raw_reward, update_user_aoi
from mira_leo_study.calibration import evaluate_counterfactual_slot
from mira_leo_study.config import load_config, validate_fixed_study_config
from mira_leo_study.errors import ConfigurationError
from mira_leo_study.models import Group, Route
from mira_leo_study.paper_results import _validate_run
from mira_leo_study.reward_pilot import apply_reward_bounds

CONFIG = Path(__file__).parents[1] / "configs/base.yaml"


def group(index, gain, edge, *, satellite=0):
    return Group(satellite, 0, index, np.array([index], dtype=np.int32),
                 gain, 12.0 if edge else 50.0, edge, scheduled=True)


def incompatible_groups():
    # Three protected groups cannot all have >= the lone ordinary group's 0.3.
    return [group(i, i + 1, i < 3) for i in range(4)]


class PowerOutageTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)

    def test_phi_fifteen_and_outage_policy_are_formal_defaults(self):
        validate_fixed_study_config(self.config)
        self.assertEqual(self.config.get("algorithm.phi_theta_deg"), 15.0)
        self.assertEqual(self.config.get("algorithm.infeasible_power_handling"), "transmission_failure")
        with self.assertRaises(ConfigurationError):
            validate_fixed_study_config(self.config.with_overrides({"algorithm.phi_theta_deg": 30.0}))
        with self.assertRaisesRegex(ValueError, "matching phi=15"):
            apply_reward_bounds(self.config, {"phi_theta_deg": 30.0, "B_penalty": .05}, 0)

    def test_infeasible_plan_is_preserved_but_actual_transmission_is_zero(self):
        groups = incompatible_groups()
        for g in groups:
            g.success = g.sic_pass = g.rate_pass = True
            g.power = 17.0
        raw = evaluate_power_plan(self.config, groups)
        self.assertFalse(raw["valid"])
        self.assertEqual(raw["projection_solver_status"], 2)
        self.assertEqual([g.power for g in groups], [17.] * 4)
        diagnostic = allocate_power(self.config, groups)
        self.assertFalse(diagnostic["valid"])  # Never pretend the proposed plan is feasible.
        self.assertTrue(diagnostic["transmission_failure"])
        self.assertFalse(diagnostic["transmission_attempted"])
        self.assertEqual(diagnostic["status"], "POWER_INFEASIBLE_NO_TRANSMISSION")
        self.assertAlmostEqual(diagnostic["total_power"], 1.)
        self.assertEqual(diagnostic["actual_total_power"], 0.)
        self.assertEqual(diagnostic["removed_groups"], [])
        self.assertTrue(all(g.scheduled for g in groups))
        self.assertTrue(all(g.power == 0 and not g.success and not g.rate_pass and not g.sic_pass for g in groups))
        self.assertTrue(all(g.metadata["power_failure_no_transmission"] for g in groups))

    def test_legacy_error_policy_still_returns_an_unhandled_invalid_plan(self):
        # legacy_archived: pure old-policy fixture only; formal runners reject it.
        config = self.config.with_overrides({"algorithm.infeasible_power_handling": "error",
                                            "algorithm.power_mode": "twc_aim_staircase"})
        diagnostic = allocate_power(config, [group(0, 1, False), group(1, 2, True)])
        self.assertFalse(diagnostic["valid"])
        self.assertFalse(diagnostic["transmission_failure"])

    def test_numerical_solver_failure_is_counted_as_an_outage(self):
        fake = SimpleNamespace(success=False, status=4, message="artificial numerical failure")
        with patch("scipy.optimize.linprog", return_value=fake):
            diagnostic = allocate_power(self.config, [group(i, i+1, i==6) for i in range(10)])
        self.assertEqual(diagnostic["projection_solver_status"], 4)
        self.assertTrue(diagnostic["transmission_failure"])
        self.assertEqual(diagnostic["power_infeasible_reason"], "LP_NUMERIC_FAILURE")
        self.assertFalse(diagnostic["valid"])

    def test_floor_configuration_error_is_not_swallowed(self):
        diagnostic = allocate_power(self.config, [group(i, i + 1, False) for i in range(63)])
        self.assertTrue(diagnostic["floor_violations"])
        self.assertTrue(diagnostic["transmission_failure"])
        self.assertEqual(diagnostic["power_infeasible_reason"], "FLOOR_EXCEEDS_POOL")

    def test_outage_uses_failure_aoi_and_remains_in_reward(self):
        groups = incompatible_groups()
        diagnostic = allocate_power(self.config, groups)
        self.assertTrue(diagnostic["transmission_failure"])
        for handover, expected in ((False, 21.), (True, 23.5)):
            age = update_user_aoi(20., successful=groups[0].success, delta_c_slots=1.,
                                  propagation_delay_slots=0., isl_delay_slots=0.,
                                  handover_event=handover, handover_delay_slots=2.5)
            self.assertEqual(age, expected)
            self.assertAlmostEqual(raw_reward(0, age, .05, 3, 40, 1), -age - 6.)

    def test_counterfactual_outage_keeps_full_load_external_power(self):
        # Artificial arrays and a mocked group builder; no orbital/slot simulation.
        members = np.arange(120, dtype=np.int32)
        scenario = SimpleNamespace(
            devices=SimpleNamespace(ids=tuple(str(i) for i in members)),
            constellation=SimpleNamespace(elements=[SimpleNamespace(satellite_id=f"sat_{i+1:02d}") for i in range(8)]),
            subscriptions=SimpleNamespace(active_device_ids=members,
                message_members={m: members[m*40:(m+1)*40] for m in range(3)}),
        )
        failed = incompatible_groups()
        association = SimpleNamespace(serving_satellite=0, handover=False)
        routes = {0: Route(True, (0,), 0., 0, 0., 0.)}
        with patch("mira_leo_study.calibration.make_groups", return_value=failed):
            with patch("mira_leo_study.calibration.expected_cross_link_interference", return_value=.03) as interference:
                result = evaluate_counterfactual_slot(
                    self.config, scenario, slot=0, seed=0, ages_before=np.full(120, 100.),
                    association=association, distance=np.full((8, 120), 800.),
                    elevation=np.full((8, 120), 50.), routes=routes, action=4,
                    channel_amplitude=np.ones((8, 120)),
                )
        self.assertTrue(np.all(result["ages_after"] == 101.))
        self.assertEqual(len(result["rows"]), 1)
        row = result["rows"][0]
        self.assertTrue(row["transmission_failure_power"])
        self.assertEqual(row["actual_transmitted_power"], 0.)
        self.assertEqual(row["success_group_count"], 0)
        self.assertAlmostEqual(row["weighted_post_aoi_slots"], 4040.)
        self.assertEqual(interference.call_count, 4)
        for call in interference.call_args_list:
            np.testing.assert_array_equal(call.args[-1], [0., 1., 1., 1., 1., 1., 1., 1.])
        self.assertTrue(all(g.interference == .03 for g in failed))

    def test_formal_validator_accepts_and_audits_zero_power_outage(self):
        result = {"status": "COMPLETED_WITH_POWER_OUTAGES", "policy": "fixed_grouping",
                  "invalid_power_or_sic_count": 0, "post_schedule_power_drop_count": 0,
                  "max_transmitted_power_normalized": 1., "phi_theta_deg": 15., "B_penalty": .05,
                  "sic_projection_count": 1, "power_infeasible_serving_slots": 1, "power_outage_slots": 1,
                  "action_decision_count": {"sat_01:a=4": 1}, "bandit_observation_count": 0}
        serving_rows = []
        for c in range(1):
            serving_rows.append({"slot": 0, "action": 4, "regrouping_epoch": 0,
                "is_regrouping_slot": 1, "membership_hash": f"artificial-{c}",
                "action_owner_satellite_id": f"sat_{c+1:02d}", "regrouping_action_change": 0,
                "device_regrouping_events": 0, "csi_slot_index": 0, "power_csi_slot_index": 0,
                "scheduler_algorithm": "Algorithm2_strict", "power_allocator": "csi_expanded_protected_pool",
                "candidate_groups": 2, "candidate_edge_groups": 1, "candidate_nonedge_positive_priority": 1,
                "scheduled_edge_groups": 1, "scheduled_nonedge_groups": 1,
                "post_schedule_power_drop_count": 0, "transmitted_sic_violation_count": 0,
                "transmission_failure_power": int(c == 0), "power_plan_valid": int(c == 1),
                "transmission_attempted": int(c == 1), "transmitted_groups": 0 if c == 0 else 2,
                "successful_groups": 0 if c == 0 else 2, "transmitted_power_normalized": float(c),
                "power_solver_status": 2 if c == 0 else -1, "sic_violation_after_count": int(c == 0),
                "sic_projection_used": int(c == 0)})
        rows = [{"slot": 0, "scheduled_groups": 4, "successful_groups": 2}]
        _validate_run(result, rows, serving_rows, 1, self.config)
        serving_rows[0]["transmitted_power_normalized"] = .1
        with self.assertRaisesRegex(RuntimeError, "infeasible power must mean no transmission"):
            _validate_run(result, rows, serving_rows, 1, self.config)


if __name__ == "__main__":
    unittest.main()
