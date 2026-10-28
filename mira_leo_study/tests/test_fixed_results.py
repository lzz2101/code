from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mira_leo_study.algorithms import (
    allocate_power, check_power_feasibility, schedule_groups, SlidingWindowUCB,
)
from mira_leo_study.channel import serving_channel_amplitude
from mira_leo_study.config import load_config, validate_fixed_study_config
from mira_leo_study.errors import ConfigurationError
from mira_leo_study.metrics import AssignmentTracker, advance_trace_hash, regrouping_overhead, running_mean
from mira_leo_study.models import Group
from mira_leo_study.paper_results import (
    ENVIRONMENT_KEYS, POLICIES, check_common_environment, run_paper_results, summarize_trace,
)
from mira_leo_study.reward_pilot import apply_reward_bounds, reward_context
from mira_leo_study.simulation import _random_action, run_simulation, select_grouping_action
from mira_leo_study.storage import AppendCsvWriter, write_csv
from mira_leo_study.tests import simulation_test


CONFIG = Path(__file__).parents[1] / "configs" / "base.yaml"


def group(index, edge=False, gain=None, members=None, satellite=0):
    return Group(satellite, 0, 0, index,
                 np.array([index % 300] if members is None else members, dtype=np.int32),
                 float(index + 1 if gain is None else gain), 15.0 if edge else 50.0, edge)


class StrictAlgorithm2Tests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)

    def schedule(self, groups, *, cap=62, ages=None):
        config = copy.deepcopy(self.config)
        config.data["derived"]["schedule_capacity_single_class_floor_only"] = cap
        return schedule_groups(config, groups, np.full(300, 100.0) if ages is None else ages,
                               np.zeros((8, 300)), False, 0.0)

    def test_edge_over_cap_selects_top_U_even_with_negative_priorities(self):
        edges = [group(i, True) for i in range(5)]
        nonedges = [group(i + 10) for i in range(5)]
        ages = np.zeros(300)
        ages[:5] = [-10, -20, -30, -40, -50]
        selected = self.schedule(edges + nonedges, cap=3, ages=ages)
        self.assertEqual([g.group_id for g in selected], [0, 1, 2])
        self.assertTrue(all(g.priority < 0 for g in selected))
        self.assertTrue(all(not g.scheduled for g in nonedges))

    def test_edge_within_cap_is_scheduled_even_when_negative(self):
        edges = [group(0, True), group(1, True)]
        ages = np.zeros(300)
        ages[0] = -10
        self.assertEqual(len(self.schedule(edges, cap=5, ages=ages)), 2)
        self.assertTrue(edges[0].priority < 0)
        self.assertTrue(all(g.scheduled for g in edges))

    def test_nonedge_only_positive_priority_and_remaining_capacity(self):
        groups = [group(0, True), group(1, True)] + [group(i) for i in range(2, 7)]
        ages = np.zeros(300)
        ages[2:7] = [10, 5, 1, 0, -1]
        selected = self.schedule(groups, cap=5, ages=ages)
        self.assertEqual([g.group_id for g in selected], [0, 1, 2, 3, 4])
        self.assertFalse(groups[-1].scheduled)
        self.assertFalse(groups[-2].scheduled)

    def test_scheduler_never_consults_power_feasibility(self):
        groups = [group(0, True, gain=2), group(1, gain=1)]
        with patch("mira_leo_study.algorithms.check_power_feasibility", side_effect=AssertionError("scheduler called power check")):
            self.assertEqual(len(self.schedule(groups)), 2)
        diagnostic = allocate_power(self.config, groups)
        self.assertFalse(diagnostic["valid"])
        self.assertEqual(diagnostic["removed_groups"], [])
        self.assertTrue(all(g.scheduled for g in groups))

    def test_scheduler_does_not_enforce_mixed_class_floor_caps(self):
        groups = [group(0, True, gain=.1)] + [group(i) for i in range(1, 41)]
        self.assertEqual(len(self.schedule(groups)), 41)
        diagnostic = check_power_feasibility(self.config, groups)
        self.assertFalse(diagnostic["valid"])
        self.assertGreater(diagnostic["nonedge_floor_total"], .3)
        self.assertEqual(diagnostic["removed_groups"], [])

    def test_single_class_algorithm2_cap_and_full_power(self):
        for edge in (False, True):
            groups = [group(i, edge) for i in range(70)]
            self.assertEqual(len(self.schedule(groups)), 62)
            diagnostic = allocate_power(self.config, groups)
            self.assertTrue(diagnostic["valid"])
            self.assertAlmostEqual(diagnostic["total_power"], 1)
            self.assertEqual(diagnostic["removed_groups"], [])

    def test_nonedge_nonpositive_is_not_scheduled(self):
        groups = [group(0), group(1)]
        self.assertEqual(self.schedule(groups, ages=np.zeros(300)), [])
        self.assertTrue(all(g.metadata["scheduler_rejection_reason"] == "NONPOSITIVE_PRIORITY" for g in groups))

    def test_tie_break_is_deterministic(self):
        groups = [group(2, True), group(0, True), group(1, True)]
        self.assertEqual([g.group_id for g in self.schedule(groups, cap=2)], [0, 1])

    def test_pure_power_plan_preserves_group_state(self):
        groups = [group(0, True), group(1, gain=2)]
        for g in groups:
            g.power, g.success, g.sinr = 17.0, True, 9.0
            g.metadata["existing"] = "preserved"
        before = copy.deepcopy(groups)
        plan = check_power_feasibility(self.config, groups)
        self.assertTrue(plan["valid"])
        for g, prior in zip(groups, before):
            for key in ("power", "success", "sinr", "scheduled", "metadata", "priority"):
                self.assertEqual(getattr(g, key), getattr(prior, key))
            g.scheduled = True
        self.assertEqual(plan["powers"], allocate_power(self.config, groups)["powers"])


class FixedMetricsTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)

    def test_fixed_config_and_floor_only_diagnostics(self):
        validate_fixed_study_config(self.config)
        self.assertEqual(self.config.get("algorithm.phi_theta_deg"), 20.0)
        self.assertEqual(self.config.get("algorithm.B_penalty"), 0.05)
        self.assertIsNone(self.config.get("search"))
        self.assertEqual(self.config.get("algorithm.action_set"), [1,2,4,5,8,10])
        self.assertEqual(self.config.get("derived.regrouping_interval_slots"), 100)
        self.assertEqual(self.config.get("derived.sw_ucb_window_epochs"), 10)
        for key, expected in (("single_class", 62), ("edge_when_mixed", 52), ("nonedge_when_mixed", 34)):
            self.assertEqual(self.config.get(f"derived.schedule_capacity_{key}_floor_only"), expected)
        for key, value in (("algorithm.phi_theta_deg", 25), ("algorithm.B_penalty", 1), ("algorithm.V_phi", 0.001)):
            with self.assertRaises(ConfigurationError):
                validate_fixed_study_config(self.config.with_overrides({key: value}))

    def test_random_reproducible_legal_and_fixed_always_four(self):
        actions = self.config.get("algorithm.action_set")
        first = [_random_action(actions, 0, slot, 3) for slot in range(100)]
        self.assertEqual(first, [_random_action(actions, 0, slot, 3) for slot in range(100)])
        self.assertTrue(set(first).issubset(set(actions)))
        fixed = [select_grouping_action("fixed_grouping", actions=actions, slot=slot, satellite=3, policy_seed=0) for slot in range(100)]
        self.assertEqual(fixed, [4] * 100)
        agent = SlidingWindowUCB(actions, 1000, 2**-0.5, 3)
        self.assertEqual(select_grouping_action("mira_sw_ucb", actions=actions, slot=0, satellite=3, policy_seed=0, agent=agent), agent.select(0))

    def test_policy_draws_do_not_change_indexed_channel(self):
        distance = np.full((8, 300), 800.0)
        before = serving_channel_amplitude(self.config, 0, 7, distance)
        for slot in range(100):
            _random_action(self.config.get("algorithm.action_set"), 123, slot, 3)
        after = serving_channel_amplitude(self.config, 0, 7, distance)
        np.testing.assert_array_equal(before, after)
        self.assertEqual(advance_trace_hash("", before), advance_trace_hash("", after))

    def test_running_mean_and_overhead_normalization(self):
        np.testing.assert_array_equal(running_mean([10, 20, 30]), [10, 15, 20])
        value = regrouping_overhead(120, 60, 120)
        self.assertEqual(value["events_per_device_per_min"], 1.0)
        self.assertEqual(value["evaluated_duration_min"], 2.0)
        self.assertEqual(value["eligible_device_service_count"], 60)

    def test_initial_assignment_handover_and_coverage_are_excluded(self):
        tracker = AssignmentTracker()
        visible = np.ones((8, 300), dtype=bool)
        groups = [group(0, members=[0, 1])]
        self.assertEqual(tracker.update(groups, visible)["regrouping_events"], 0)
        self.assertEqual(tracker.update([group(0, members=[0, 1])], visible)["regrouping_events"], 0)
        moved = [group(0, members=[0, 1], satellite=1)]
        event = tracker.update(moved, visible)
        self.assertEqual(event["regrouping_events"], 0)
        self.assertEqual(event["handover_assignments"], 2)
        split = [group(0, members=[0], satellite=1), group(1, members=[1], satellite=1)]
        self.assertEqual(tracker.update(split, visible)["regrouping_events"], 2)
        visible[:] = False
        self.assertEqual(tracker.update(split, visible)["regrouping_events"], 0)
        visible[:] = True
        self.assertEqual(tracker.update(groups, visible)["regrouping_events"], 0)

    def test_reward_bounds_identity_and_fixed_B(self):
        payload = {"phi_theta_deg": 20.0, "B_penalty": 0.05,
                   "reward_context": reward_context(self.config, 0),
                   "reward_lower": -100, "reward_upper": 10, "reward_identity": "unit-test"}
        resolved = apply_reward_bounds(self.config, payload, 0)
        self.assertEqual(resolved.get("algorithm.B_penalty"), 0.05)
        self.assertEqual(resolved.get("algorithm.reward_normalization"), "pilot_fixed_clip")
        with self.assertRaises(ValueError):
            apply_reward_bounds(self.config, payload, 1)
        with self.assertRaises(ValueError):
            apply_reward_bounds(self.config, payload | {"B_penalty": 0.1}, 0)

    def test_fresh_csv_writer_does_not_append_previous_run(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data.csv"
            AppendCsvWriter(path).append([{"slot": 0}])
            AppendCsvWriter(path).append([{"slot": 1}])
            with path.open(newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [{"slot": "1"}])


class FormalPackagingTests(unittest.TestCase):
    """Artificial unit fixtures stay in temporary directories; no simulation runs."""

    def test_artifacts_shared_environment_and_tau_api_with_stubbed_simulator(self):
        config = load_config(CONFIG)
        config = config.with_overrides({"algorithm.reward_bounds": {
            "lower": -1e6, "upper": 1e6, "identity": "artificial-unit-fixture", "context": reward_context(config, 0),
        }})

        def fake_run(config, *, policy, channel_seed, run_directory, max_slots, **kwargs):
            directory = Path(run_directory)
            directory.mkdir(parents=True, exist_ok=True)
            values = [10.0, 20.0, 30.0]
            rows = [{"slot": i, "time_s": i * 0.01, "policy": policy,
                     "weighted_sum_aoi_total": v, "weighted_sum_aoi_per_satellite_mean": v / 8,
                     "weighted_sum_aoi_per_satellite_max": v / 2,
                     "running_mean_weighted_sum_aoi_total": float(running_mean(values)[i]),
                     "scheduled_groups": 2, "successful_groups": 1,
                     "action_summary": "{\"sat_03\":4}"} for i, v in enumerate(values)]
            cohorts = [{"slot": i, "time_s": i * .01, "cohort_id": c, "satellite_id": f"sat_0{c + 1}",
                        "action": 4, "regrouping_action_change": 0, "device_regrouping_events": 0,
                        "initial_assignments": 120 if i == 0 else 0, "handover_assignments": 0,
                        "transmitted_power_normalized": 1, "candidate_groups": 12, "scheduled_groups": 1,
                        "rejected_by_power_groups": 11, "successful_groups": int(c == 0),
                        "scheduled_edge_groups": 0, "scheduled_nonedge_groups": 1,
                        "regrouping_epoch": 0, "is_regrouping_slot": int(i == 0),
                        "membership_hash": f"artificial-partition-{c}", "action_owner_satellite_id": f"sat_0{c+1}",
                        "csi_slot_index": i, "power_csi_slot_index": i,
                        "scheduler_algorithm": "Algorithm2_strict", "power_allocator": "twc_aim_staircase",
                        "post_schedule_power_drop_count": 0, "sic_violation_after_count": 0,
                        "candidate_edge_groups": 0, "candidate_nonedge_groups": 12,
                        "candidate_nonedge_positive_priority": 1, "sic_projection_used": 0,
                        "power_plan_valid": 1, "transmission_attempted": 1,
                        "transmission_failure_power": 0, "transmitted_groups": 1,
                        "power_solver_status": -1, "transmitted_sic_violation_count": 0,
                        "rejected_by_algorithm2_cap": 0}
                       for i in range(max_slots) for c in range(2)]
            write_csv(directory / "convergence_slot_metrics.csv", rows)
            write_csv(directory / "slot_metrics.csv", cohorts)
            return {"summary": {"policy": policy, "status": "COMPLETED", "channel_seed": channel_seed,
                    "phi_theta_deg": 20.0, "B_penalty": .05, "slots_completed": max_slots,
                    "invalid_power_or_sic_count": 0, "post_schedule_power_drop_count": 0,
                    "max_transmitted_power_normalized": 1.0, "action_change_events": 0,
                    "sic_projection_count": 0, "power_infeasible_cohort_slots": 0, "power_outage_slots": 0, "bandit_observation_count": 2 if policy == "mira_sw_ucb" else 0,
                    "action_decision_count": {"sat_01:a=4": 1, "sat_02:a=4": 1},
                    **regrouping_overhead(0, 240, max_slots * .01),
                    **{key: "artificial-shared-identity" for key in ENVIRONMENT_KEYS}}}

        with tempfile.TemporaryDirectory() as temp:
            with patch("mira_leo_study.paper_results.run_simulation", side_effect=fake_run) as simulator:
                result = run_paper_results(config, temp, slots=3, tau_s=[5, 10, 20])
            self.assertEqual(simulator.call_count, 5)
            self.assertEqual([call.kwargs["policy"] for call in simulator.call_args_list[:3]], list(POLICIES))
            convergence = Path(result["convergence_directory"])
            overhead = Path(result["overhead_directory"])
            for filename in ("comparison_slot_metrics.csv", "comparison_summary.csv", "comparison_summary.json",
                             "convergence_long_run_aoi.png", "convergence_long_run_aoi.pdf", "weighted_sum_aoi_raw_trace.png"):
                self.assertTrue((convergence / filename).is_file(), filename)
            for filename in ("regrouping_events.csv", "regrouping_overhead_summary.csv", "regrouping_overhead_table.md"):
                self.assertTrue((overhead / filename).is_file(), filename)
            with (convergence / "comparison_slot_metrics.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 9)
            self.assertEqual(result["comparison"][0]["final_running_mean_aoi"], 20)
            self.assertTrue(result["environment_identity_equal"])
            self.assertEqual(result["unavailable_baselines"]["aim_static"], "NOT_IMPLEMENTED")
            self.assertFalse((Path(temp) / "comparison" / "selected_phi.json").exists())

    def test_environment_mismatch_is_rejected(self):
        left = dict.fromkeys(ENVIRONMENT_KEYS, "same") | {"slots_completed": 3}
        right = left | {"channel_trace_hash": "different"}
        with self.assertRaises(RuntimeError):
            check_common_environment([left, right])

    def test_long_run_average_cannot_use_reward_values(self):
        summary = {"policy": "mira_sw_ucb", "status": "COMPLETED", "channel_seed": 0,
                   "invalid_power_or_sic_count": 0, "post_schedule_power_drop_count": 0,
                   "max_transmitted_power_normalized": 1, "sic_projection_count": 0, "power_infeasible_cohort_slots": 0, "power_outage_slots": 0}
        rows = [{"weighted_sum_aoi_total": v, "running_mean_weighted_sum_aoi_total": m,
                 "scheduled_groups": 1, "successful_groups": 1} for v, m in ((10, 10), (20, 15), (30, 20))]
        self.assertEqual(summarize_trace(rows, summary)["final_running_mean_aoi"], 20)
        rows[-1]["running_mean_weighted_sum_aoi_total"] = 99
        with self.assertRaises(ValueError):
            summarize_trace(rows, summary)


class SimulationFairnessTests(unittest.TestCase):
    @simulation_test
    def test_three_policies_share_environment_and_power_pipeline(self):
        config = load_config(CONFIG).with_overrides({"algorithm.reward_bounds": {
            "lower": -1e6, "upper": 1e6, "identity": "integration-test-only",
        }})
        summaries = []
        for policy in POLICIES:
            with patch("mira_leo_study.simulation.schedule_groups", wraps=schedule_groups) as scheduler:
                with patch("mira_leo_study.simulation.allocate_power", wraps=allocate_power) as allocator:
                    with patch("mira_leo_study.algorithms.check_power_feasibility", wraps=check_power_feasibility) as checker:
                        with patch("mira_leo_study.simulation.link_geometry", return_value=(np.full((8, 300), 800.), np.full((8, 300), 50.))):
                            result = run_simulation(config, channel_seed=0, policy=policy, max_slots=12, collect_slot_metrics=True)
            self.assertGreater(scheduler.call_count, 0)
            self.assertEqual(scheduler.call_count, allocator.call_count)
            self.assertEqual(checker.call_count, 0)
            summary = result["summary"]
            summaries.append(summary)
            self.assertEqual(summary["invalid_power_or_sic_count"], 0)
            self.assertEqual(summary["post_schedule_power_drop_count"], 0)
            self.assertLessEqual(summary["max_transmitted_power_normalized"], 1 + 1e-10)
            self.assertTrue(all(int(r["action"]) in config.get("algorithm.action_set") for r in result["slot_metrics"]))
            if policy == "fixed_grouping":
                self.assertTrue(all(r["action"] == 4 for r in result["slot_metrics"]))
            summarize_trace(result["convergence_metrics"], summary)
        check_common_environment(summaries)


if __name__ == "__main__":
    unittest.main()
