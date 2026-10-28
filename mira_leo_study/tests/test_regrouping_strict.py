"""Artificial component tests; no orbital/channel simulation or pilot runs."""

from __future__ import annotations

import copy
import csv
import pickle
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mira_leo_study.algorithms import (
    RewardNormalizer, SlidingWindowUCB, allocate_power, build_group_partition,
    decode_groups, evaluate_twc_aim_power_plan, materialize_groups_from_partition,
    schedule_groups, twc_aim_staircase_target,
)
from mira_leo_study.config import load_config, validate_fixed_study_config
from mira_leo_study.errors import ConfigurationError, ModelConsistencyError
from mira_leo_study.metrics import AssignmentTracker
from mira_leo_study.models import Group
from mira_leo_study.paper_results import _validate_run, run_paper_results
from mira_leo_study.regrouping import groups_for_slot, initial_epoch_state, observe_slot_reward, partition_hash
from mira_leo_study.reward_pilot import apply_reward_bounds, reward_context
from mira_leo_study.simulation import CHECKPOINT_SCHEMA_VERSION, build_parameter_snapshot, run_simulation, select_grouping_action
from mira_leo_study.tests import simulation_test

CONFIG = Path(__file__).parents[1] / "configs" / "base.yaml"


def small_group(index, gain, edge):
    return Group(0, 0, 0, index, np.array([index], dtype=np.int32), gain,
                 15.0 if edge else 45.0, edge, scheduled=True)


def component_state(config):
    return {
        **initial_epoch_state(), "previous_actions": {}, "action_change_events": 0,
        "action_histogram": Counter(),
        "agents": [SlidingWindowUCB(config.get("algorithm.action_set"),
                                    window_steps=config.get("derived.sw_ucb_window_epochs"),
                                    satellite_id=s) for s in range(8)],
        "normalizers": [RewardNormalizer(-1000, 1000, "artificial-component-bounds") for _ in range(8)],
    }


class RegroupingConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)

    def test_intervals_compact_and_snapshot(self):
        validate_fixed_study_config(self.config)
        self.assertEqual(self.config.get("algorithm.action_set_name"), "compact")
        self.assertEqual(self.config.get("algorithm.action_set"), [1, 2, 4, 5, 8, 10])
        self.assertEqual(self.config.get("derived.regrouping_interval_slots"), 100)
        self.assertEqual(self.config.get("derived.sw_ucb_window_slots"), 1000)
        self.assertEqual(self.config.get("derived.sw_ucb_window_epochs"), 10)
        self.assertEqual(self.config.get("derived.algorithm2_U"), 62)
        snapshot = build_parameter_snapshot(self.config, "mira_sw_ucb", 0)
        self.assertEqual(snapshot["bandit_time_index"], "global_regrouping_epoch")
        self.assertEqual(snapshot["regrouping_interval_s"], 1)
        self.assertEqual(snapshot["power_allocator"], "twc_aim_staircase")

    def test_invalid_interval_and_fractional_window_rejected(self):
        for interval in (0, -1, float("nan"), float("inf"), 1.005):
            with self.subTest(interval=interval), self.assertRaises(ConfigurationError):
                self.config.with_overrides({"algorithm.regrouping_interval_s": interval})
        with self.assertRaises(ConfigurationError):
            self.config.with_overrides({"algorithm.sw_ucb_window_s": 10.5})
        with self.assertRaises(ConfigurationError):
            validate_fixed_study_config(self.config.with_overrides({"formal_results.regrouping_overhead_tau_s": [5.5]}))
        with self.assertRaises(ConfigurationError):
            validate_fixed_study_config(self.config.with_overrides({"algorithm.power_mode": "literal_with_checks"}))

    def test_old_reward_bounds_cannot_apply_to_compact_context(self):
        old = self.config.with_overrides({"algorithm.action_set_name": "full",
                                         "algorithm.action_set": [1, 2, 4, 5, 8, 10, 20, 40],
                                         "algorithm.regrouping_interval_s": .01})
        payload = {"phi_theta_deg": 20.0, "B_penalty": .05, "reward_lower": -1000,
                   "reward_upper": 1000, "reward_identity": "old-artifact",
                   "reward_context": reward_context(old, 0)}
        self.assertNotEqual(reward_context(old, 0), reward_context(self.config, 0))
        with self.assertRaises(ValueError):
            apply_reward_bounds(self.config, payload, 0)


class PartitionAndEpochTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)
        self.members = {(0, m): np.arange(40 * m, 40 * (m + 1), dtype=np.int32) for m in range(3)}
        self.channel = np.tile(np.arange(1, 301, dtype=float), (8, 1))
        self.elevation = np.full((8, 300), 50.0)

    def prepare(self, state, slot, *, policy="mira_sw_ucb", satellite=0, channel=None, selector=select_grouping_action):
        return groups_for_slot(self.config, state, cohort=0, satellite=satellite, slot=slot,
                               policy=policy, policy_seed=7, fixed_action=4, message_members=self.members,
                               channel_amplitude=self.channel if channel is None else channel,
                               elevation_deg=self.elevation, select_action=selector)

    def test_membership_is_immutable_and_dynamic_fields_use_current_slot(self):
        partition = build_group_partition(self.config, 0, 0, 4, self.members, self.channel)
        self.assertIsInstance(partition[0].member_indices, tuple)
        restored = pickle.loads(pickle.dumps(partition))
        self.assertEqual(partition, restored)
        before = materialize_groups_from_partition(self.config, 0, 0, partition, self.channel, self.elevation)
        new_channel = self.channel[:, ::-1].copy()
        new_elevation = np.full((8, 300), 15.0)
        after = materialize_groups_from_partition(self.config, 0, 0, partition, new_channel, new_elevation)
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a.member_indices, b.member_indices)
            self.assertNotEqual(a.gain, b.gain)
            self.assertFalse(a.is_edge)
            self.assertTrue(b.is_edge)
            self.assertNotEqual(a.max_elevation_deg, b.max_elevation_deg)
        self.assertEqual(partition_hash(partition), partition_hash(restored))

    def test_action_and_partition_held_350_component_ticks(self):
        state = component_state(self.config)
        hashes, actions, changes = [], [], []
        selector = unittest.mock.Mock(wraps=select_grouping_action)
        tracker = AssignmentTracker()
        for slot in range(350):
            channel = np.roll(self.channel, slot, axis=1)
            action, groups, changed = self.prepare(state, slot, channel=channel, selector=selector)
            actions.append(action)
            changes.append(changed)
            hashes.append(state["held_partition_hash"][0])
            events = tracker.update(groups, np.ones((8, 300), dtype=bool))
            if slot % 100:
                self.assertEqual(events["regrouping_events"], 0)
                self.assertEqual(changed, 0)
            observe_slot_reward(self.config, state, cohort=0, slot=slot, slot_limit=350, raw_reward=-100.0)
        self.assertEqual(selector.call_count, 4)
        self.assertEqual([c.kwargs["slot"] for c in selector.call_args_list], [0, 1, 2, 3])
        self.assertEqual([actions[i] for i in (0, 100, 200, 300)], [1, 2, 4, 5])
        for start in (0, 100, 200, 300):
            self.assertEqual(len(set(actions[start:start + 100])), 1)
            self.assertEqual(len(set(hashes[start:start + 100])), 1)
        self.assertEqual(state["bandit_observation_count"], 4)
        self.assertEqual(sum(state["action_histogram"].values()), 350)
        self.assertEqual(sum(state["action_decision_histogram"].values()), 4)

    def test_random_only_draws_at_boundaries_and_is_reproducible(self):
        sequences = []
        for _ in range(2):
            state = component_state(self.config)
            selector = unittest.mock.Mock(wraps=select_grouping_action)
            actions = [self.prepare(state, slot, policy="random_grouping", selector=selector)[0] for slot in range(250)]
            self.assertEqual(selector.call_count, 3)
            for start in (0, 100, 200):
                self.assertEqual(len(set(actions[start:start + 100])), 1)
            self.assertTrue(set(actions).issubset({1, 2, 4, 5, 8, 10}))
            sequences.append(actions)
        self.assertEqual(sequences[0], sequences[1])

    def test_fixed_rebuilds_only_at_boundary(self):
        state = component_state(self.config)
        with patch("mira_leo_study.regrouping.build_group_partition", wraps=build_group_partition) as builder:
            actions = [self.prepare(state, slot, policy="fixed_grouping")[0] for slot in range(250)]
        self.assertEqual(actions, [4] * 250)
        self.assertEqual(builder.call_count, 3)

    def test_handover_holds_membership_and_rewards_original_owner(self):
        state = component_state(self.config)
        selector = unittest.mock.Mock(wraps=select_grouping_action)
        tracker = AssignmentTracker()
        visible = np.ones((8, 300), dtype=bool)
        initial_hash = None
        for slot in range(100):
            satellite = 0 if slot < 50 else 1
            action, groups, changed = self.prepare(state, slot, satellite=satellite, selector=selector)
            events = tracker.update(groups, visible)
            if initial_hash is None:
                initial_hash = state["held_partition_hash"][0]
            self.assertEqual(initial_hash, state["held_partition_hash"][0])
            self.assertEqual(state["epoch_action_owner_satellite"][0], 0)
            if slot == 50:
                self.assertEqual(events["handover_assignments"], 120)
                self.assertEqual(events["regrouping_events"], 0)
                self.assertEqual(changed, 0)
                self.assertTrue(all(g.satellite_index == 1 for g in groups))
            observe_slot_reward(self.config, state, cohort=0, slot=slot, slot_limit=200, raw_reward=-100)
        self.assertEqual(selector.call_count, 1)
        self.assertEqual(len(state["agents"][0].observations), 1)
        self.assertEqual(len(state["agents"][1].observations), 0)
        self.prepare(state, 100, satellite=1, selector=selector)
        self.assertEqual(state["epoch_action_owner_satellite"][0], 1)
        self.assertEqual(selector.call_count, 2)

    def test_epoch_reward_mean_and_no_finalize_when_checkpoint_pauses(self):
        state = component_state(self.config)
        for slot in range(150):
            self.prepare(state, slot)
            observe_slot_reward(self.config, state, cohort=0, slot=slot, slot_limit=300, raw_reward=slot)
        self.assertEqual(state["bandit_observation_count"], 1)
        self.assertEqual(state["epoch_reward_count"][0], 50)
        self.assertFalse(state["epoch_reward_finalized"][0])
        self.assertAlmostEqual(state["agents"][0].observations[0].reward,
                               np.mean([(s + 1000) / 2000 for s in range(100)]))
        restored = pickle.loads(pickle.dumps(state))
        traces = []
        for candidate in (state, restored):
            trace = []
            for slot in range(150, 300):
                action, _, _ = self.prepare(candidate, slot)
                normalized, epoch_reward = observe_slot_reward(self.config, candidate, cohort=0,
                    slot=slot, slot_limit=300, raw_reward=slot)
                trace.append((action, candidate["held_partition_hash"][0], normalized, epoch_reward))
            traces.append(trace)
            self.assertEqual(candidate["bandit_observation_count"], 3)
        self.assertEqual(traces[0], traces[1])
        self.assertEqual(list(state["agents"][0].observations), list(restored["agents"][0].observations))

    def test_compact_initial_exploration_and_global_epoch_pruning(self):
        agent = SlidingWindowUCB([1, 2, 4, 5, 8, 10], window_steps=10)
        first = []
        for epoch in range(6):
            action = agent.select(epoch)
            first.append(action)
            agent.update(epoch, action, .5)
        self.assertEqual(first, [1, 2, 4, 5, 8, 10])
        agent.select(15)
        self.assertTrue(all(record.slot >= 5 for record in agent.observations))
        agent.select(40)
        self.assertEqual(list(agent.observations), [])


class TwcPowerTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)

    def test_class_budgets_and_continuous_residual_exact_values(self):
        groups = [small_group(i, i + 1, False) for i in range(3)]
        target, floor = twc_aim_staircase_target(groups, .3, .0005)
        self.assertAlmostEqual(floor, .003)
        for g, reserved in zip(groups, (.0015, .001, .0005)):
            self.assertAlmostEqual(target[id(g)], reserved + .099)
        for edge in (False, True):
            for g in groups:
                g.is_edge = edge
            diagnostic = allocate_power(self.config, groups)
            self.assertTrue(diagnostic["valid"])
            self.assertEqual(diagnostic["edge_budget"], 1.0 if edge else 0.0)
            self.assertEqual(diagnostic["nonedge_budget"], 0.0 if edge else 1.0)
            self.assertAlmostEqual(diagnostic["total_power"], 1)
        mixed = [small_group(0, 1, True), small_group(1, 2, False)]
        diagnostic = allocate_power(self.config, mixed)
        self.assertTrue(diagnostic["valid"])
        self.assertEqual((diagnostic["edge_budget"], diagnostic["nonedge_budget"]), (.7, .3))

    def test_compact_30_group_floor_invariant(self):
        groups = [small_group(i, i + 1, False) for i in range(30)]
        target, floor = twc_aim_staircase_target(groups, .3, .0005)
        self.assertAlmostEqual(floor, .2325)
        self.assertLess(floor, .3)
        self.assertAlmostEqual(sum(target.values()), .3)

    def test_global_sic_projection_repairs_power_only(self):
        groups = [small_group(0, 1, True), small_group(1, 2, False), small_group(2, 3, True)]
        before = [g.key for g in groups if g.scheduled]
        diagnostic = allocate_power(self.config, groups)
        self.assertTrue(diagnostic["valid"], diagnostic)
        self.assertTrue(diagnostic["sic_projection_used"])
        self.assertGreater(len(diagnostic["sic_order_violations_before"]), 0)
        self.assertEqual(diagnostic["sic_order_violations_after"], [])
        self.assertEqual([g.key for g in groups if g.scheduled], before)
        self.assertEqual(diagnostic["removed_groups"], [])
        self.assertAlmostEqual(groups[0].power + groups[2].power, .7)
        self.assertAlmostEqual(groups[1].power, .3)
        self.assertGreaterEqual(groups[0].power + 1e-10, groups[1].power)
        self.assertGreaterEqual(groups[1].power + 1e-10, groups[2].power)

    def test_infeasible_projection_reports_error_without_dropping(self):
        groups = [small_group(0, 1, False), small_group(1, 2, True)]
        before = copy.deepcopy(groups)
        diagnostic = evaluate_twc_aim_power_plan(self.config, groups)
        self.assertFalse(diagnostic["valid"])
        self.assertEqual(diagnostic["status"], "MODEL_CONSISTENCY_ERROR")
        self.assertTrue(diagnostic["projection_infeasible"])
        self.assertEqual(diagnostic["removed_groups"], [])
        for actual, prior in zip(groups, before):
            self.assertEqual(actual.power, prior.power)
            self.assertEqual(actual.scheduled, prior.scheduled)
        applied = allocate_power(self.config, groups)
        self.assertFalse(applied["valid"])
        self.assertTrue(all(g.scheduled for g in groups))
        self.assertTrue(all(g.power == 0 for g in groups))

    def test_current_csi_order_changes_immediately_and_future_csi_is_unused(self):
        members = {(0, m): np.arange(m * 40, (m + 1) * 40, dtype=np.int32) for m in range(3)}
        channel = np.tile(np.arange(1, 301, dtype=float), (8, 1))
        partition = build_group_partition(self.config, 0, 0, 2, members, channel)
        elevation = np.full((8, 300), 50.0)
        future = channel.copy()

        def evaluate(current):
            groups = materialize_groups_from_partition(self.config, 0, 0, partition, current, elevation)
            schedule_groups(self.config, groups, np.full(300, 1000.), np.full((8, 300), 800.), False, 0.)
            result = allocate_power(self.config, groups)
            self.assertTrue(result["valid"])
            decode_groups(self.config, groups)
            return groups, [(g.key, g.scheduled, g.power, g.rate_bps_hz, g.success) for g in groups]

        before, trace = evaluate(channel)
        future[:] = 10000  # Altering future data cannot affect a current-only API.
        _, same = evaluate(channel)
        self.assertEqual(trace, same)
        current = channel[:, ::-1].copy()
        after, _ = evaluate(current)
        self.assertGreater(before[0].power, before[-1].power)
        self.assertLess(after[0].power, after[-1].power)
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a.member_indices, b.member_indices)


class FormalConsistencyTests(unittest.TestCase):
    def test_validator_rejects_action_twenty_under_compact_config(self):
        result = {"status": "COMPLETED", "policy": "mira_sw_ucb",
                  "invalid_power_or_sic_count": 0, "post_schedule_power_drop_count": 0,
                  "max_transmitted_power_normalized": 1., "phi_theta_deg": 20., "B_penalty": .05}
        with self.assertRaisesRegex(RuntimeError, "invalid grouping action"):
            _validate_run(result, [{"slot": 0}], [{"slot": 0, "cohort_id": 0, "action": 20}],
                          1, load_config(CONFIG))

    def test_failed_model_never_produces_a_completed_comparison(self):
        config = load_config(CONFIG)
        config = config.with_overrides({"algorithm.reward_bounds": {
            "lower": -1000, "upper": 1000, "identity": "artificial-unit-fixture",
            "context": reward_context(config, 0),
        }})
        with tempfile.TemporaryDirectory() as temp:
            with patch("mira_leo_study.paper_results.run_simulation",
                       side_effect=ModelConsistencyError("artificial-infeasible-power")):
                with self.assertRaises(ModelConsistencyError):
                    run_paper_results(config, temp, slots=3)
            import json
            root = Path(temp) / "debug" / "slots_3_seed_0"
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "MODEL_CONSISTENCY_ERROR")
            self.assertFalse((root / "formal_results/convergence_long_run_aoi/comparison_summary.csv").exists())


class EpochSimulationIntegrationTests(unittest.TestCase):
    @simulation_test
    def test_resume_at_150_matches_300_slots(self):
        config = load_config(CONFIG).with_overrides({"algorithm.reward_bounds": {
            "lower": -1e9, "upper": 1e9, "identity": "integration-only",
        }})
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            # A controlled all-nonedge environment makes this a cadence/resume test,
            # independent of genuinely infeasible mixed-class power instances.
            def geometry(ecef, devices):
                return np.full((8, len(devices)), 800.), np.full((8, len(devices)), 50.)
            with patch("mira_leo_study.simulation.link_geometry", side_effect=geometry):
                complete = run_simulation(config, channel_seed=0, policy="mira_sw_ucb",
                                           max_slots=300, run_directory=left)["summary"]
                paused = run_simulation(config, channel_seed=0, policy="mira_sw_ucb", max_slots=300,
                                         run_directory=right, stop_after_slots=150)["summary"]
                self.assertEqual(paused["status"], "CHECKPOINTED")
                checkpoint = pickle.loads((Path(right) / "checkpoint.pkl").read_bytes())
                self.assertEqual(checkpoint["schema_version"], CHECKPOINT_SCHEMA_VERSION)
                self.assertTrue(all(n == 50 for n in checkpoint["state"]["epoch_reward_count"].values()))
                resumed = run_simulation(config, channel_seed=0, policy="mira_sw_ucb", max_slots=300,
                                          run_directory=right, resume=True)["summary"]
            for filename in ("slot_metrics.csv", "convergence_slot_metrics.csv", "bandit_observations.csv"):
                self.assertEqual((Path(left) / filename).read_bytes(), (Path(right) / filename).read_bytes())
            for key in ("bandit_observations", "action_decision_count", "regrouping_events_total", "environment_trace_hash"):
                self.assertEqual(complete[key], resumed[key])


if __name__ == "__main__":
    unittest.main()
