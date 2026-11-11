"""Deterministic component checks; these never enter the simulation slot loop."""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mira_leo_study.algorithms import build_group_partition, materialize_groups_from_partition
from mira_leo_study.association import AssociationTracker
from mira_leo_study.channel import (
    expected_cross_link_interference, expected_cross_link_interference_by_ue,
    external_interferer_powers,
)
from mira_leo_study.calibration import evaluate_counterfactual_slot
from mira_leo_study.config import load_config
from mira_leo_study.errors import ConfigurationError
from mira_leo_study.geometry import generate_devices
from mira_leo_study.models import AssociationDecision, Group, Route, SubscriptionTable
from mira_leo_study.reward_pilot import apply_reward_bounds, reward_context
from mira_leo_study.scenario import generate_subscriptions
from mira_leo_study.simulation import (
    CHECKPOINT_SCHEMA_VERSION, _initial_state, _new_normalizers, build_parameter_snapshot,
    validate_checkpoint_identity,
)

CONFIG = Path(__file__).parents[1] / "configs/base.yaml"


class SinglePopulationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)

    def test_all_120_are_active_in_three_disjoint_messages(self):
        subscriptions = generate_subscriptions(self.config)
        self.assertEqual(set(subscriptions.message_members), {0, 1, 2})
        self.assertTrue(all(len(v) == 40 for v in subscriptions.message_members.values()))
        members = np.concatenate(list(subscriptions.message_members.values()))
        np.testing.assert_array_equal(np.sort(members), np.arange(120))
        np.testing.assert_array_equal(np.sort(subscriptions.active_device_ids), np.arange(120))
        for message, subset in subscriptions.message_members.items():
            self.assertTrue(np.all(subscriptions.message_by_device[subset] == message))
        for model in (SubscriptionTable, Group, AssociationDecision):
            self.assertFalse(any("cohort" in f.name for f in fields(model)))
        self.assertNotIn("cohort", str(self.config.to_dict()))

    def test_seeds_and_hashes_are_reproducible_and_sensitive(self):
        a, b = generate_subscriptions(self.config), generate_subscriptions(self.config)
        self.assertEqual(a.subscription_hash, b.subscription_hash)
        np.testing.assert_array_equal(a.message_by_device, b.message_by_device)
        changed = self.config.with_overrides({"devices.subscription_seed": 1043})
        self.assertNotEqual(a.subscription_hash, generate_subscriptions(changed).subscription_hash)
        x, y = generate_devices(self.config), generate_devices(self.config)
        self.assertEqual(len(x.ids), 120)
        self.assertEqual(x.layout_hash, y.layout_hash)
        np.testing.assert_array_equal(x.ecef_km, y.ecef_km)
        changed = self.config.with_overrides({"devices.layout_seed": 43})
        self.assertNotEqual(x.layout_hash, generate_devices(changed).layout_hash)

    def test_invalid_counts_actions_and_legacy_config_are_rejected(self):
        for key, value in (("devices.unique_count", 300), ("devices.unique_count", 120.5),
                           ("services.messages_per_active_satellite", 0),
                           ("services.subscribers_per_message", True),
                           ("services.cohort_count", 1),
                           ("algorithm.action_set", [1.5, 2, 4, 5, 8, 10]),
                           ("channel.interferer_power_mode", "scheduled")):
            with self.subTest(key=key), self.assertRaises(ConfigurationError):
                self.config.with_overrides({key: value})

    def test_actions_and_edge_labels_use_current_serving_geometry(self):
        subscriptions = generate_subscriptions(self.config)
        amplitudes = np.ones((8, 120))
        elevation = np.full((8, 120), 14.)
        elevation[1] = 15.
        for action in self.config.get("algorithm.action_set"):
            partition = build_group_partition(self.config, 0, action, subscriptions.message_members, amplitudes)
            self.assertEqual(len(partition), 3 * action)
            self.assertLessEqual(len(partition), 30)
            a = materialize_groups_from_partition(self.config, 0, partition, amplitudes, elevation)
            b = materialize_groups_from_partition(self.config, 1, partition, amplitudes, elevation)
            self.assertTrue(all(g.is_edge for g in a))
            self.assertTrue(all(not g.is_edge for g in b))
            self.assertTrue(all(g.satellite_index == 1 and len(g.key) == 2 for g in b))
            self.assertEqual([g.key for g in a], [g.key for g in b])

    def test_snapshot_declares_new_model_and_old_bounds_cannot_apply(self):
        snapshot = build_parameter_snapshot(self.config, "mira_sw_ucb", 0)
        for key, value in (("total_ue", 120), ("message_count", 3),
                           ("subscribers_per_message", 40), ("serving_satellite_count", 1),
                           ("interference_mode", "fixed_full_load"), ("interferer_power_normalized", 1.)):
            self.assertEqual(snapshot[key], value)
        with self.assertRaises(ValueError):
            apply_reward_bounds(self.config, {"phi_theta_deg": 20., "B_penalty": .05,
                "reward_context": "old-300ue-artifact", "reward_identity": "old",
                "reward_lower": -1000., "reward_upper": 1.}, 0)
        self.assertEqual(reward_context(self.config, 0), reward_context(load_config(CONFIG), 0))

    def test_old_checkpoint_and_contextless_bounds_cannot_be_reused(self):
        identity = dict(config_id=self.config.config_id, channel_seed=0, slot_limit=3,
                        policy="fixed_grouping", fixed_action=4, save_group_trace=False)
        validate_checkpoint_identity({**identity, "schema_version": CHECKPOINT_SCHEMA_VERSION}, **identity)
        with self.assertRaises(ValueError):
            validate_checkpoint_identity({**identity, "schema_version": 5}, **identity)
        contextless = self.config.with_overrides({"algorithm.reward_bounds": {
            "lower": -1000., "upper": 1., "identity": "legacy"}})
        with self.assertRaisesRegex(ValueError, "context"):
            _new_normalizers(contextless, 8)

    def test_policy_initial_states_have_one_population_and_no_neighbor_groups(self):
        subscriptions = generate_subscriptions(self.config)
        scenario = SimpleNamespace(devices=SimpleNamespace(ids=tuple(range(120))),
                                   subscriptions=subscriptions,
                                   constellation=SimpleNamespace(elements=tuple(range(8))))
        config = self.config.with_overrides({"algorithm.reward_bounds": {
            "lower": -1000., "upper": 1., "identity": "artificial-state-only",
            "context": reward_context(self.config, 0)}})
        for policy in ("mira_sw_ucb", "random_grouping", "fixed_grouping"):
            with self.subTest(policy=policy):
                state = _initial_state(config, scenario, policy, 3, 0)
                self.assertEqual(state["ages"].shape, (120,))
                self.assertTrue(np.all(np.isfinite(state["ages"])))
                self.assertEqual(state["aoi_samples"].shape, (3, 120))
                self.assertIsNone(state["held_partition"])
                self.assertIsNone(state["held_action"])
                self.assertFalse(any("cohort" in key or "groups_by" in key for key in state))

    def test_one_invisible_member_blocks_entire_multicast_group(self):
        # A single synthetic counterfactual with mocked delivery, no simulator.
        members = {m: np.arange(m*40, (m+1)*40) for m in range(3)}
        groups = [Group(0, m, 0, ids, 1., 50., False) for m, ids in members.items()]
        scenario = SimpleNamespace(devices=SimpleNamespace(ids=tuple(range(120))),
            subscriptions=SimpleNamespace(active_device_ids=np.arange(120), message_members=members),
            constellation=SimpleNamespace(elements=[SimpleNamespace(satellite_id=f"sat_{i+1:02d}") for i in range(8)]))
        elevation = np.full((8, 120), 50.)
        elevation[0, 0] = 9.99
        def decoded(config, current):
            for group in current:
                group.success = True
        diagnostic = dict(valid=True, status="OK", actual_total_power=1., transmission_attempted=True)
        with patch("mira_leo_study.calibration.make_groups", return_value=groups), \
             patch("mira_leo_study.calibration.allocate_power", return_value=diagnostic), \
             patch("mira_leo_study.calibration.decode_groups", side_effect=decoded):
            result = evaluate_counterfactual_slot(self.config, scenario, slot=0, seed=0,
                ages_before=np.full(120, 100.), association=SimpleNamespace(serving_satellite=0, handover=False),
                distance=np.full((8, 120), 800.), elevation=elevation,
                routes={0: Route(True, (0,), 0., 0, 0., 0.)}, action=1,
                channel_amplitude=np.ones((8, 120)))
        self.assertFalse(groups[0].success)
        self.assertTrue(groups[1].success and groups[2].success)
        self.assertTrue(np.all(result["ages_after"][:40] == 101.))
        self.assertTrue(np.all(result["ages_after"][40:] < 100.))


class SingleAssociationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)
        self.tracker = AssociationTracker(self.config, generate_subscriptions(self.config))

    def test_coverage_first_and_partial_fallback_with_deterministic_ties(self):
        e = np.full((8, 120), 5.)
        e[0, :119] = 80.  # Higher average, but incomplete footprint.
        e[1] = 10.
        first = self.tracker.update(0, e)
        self.assertEqual(first.serving_satellite, 1)
        self.assertFalse(first.handover)
        self.assertFalse(first.fallback_used)
        e[1, -1] = 4.
        e[2] = e[0]  # Equal coverage and minimum: lower satellite index wins.
        decision = self.tracker.update(1, e)
        self.assertEqual(decision.serving_satellite, 0)
        self.assertTrue(decision.handover)
        self.assertTrue(decision.fallback_used)
        self.assertEqual(decision.outage_receivers, 1)

    def test_hysteresis_and_ttt_and_visibility_loss(self):
        e = np.full((8, 120), 5.)
        e[0] = 30.; e[1] = 29.
        self.assertEqual(self.tracker.update(0, e).serving_satellite, 0)
        e[1] = 30.5
        self.assertEqual(self.tracker.update(1, e).serving_satellite, 0)
        e[1] = 32.
        for slot in range(2, 101):
            self.assertEqual(self.tracker.update(slot, e).serving_satellite, 0)
        event = self.tracker.update(101, e)
        self.assertEqual(event.serving_satellite, 1)
        self.assertTrue(event.handover)
        self.assertFalse(self.tracker.update(102, e).handover)
        e[1, 0] = 9.99
        self.assertEqual(self.tracker.update(103, e).serving_satellite, 0)

    def test_no_coverage_reports_all_120_outages(self):
        result = self.tracker.update(0, np.full((8, 120), 0.))
        self.assertEqual(result.serving_satellite, 0)
        self.assertEqual(result.outage_receivers, 120)
        self.assertTrue(result.fallback_used)


class ExpectedInterferenceTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)
        self.mean = self.config.get("derived.shadowed_rician_mean_power")
        self.gains = np.zeros((8, 4))
        self.visible = np.zeros((8, 4), dtype=bool)
        self.members = np.arange(4)

    def calculate(self, serving=0, powers=None):
        if powers is None:
            powers = external_interferer_powers(self.config, 8, serving)
        return expected_cross_link_interference(self.config, serving, self.members,
                                               self.gains, self.visible, powers)

    def test_one_visible_neighbor(self):
        self.gains[1] = .03
        self.visible[1] = True
        self.assertAlmostEqual(self.calculate(), .03 * self.mean)

    def test_two_visible_neighbors_add_per_ue(self):
        self.gains[1] = .03; self.gains[2] = .02
        self.visible[1:3] = True
        self.assertAlmostEqual(self.calculate(), .05 * self.mean)

    def test_invisible_neighbors_contribute_zero(self):
        self.gains[:] = 1e6
        self.assertEqual(self.calculate(), 0.)

    def test_serving_signal_is_excluded_even_with_huge_power(self):
        self.gains[0] = 1e6; self.visible[0] = True
        self.gains[1] = .02; self.visible[1] = True
        powers = np.ones(8); powers[0] = 1e9
        self.assertAlmostEqual(self.calculate(powers=powers), .02 * self.mean)

    def test_handover_changes_exclusion_and_old_serving_can_interfere(self):
        self.gains[0] = .03; self.gains[1] = .02
        self.visible[:2] = True
        self.assertAlmostEqual(self.calculate(serving=0), .02 * self.mean)
        self.assertAlmostEqual(self.calculate(serving=1), .03 * self.mean)

    def test_group_worst_ue_is_max_of_neighbor_sum(self):
        self.gains[1] = np.array([.01, .03, .08, .02]) / self.mean
        self.visible[1] = True
        self.assertAlmostEqual(self.calculate(), .08)

    def test_max_of_sum_is_not_sum_of_neighbor_maxima(self):
        self.gains[1] = [.07, .01, 0., 0.]
        self.gains[2] = [.01, .06, 0., 0.]
        self.visible[1:3] = True
        self.assertAlmostEqual(self.calculate(), .08 * self.mean)
        self.assertNotAlmostEqual(self.calculate(), .13 * self.mean)

    def test_outage_and_policy_cannot_change_external_power(self):
        self.gains[1] = .04; self.visible[1] = True
        for policy in ("mira_sw_ucb", "random_grouping", "fixed_grouping"):
            # No policy, group list, actual serving TX or neighbor traffic is an input.
            with self.subTest(policy=policy):
                actual_serving_power = 0.
                powers = external_interferer_powers(self.config, 8, 0)
                self.assertEqual(actual_serving_power, 0.)
                np.testing.assert_array_equal(powers, [0., 1., 1., 1., 1., 1., 1., 1.])
                values = expected_cross_link_interference_by_ue(
                    self.config, 0, self.members, self.gains, self.visible, powers)
                np.testing.assert_allclose(values, np.full(4, .04 * self.mean))

    def test_shape_finite_nonnegative_and_index_validation(self):
        powers = np.ones(8)
        cases = ((self.gains, self.visible[:, :2], powers),
                 (self.gains, self.visible, np.ones(7)),
                 (np.full((8, 4), np.nan), self.visible, powers),
                 (self.gains, self.visible, -powers))
        for gains, visible, power in cases:
            with self.assertRaises(ValueError):
                expected_cross_link_interference(self.config, 0, self.members, gains, visible, power)
        with self.assertRaises(ValueError):
            expected_cross_link_interference(self.config, 0, np.array([4]), self.gains, self.visible, powers)
        self.assertEqual(expected_cross_link_interference(
            self.config, 0, np.array([], dtype=int), self.gains, self.visible, powers), 0.)


if __name__ == "__main__":
    unittest.main()
