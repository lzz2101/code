from __future__ import annotations

import unittest

from mira_leo_v2a3_verified.algorithms.sw_ucb import SWUCB


class SWUCBWindowTests(unittest.TestCase):
    def test_sw_01_basic_window_contents(self) -> None:
        policy = SWUCB((1, 2), tau=3, satellite_id=0)
        policy.update(1, 1, 0.2)
        policy.update(2, 2, 0.8)
        policy.update(3, 1, 0.4)

        self.assertEqual(policy.get_count(4, 1), 2)
        self.assertAlmostEqual(policy.get_mean_reward(4, 1), 0.3)
        self.assertEqual(policy.get_count(4, 2), 1)
        self.assertAlmostEqual(policy.get_mean_reward(4, 2), 0.8)

        policy.update(4, 2, 0.6)
        self.assertEqual(policy.get_count(5, 1), 1)
        self.assertAlmostEqual(policy.get_mean_reward(5, 1), 0.4)
        self.assertEqual(policy.get_count(5, 2), 2)
        self.assertAlmostEqual(policy.get_mean_reward(5, 2), 0.7)

    def test_sw_02_unselected_arm_expires_by_global_slot(self) -> None:
        policy = SWUCB((1, 2), tau=3)
        policy.update(1, 1, 0.9)
        policy.update(2, 2, 0.5)
        policy.update(3, 2, 0.5)
        policy.update(4, 2, 0.5)

        self.assertEqual(policy.get_count(5, 1), 0)
        self.assertEqual(policy.get_count(5, 2), 3)

    def test_sw_03_window_boundary_excludes_current_and_stale_slots(self) -> None:
        policy = SWUCB((1, 2), tau=3)
        for slot in range(1, 5):
            policy.update(slot, 1 if slot % 2 else 2, slot / 10.0)

        records = policy.get_window_records(5)
        self.assertEqual([record.slot for record in records], [2, 3, 4])

    def test_sw_04_satellite_policies_are_independent(self) -> None:
        sat0 = SWUCB((1, 2), tau=3, satellite_id=0)
        sat1 = SWUCB((1, 2), tau=3, satellite_id=1)
        for slot in range(3):
            sat0.update(slot, 1, 0.9)
            sat1.update(slot, 1, 0.1)

        self.assertAlmostEqual(sat0.get_mean_reward(3, 1), 0.9)
        self.assertAlmostEqual(sat1.get_mean_reward(3, 1), 0.1)

    def test_sw_05_expired_arm_is_explored(self) -> None:
        policy = SWUCB((1, 2), tau=3)
        policy.update(1, 1, 0.9)
        policy.update(2, 2, 0.5)
        policy.update(3, 2, 0.5)
        policy.update(4, 2, 0.5)

        self.assertEqual(policy.select_arm(5), 1)

    def test_sw_06_action_sequence_is_reproducible(self) -> None:
        def sequence() -> list[int]:
            policy = SWUCB((1, 2, 4), tau=4)
            actions = []
            for slot in range(12):
                arm = policy.select_arm(slot)
                actions.append(arm)
                policy.update(slot, arm, 0.2 + 0.1 * (arm == 2))
            return actions

        self.assertEqual(sequence(), sequence())


if __name__ == "__main__":
    unittest.main()
