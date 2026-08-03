from __future__ import annotations

import unittest
from dataclasses import replace

from mira_leo_v2a3_verified.aoi.metrics import (
    compute_regrouping_reward_details,
    normalize_reward_fixed,
)
from mira_leo_v2a3_verified.config import V2Config
from mira_leo_v2a3_verified.main import run_mira_leo_v2a3_verified
from mira_leo_v2a3_verified.validation.validators import (
    ValidationError,
    validate_reward_record,
)


def reward_details(success: int, aoi: float, edge: int) -> dict[str, object]:
    return compute_regrouping_reward_details(
        success_count=success,
        weighted_sum_aoi=aoi,
        edge_count=edge,
        B=0.2,
        normalize=True,
        reward_min=-1000.0,
        reward_max=100.0,
    )


class RewardBoundTests(unittest.TestCase):
    def test_rw_01_fixed_bound_examples(self) -> None:
        self.assertEqual(normalize_reward_fixed(-50.0, -150.0, -50.0), 1.0)
        self.assertEqual(normalize_reward_fixed(-100.0, -150.0, -50.0), 0.5)
        self.assertEqual(normalize_reward_fixed(-150.0, -150.0, -50.0), 0.0)

    def test_rw_02_lower_aoi_has_higher_reward(self) -> None:
        self.assertGreater(
            reward_details(2, 10.0, 1)["normalized_reward"],
            reward_details(2, 20.0, 1)["normalized_reward"],
        )

    def test_rw_03_more_success_does_not_reduce_reward(self) -> None:
        self.assertGreaterEqual(
            reward_details(3, 10.0, 1)["normalized_reward"],
            reward_details(1, 10.0, 1)["normalized_reward"],
        )

    def test_rw_04_more_edge_groups_does_not_increase_reward(self) -> None:
        self.assertLessEqual(
            reward_details(1, 10.0, 5)["normalized_reward"],
            reward_details(1, 10.0, 0)["normalized_reward"],
        )

    def test_rw_05_legal_boundaries_are_finite(self) -> None:
        cases = [(0, 0.0, 0), (0, 100.0, 10), (10, 0.0, 10)]
        for success, aoi, edge in cases:
            with self.subTest(success=success, aoi=aoi, edge=edge):
                value = reward_details(success, aoi, edge)["normalized_reward"]
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_rw_06_validator_rejects_material_out_of_bounds_values(self) -> None:
        base = {
            "slot": 2,
            "satellite_id": 0,
            "action": 4,
            "raw_weighted_sum_aoi": 10.0,
            "raw_reward": -10.0,
            "normalization_parameters": {"method": "test"},
        }
        for value in (-0.2, 1.2):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    validate_reward_record(
                        {**base, "normalized_reward": value}, strict=True
                    )

    def test_rw_07_normalization_has_no_runtime_minmax_state(self) -> None:
        first = reward_details(2, 20.0, 1)
        second = reward_details(2, 20.0, 1)
        self.assertEqual(first, second)

    def test_runtime_rewards_given_to_each_policy_are_bounded(self) -> None:
        cfg = replace(V2Config(), T=30, random_seed=7)
        result = run_mira_leo_v2a3_verified(cfg, fading=False)
        for policy in result["sw_ucb_policies"]:
            self.assertTrue(policy.observed_rewards)
            self.assertTrue(all(0.0 <= value <= 1.0 for value in policy.observed_rewards))


if __name__ == "__main__":
    unittest.main()
