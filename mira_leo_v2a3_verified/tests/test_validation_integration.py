from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from mira_leo_v2a3_verified.config import V2Config
from mira_leo_v2a3_verified.main import run_mira_leo_v2a3_verified
from mira_leo_v2a3_verified.validation.run_validation import (
    run_controlled_scenarios,
)


class ValidationIntegrationTests(unittest.TestCase):
    def test_int_01_through_int_07_controlled_scenarios_pass(self) -> None:
        rows, coverage = run_controlled_scenarios(strict=True)
        self.assertEqual(len(rows), 7)
        self.assertTrue(all(row["status"] == "PASS" for row in rows))
        self.assertTrue(all(coverage.values()))

    def test_runtime_has_per_satellite_validation_records(self) -> None:
        cfg = replace(V2Config(), T=20, random_seed=11)
        result = run_mira_leo_v2a3_verified(
            cfg,
            policy="fixed",
            fixed_action=4,
            fading=False,
            validate=True,
            strict_validation=True,
        )
        records = result["validation_satellite_records"]
        self.assertEqual(len(records), cfg.T * cfg.L)
        self.assertEqual({record["satellite_id"] for record in records}, {0, 1})
        self.assertTrue(
            all(record["total_allocated_power"] <= cfg.P_sat + 1e-9 for record in records)
        )
        self.assertTrue(all(0.0 <= record["normalized_reward"] <= 1.0 for record in records))
        self.assertEqual(sum(record["sic_power_order_violations"] for record in records), 0)
        self.assertEqual(sum(record["sic_decode_violations"] for record in records), 0)
        self.assertTrue(np.isfinite(result["histories"]["weighted_sum_aoi"]).all())


if __name__ == "__main__":
    unittest.main()
