from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mira_leo_study.config import load_config
from mira_leo_study.errors import ConfigurationError
from mira_leo_study.simulation import run_simulation
from mira_leo_study.tests import simulation_test


CONFIG = Path(__file__).parents[1] / "configs" / "base.yaml"


class ProtocolSimulationTests(unittest.TestCase):
    def setUp(self):
        base = load_config(CONFIG)
        self.config = base.with_overrides(
            {
                "experiment.duration_s": 0.02,
                "experiment.warmup_report_s": 0.0,
                "execution.slot_chunk_size": 2,
            }
        )

    def test_phi_study_uses_one_fixed_channel_seed(self):
        self.assertEqual(self.config.get("randomness.fixed_channel_seed"), 0)
        self.assertIsNone(self.config.get("randomness.validation_channel_seeds"))
        self.assertIsNone(self.config.get("randomness.test_channel_seeds"))
        self.assertIsNone(self.config.get("randomness.pilot_channel_seeds"))

    def test_invalid_action_configuration_stops_early(self):
        with self.assertRaises(ConfigurationError):
            self.config.with_overrides(
                {"algorithm.action_set_name": "coarse", "algorithm.action_set": [1, 3]}
            )

    @simulation_test
    def test_short_end_to_end_run_has_no_nonfinite_metrics(self):
        result = run_simulation(
            self.config,
            channel_seed=0,
            policy="fixed",
            fixed_action=4,
            collect_slot_metrics=True,
        )
        self.assertEqual(result["summary"]["status"], "COMPLETED")
        self.assertEqual(result["summary"]["invalid_power_or_sic_count"], 0)
        self.assertEqual(result["summary"]["post_schedule_power_drop_count"], 0)
        self.assertEqual(result["summary"]["slots_completed"], 2)
        self.assertEqual(len(result["slot_metrics"]), 2)
        self.assertGreater(result["summary"]["weighted_mean_aoi_ms"], 0.0)
        by_slot = {}
        for row in result["slot_metrics"]:
            by_slot[int(row["slot"])] = float(row["aggregated_weighted_sum_aoi_ms"])
        self.assertAlmostEqual(
            result["summary"]["weighted_mean_aoi_after_warmup_ms"],
            sum(by_slot.values()) / len(by_slot),
        )
        self.assertNotIn("rate_required_bps_hz", result["summary"])
        self.assertNotIn("receiver_delivered_payload_bits_per_s", result["summary"])

    @simulation_test
    def test_checkpoint_resume_matches_uninterrupted_metrics(self):
        keys = [
            "weighted_mean_aoi_ms",
            "aoi_p50_ms",
            "aoi_p95_ms",
            "packet_success_ratio",
            "capacity_goodput_bps",
            "successful_group_count",
            "scheduled_group_count",
            "handover_count",
            "coverage_outage_receiver_slots",
            "violation_counts",
            "action_histogram",
        ]
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            with patch("mira_leo_study.simulation.csv_to_parquet", return_value=None):
                complete = run_simulation(
                    self.config,
                    channel_seed=3,
                    policy="fixed",
                    fixed_action=4,
                    output_directory=left,
                )["summary"]
                paused = run_simulation(
                    self.config,
                    channel_seed=3,
                    policy="fixed",
                    fixed_action=4,
                    output_directory=right,
                    stop_after_slots=1,
                )["summary"]
                self.assertEqual(paused["status"], "CHECKPOINTED")
                resumed = run_simulation(
                    self.config,
                    channel_seed=3,
                    policy="fixed",
                    fixed_action=4,
                    output_directory=right,
                    resume=True,
                )["summary"]
            for key in keys:
                self.assertEqual(complete[key], resumed[key], key)


if __name__ == "__main__":
    unittest.main()
