from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mira_leo_study.calibration import _derive_calibration
from mira_leo_study.channel import serving_channel_amplitude
from mira_leo_study.config import (
    PHI_THETA_CANDIDATES_DEG,
    load_config,
    validate_phi_study_config,
)
from mira_leo_study.errors import BaselineUnavailableError, ConfigurationError
from mira_leo_study.geometry import link_geometry
from mira_leo_study.phi_study import (
    resolved_phi_config,
    run_phi_validation_sweep,
    select_phi_theta,
    write_phi_report,
)
from mira_leo_study.plotting import load_phi_aoi_trace
from mira_leo_study.scenario import StudyScenario
from mira_leo_study.simulation import (
    _random_action,
    build_parameter_snapshot,
    canonical_policy_name,
    run_simulation,
)
from mira_leo_study.tests import simulation_test


CONFIG = Path(__file__).parents[1] / "configs" / "base.yaml"


class PhiStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Historical API regression fixture. The production base has no searches.
        cls.config = load_config(CONFIG).with_overrides({
            "search.phi_theta_deg": PHI_THETA_CANDIDATES_DEG,
            "algorithm.B0_calibration_quantile": 0.75,
            "algorithm.beta_B": 1.0,
            "algorithm.action_set_name": "full",
            "algorithm.action_set": [1, 2, 4, 5, 8, 10, 20, 40],
            "algorithm.regrouping_interval_s": 0.01,
            "algorithm.power_mode": "literal_with_checks",
        })

    def test_formal_parameter_snapshot(self):
        validate_phi_study_config(self.config)
        self.assertEqual(self.config.get("experiment.slot_dt_s"), 0.01)
        self.assertEqual(self.config.get("experiment.duration_s"), 480.0)
        self.assertEqual(self.config.get("experiment.warmup_report_s"), 10.0)
        self.assertEqual(self.config.get("derived.handover_delay_slots"), 2.5)
        self.assertEqual(self.config.get("derived.fixed_isl_per_hop_slots"), 0.5)
        self.assertEqual(self.config.get("channel.carrier_hz"), 20e9)
        self.assertEqual(self.config.get("services.bandwidth_hz"), 50e6)
        self.assertEqual(self.config.get("radio.scs_hz"), 60e3)
        self.assertEqual(self.config.get("algorithm.R_theta"), 1e-5)
        self.assertEqual(self.config.get("algorithm.V_c"), 0.7)
        self.assertEqual(self.config.get("algorithm.V_nc"), 0.3)
        self.assertEqual(self.config.get("algorithm.V_phi"), 5e-4)
        self.assertEqual(self.config.get("services.messages_per_active_satellite"), 3)
        self.assertEqual(self.config.get("services.subscribers_per_message"), 40)
        self.assertEqual(self.config.get("services.weights"), [0.7, 0.2, 0.1])
        self.assertEqual(self.config.get("search.phi_theta_deg"), PHI_THETA_CANDIDATES_DEG)
        self.assertEqual(
            self.config.get("channel.shadowed_rician"),
            {"b0": 0.126, "Omega": 0.835, "m": 10.1},
        )
        self.assertIsNone(self.config.get("services.payload_bytes"))

    def test_parameter_snapshot_contains_no_payload(self):
        resolved = self.config.with_overrides(
            {
                "algorithm.B_penalty": 0.25,
                "algorithm.reward_bounds": {"lower": -10.0, "upper": 2.0, "identity": "phi20"},
            }
        )
        snapshot = build_parameter_snapshot(resolved, "mira_sw_ucb", 0)
        self.assertFalse(any("payload" in key.lower() for key in snapshot))
        self.assertEqual(snapshot["schedule_capacity_integer"], 62)
        self.assertNotIn("phi_theta_candidates_deg", snapshot)
        self.assertEqual(snapshot["B_penalty"], 0.25)
        self.assertIn("Algorithm 2", snapshot["capacity_semantics"])

    def test_formal_validation_rejects_parameter_drift_and_seed_sweeps(self):
        with self.assertRaises(ConfigurationError):
            validate_phi_study_config(
                self.config.with_overrides({"services.weights": [0.5, 0.3, 0.2]})
            )
        with self.assertRaises(ConfigurationError):
            validate_phi_study_config(
                self.config.with_overrides(
                    {"randomness.validation_channel_seeds": [0, 1]}
                )
            )

    def test_angle_specific_calibration_identity_and_frozen_config(self):
        identities = set()
        for phi in PHI_THETA_CANDIDATES_DEG:
            angle = self.config.with_overrides({"algorithm.phi_theta_deg": float(phi)})
            samples = [
                {"seed": 0, "window_id": 0, "snapshot_slot": 0, "cohort_id": 0, "action": 1, "q": -10.0, "edge_group_count": 1, "feasible": True},
                {"seed": 0, "window_id": 0, "snapshot_slot": 0, "cohort_id": 0, "action": 2, "q": -8.0, "edge_group_count": 2, "feasible": True},
            ]
            calibration = _derive_calibration(angle, samples)
            self.assertTrue(calibration["B_identifiable"])
            self.assertEqual(calibration["ratio_sample_count"], 1)
            identities.add(calibration["reward_identity"])
            entry = {
                "status": "OK",
                "B_penalty_star": calibration["B_penalty_star"],
                "reward_lower": calibration["reward_lower"],
                "reward_upper": calibration["reward_upper"],
                "reward_bounds_identity": calibration["reward_identity"],
            }
            resolved = resolved_phi_config(self.config, phi, entry)
            self.assertEqual(resolved.get("algorithm.B_penalty"), calibration["B_penalty_star"])
            self.assertEqual(resolved.get("algorithm.reward_bounds.identity"), calibration["reward_identity"])
        self.assertEqual(len(identities), len(PHI_THETA_CANDIDATES_DEG))

    def test_unidentifiable_calibration_does_not_fallback_to_one(self):
        result = _derive_calibration(self.config, [])
        self.assertFalse(result["B_identifiable"])
        self.assertIsNone(result["B_penalty_star"])
        self.assertEqual(result["ratio_sample_count"], 0)
        self.assertEqual(result["B_status"], "B_NOT_IDENTIFIABLE_FROM_PILOT")

    def test_fixed_channel_sample_is_identical_across_phi(self):
        scenario = StudyScenario(self.config)
        _, ecef = scenario.constellation.state(1.0)
        distance, _ = link_geometry(ecef, scenario.devices.ecef_km)
        reference = serving_channel_amplitude(self.config, 0, 100, distance)
        for phi in PHI_THETA_CANDIDATES_DEG:
            candidate = self.config.with_overrides({"algorithm.phi_theta_deg": float(phi)})
            actual = serving_channel_amplitude(candidate, 0, 100, distance)
            np.testing.assert_array_equal(reference, actual)

    def test_random_policy_is_deterministic_and_always_legal(self):
        actions = self.config.get("algorithm.action_set")
        first = [_random_action(actions, 0, slot, 3) for slot in range(100)]
        second = [_random_action(actions, 0, slot, 3) for slot in range(100)]
        self.assertEqual(first, second)
        self.assertTrue(set(first).issubset(set(actions)))

    @simulation_test
    def test_mira_selects_only_actions_while_B_and_bounds_stay_frozen(self):
        short = self.config.with_overrides(
            {
                "experiment.duration_s": 0.09,
                "experiment.warmup_report_s": 0.0,
                "algorithm.B_penalty": 0.375,
                "algorithm.reward_bounds": {
                    "lower": -1e9,
                    "upper": 1e9,
                    "identity": "phi30-frozen",
                },
            }
        )
        result = run_simulation(
            short,
            channel_seed=0,
            policy="mira_sw_ucb",
            collect_slot_metrics=True,
        )
        selected = {int(row["action"]) for row in result["slot_metrics"]}
        self.assertTrue(selected.issubset(set(short.get("algorithm.action_set"))))
        self.assertEqual(result["summary"]["B_penalty"], 0.375)
        self.assertEqual(result["summary"]["reward_bounds_identity"], "phi30-frozen")
        self.assertEqual(short.get("algorithm.B_penalty"), 0.375)

    def test_selection_requires_full_runs_and_uses_deterministic_tie_break(self):
        rows = []
        b_entries = {}
        lowest_two = set(PHI_THETA_CANDIDATES_DEG[:2])
        first_phi, second_phi = PHI_THETA_CANDIDATES_DEG[:2]
        for phi in PHI_THETA_CANDIDATES_DEG:
            rows.append(
                {
                    "phi_theta_deg": phi,
                    "policy": "mira_sw_ucb",
                    "status": "COMPLETED",
                    "fixed_channel_seed": 0,
                    "slots_completed": self.config.slot_count,
                    "weighted_mean_aoi_after_warmup_ms": 10.0 if phi in lowest_two else 20.0 + phi,
                    "mean_interference_power": 2.0 if phi == first_phi else 1.0,
                    "regrouping_events_total": 5,
                }
            )
            b_entries[phi] = {"B_penalty_star": phi / 100.0}
        selected = select_phi_theta(rows, self.config, b_entries)
        self.assertEqual(selected["selected_phi_theta_deg"], second_phi)
        self.assertIsNotNone(selected["tie_note"])
        rows[0]["slots_completed"] -= 1
        with self.assertRaises(ValueError):
            select_phi_theta(rows, self.config, b_entries)

    def test_debug_sweep_is_isolated_and_writes_explicit_baseline_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            b_path = root / "B_by_phi.json"
            entries = {
                str(phi): {
                    "phi_theta_deg": phi,
                    "status": "OK",
                    "B_penalty_star": phi / 100.0,
                    "reward_lower": -100.0,
                    "reward_upper": 10.0,
                    "reward_bounds_identity": f"phi-{phi}",
                }
                for phi in PHI_THETA_CANDIDATES_DEG
            }
            b_path.write_text(
                json.dumps(
                    {
                        "fixed_channel_seed": 0,
                        "B_penalty_star_by_phi": entries,
                    }
                ),
                encoding="utf-8",
            )

            def fake_run(config, *, channel_seed, policy, **kwargs):
                return {
                    "summary": {
                        "status": "COMPLETED",
                        "config_id": config.config_id,
                        "policy": policy,
                        "channel_seed": channel_seed,
                        "slots_completed": 1,
                        "B_penalty_star": config.get("algorithm.B_penalty"),
                        "reward_bounds_identity": config.get(
                            "algorithm.reward_bounds.identity"
                        ),
                    }
                }

            with patch("mira_leo_study.phi_study.run_simulation", side_effect=fake_run):
                result = run_phi_validation_sweep(
                    self.config,
                    b_path,
                    root,
                    max_slots=1,
                )
            debug_root = root / "debug" / "max_slots_1"
            self.assertEqual(Path(result["artifact_root"]), debug_root)
            self.assertFalse((root / "comparison").exists())
            baseline_path = debug_root / "comparison" / "baseline_summary_by_phi.csv"
            self.assertTrue(baseline_path.exists())
            text = baseline_path.read_text(encoding="utf-8")
            self.assertIn("BLOCKED_REFERENCE_IMPLEMENTATION_MISSING", text)
            self.assertIn("phi_theta_applicable", text)

    def test_empty_report_marks_results_unavailable_without_mock_values(self):
        with tempfile.TemporaryDirectory() as directory:
            result = write_phi_report(directory)
            report = Path(result["report"]).read_text(encoding="utf-8")
            self.assertIn("NOT_AVAILABLE_INCOMPLETE_FORMAL_RUNS", report)
            self.assertIn("no complete slot traces", report)
            self.assertFalse(result["selected_phi_available"])

    def test_unavailable_baselines_are_explicit_and_missing_traces_are_rejected(self):
        for policy in ("aim_static", "ppo_multicast", "ga_grouping"):
            with self.assertRaises(BaselineUnavailableError):
                canonical_policy_name(policy)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_phi_aoi_trace(Path(directory) / "missing.parquet")


if __name__ == "__main__":
    unittest.main()
