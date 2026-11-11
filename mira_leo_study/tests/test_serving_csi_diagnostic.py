"""Serving-only selection/packaging tests with artificial channel arrays only."""
from contextlib import redirect_stdout
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mira_leo_study.association import AssociationTracker
from mira_leo_study.config import load_config
from mira_leo_study.csi_elevation_no_interference import (
    BIN_COUNT, SlotMetric, build_analysis_config, elevation_bin_indices,
)
from mira_leo_study.csi_elevation_serving_satellite import (
    FIGURE_NAMES, main, run_serving_csi_elevation_study, select_serving_links,
    summarize_population,
)
from mira_leo_study.scenario import StudyScenario

CONFIG = Path(__file__).parents[1] / "configs/base.yaml"
MODULE = "mira_leo_study.csi_elevation_serving_satellite"


def artificial_metric(slot, elevation):
    elevation = np.asarray(elevation, dtype=float)
    visible = elevation >= 10.
    db = np.arange(960, dtype=float).reshape(8, 120)/100.-240.+slot*.1
    raw = 10.**(db/10.)
    return SlotMetric(raw, elevation, visible, elevation[visible],
                      elevation_bin_indices(elevation[visible]), db[visible],
                      np.full(int(visible.sum()), 1e-18),
                      (slot*960+np.flatnonzero(visible)).astype(np.uint64), None)


class ServingCsiDiagnosticTests(unittest.TestCase):
    def test_only_current_serving_visible_links_are_selected(self):
        elevation = np.full((8, 120), 12.)
        elevation[3] = 45.
        elevation[3, :5] = [9.99, 10., 14.99, 15., 90.]
        original = artificial_metric(7, elevation)
        selected = select_serving_links(original, 3)
        self.assertEqual(len(selected.gain_db), 119)
        self.assertTrue(np.all((selected.observation_ids % 960)//120 == 3))
        self.assertEqual(selected.observation_ids[0], 7*960+3*120+1)
        self.assertEqual(list(selected.angles_deg[:4]), [10., 14.99, 15., 90.])
        self.assertEqual(list(selected.angle_bins[:4]), [0, 4, 5, 79])
        np.testing.assert_array_equal(selected.raw_gain, original.raw_gain)
        np.testing.assert_array_equal(selected.elevation_deg, original.elevation_deg)
        self.assertEqual(int(original.visible.sum()), 959)
        self.assertEqual(int(selected.visible.sum()), 119)

    def test_hysteresis_ttt_and_handover_select_new_rows_at_same_slot(self):
        config = load_config(CONFIG)
        scenario = StudyScenario(config)
        tracker = AssociationTracker(config, scenario.subscriptions)
        elevation = np.full((8, 120), 20.)
        elevation[0] = 50.
        first = tracker.update(0, elevation)
        self.assertEqual(first.serving_satellite, 0)
        self.assertFalse(first.handover)
        elevation[0], elevation[1] = 40., 60.
        metric = artificial_metric(1, elevation)
        for slot in range(1, 101):
            decision = tracker.update(slot, elevation)
            expected = 0 if slot < 100 else 1
            self.assertEqual(decision.serving_satellite, expected)
            self.assertTrue(np.all((select_serving_links(metric, expected).observation_ids % 960)//120 == expected))
        self.assertTrue(decision.handover)

    def test_invisible_selected_satellite_returns_no_fake_observations(self):
        elevation = np.full((8, 120), 50.)
        elevation[2] = 5.
        selected = select_serving_links(artificial_metric(0, elevation), 2)
        self.assertEqual(len(selected.gain_db), 0)
        self.assertFalse(np.any(selected.visible))

    def test_invalid_satellite_or_interference_adjusted_data_rejected(self):
        original = artificial_metric(0, np.full((8, 120), 50.))
        for value in (-1, 8, True, .5):
            with self.assertRaises(ValueError):
                select_serving_links(original, value)
        adjusted = SlotMetric(original.raw_gain, original.elevation_deg, original.visible,
                              original.angles_deg, original.angle_bins, original.gain_db,
                              original.expected_gain_linear, original.observation_ids,
                              np.ones(len(original.gain_db)))
        with self.assertRaisesRegex(ValueError, "without interference"):
            select_serving_links(adjusted, 0)

    def test_two_global_quantiles_preserve_raw_order_and_empty_bins(self):
        with tempfile.TemporaryDirectory(prefix="ARTIFICIAL_serving_quantile_") as tmp:
            gain, bins = Path(tmp)/"gain.bin", Path(tmp)/"bins.bin"
            values = np.array([-200., -180., -220., -190., -210., -170., -230., -160., -240., -150.])
            indexes = np.array([0, 5]*5, dtype=np.uint8)
            values.astype("<f8").tofile(gain); indexes.tofile(bins)
            before = gain.read_bytes(), bins.read_bytes()
            totals = np.bincount(indexes, minlength=BIN_COUNT)
            result = summarize_population(gain, bins, len(values), totals, totals*1e-18)
            for percent in (20, 30):
                self.assertAlmostEqual(result[percent]["threshold_db"], np.quantile(values, percent/100.))
                self.assertEqual(result[percent]["count"], percent//10)
                self.assertTrue(np.isnan(result[percent]["fractions"][1]))
            self.assertTrue(np.all(result[20]["bottom_counts"] <= result[30]["bottom_counts"]))
            self.assertEqual(before, (gain.read_bytes(), bins.read_bytes()))
            self.assertFalse((Path(tmp)/"gain_for_quantile.bin").exists())

    def test_quantile_ties_report_realized_fraction_without_discarding_data(self):
        with tempfile.TemporaryDirectory(prefix="ARTIFICIAL_serving_ties_") as tmp:
            gain, bins = Path(tmp)/"gain.bin", Path(tmp)/"bins.bin"
            np.full(10, -200., dtype="<f8").tofile(gain)
            np.zeros(10, dtype=np.uint8).tofile(bins)
            totals = np.zeros(BIN_COUNT, dtype=np.int64); totals[0] = 10
            result = summarize_population(gain, bins, 10, totals, totals*1e-18)
            for percent in (20, 30):
                self.assertEqual(result[percent]["fraction"], 1.)
                self.assertEqual(result[percent]["threshold_tie_count"], 10)
                self.assertFalse(result[percent]["fraction_matches_nominal"])

    def test_artificial_pipeline_outputs_three_figures_and_aligned_trace(self):
        first = np.full((8, 120), 20.); first[0] = np.linspace(40., 60., 120)
        second = first.copy(); second[0] = 5.; second[1] = np.linspace(50., 70., 120)
        metrics = [artificial_metric(0, first), artificial_metric(1, second)]
        population = np.concatenate([select_serving_links(metrics[0], 0).gain_db,
                                     select_serving_links(metrics[1], 1).gain_db])
        with tempfile.TemporaryDirectory(prefix="ARTIFICIAL_serving_outputs_") as tmp:
            with patch(MODULE+".compute_slot_metric", side_effect=metrics) as compute, \
                    patch("mira_leo_study.csi_elevation_no_interference.serving_channel_amplitude",
                          side_effect=AssertionError("real simulation forbidden")), redirect_stdout(io.StringIO()):
                summary = run_serving_csi_elevation_study(outputs=tmp, smoke_slots=2, plot_sample_max=50)
            self.assertEqual(compute.call_count, 2)
            self.assertTrue(all(call.kwargs["with_interference"] is False for call in compute.call_args_list))
            self.assertEqual(summary["number_of_valid_link_observations"], 240)
            self.assertEqual(summary["handover_count"], 1)
            self.assertEqual(summary["number_of_plotted_scatter_points"], 50)
            self.assertTrue(summary["is_debug"])
            output = Path(summary["output_directory"])
            self.assertEqual({path.name for path in output.glob("*.png")}, set(FIGURE_NAMES))
            self.assertFalse(list(output.glob("_serving_raw_*")))
            for percent in (20, 30):
                self.assertAlmostEqual(summary[f"global_bottom{percent}_threshold_db"],
                                       np.quantile(population, percent/100.))
                self.assertAlmostEqual(summary[f"global_bottom{percent}_fraction"], percent/100.)
            with (output/"association_trace.csv").open(newline="", encoding="utf-8") as handle:
                trace = list(csv.DictReader(handle))
            self.assertEqual([r["serving_satellite_id"] for r in trace], ["sat_01", "sat_02"])
            self.assertEqual([r["handover"] for r in trace], ["0", "1"])
            with (output/"sampled_scatter_points.csv").open(newline="", encoding="utf-8") as handle:
                sample = list(csv.DictReader(handle))
            self.assertEqual(len(sample), 50)
            self.assertEqual(len({r["observation_id"] for r in sample}), 50)
            self.assertTrue(all(r["serving_satellite_id"] == trace[int(r["slot"])]["serving_satellite_id"] for r in sample))
            with (output/"elevation_bin_statistics.csv").open(newline="", encoding="utf-8") as handle:
                statistics = list(csv.DictReader(handle))
            self.assertEqual(len(statistics), 80)
            self.assertEqual(sum(int(r["total_count"]) for r in statistics), 240)
            self.assertTrue(all(r["bottom20_fraction_pct"] == "" for r in statistics if not int(r["total_count"])))

    def test_all_invisible_window_fails_without_fake_quantiles_or_figures(self):
        metric = artificial_metric(0, np.full((8, 120), 5.))
        with tempfile.TemporaryDirectory(prefix="ARTIFICIAL_serving_empty_") as tmp:
            with patch(MODULE+".compute_slot_metric", return_value=metric), \
                    redirect_stdout(io.StringIO()), patch("sys.stderr", new=io.StringIO()):
                with self.assertRaisesRegex(ValueError, "no visible serving links"):
                    run_serving_csi_elevation_study(outputs=tmp, smoke_slots=1)
            self.assertFalse(list(Path(tmp).rglob("*.png")))
            self.assertFalse(list(Path(tmp).rglob("run_summary.json")))

    def test_config_cli_and_invalid_options_do_not_start_simulation(self):
        base, diagnostic = build_analysis_config()
        self.assertEqual(base.slot_count, 48000)
        self.assertEqual(diagnostic.slot_count, 96000)
        with patch(MODULE+".compute_slot_metric", side_effect=AssertionError("simulation forbidden")):
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exited:
                main(["--help"])
            self.assertEqual(exited.exception.code, 0)
            for value in (0, True, 100001, .5):
                with self.assertRaises(ValueError):
                    run_serving_csi_elevation_study(plot_sample_max=value)
            for value in (0, True, 96000, .5):
                with self.assertRaises(ValueError):
                    run_serving_csi_elevation_study(smoke_slots=value)


if __name__ == "__main__":
    unittest.main()
