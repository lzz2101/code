from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from mira_leo_study.channel import path_gain, shadowed_rician_samples
from mira_leo_study.config import load_config
from mira_leo_study.isl import build_isl_graph, shortest_route
from mira_leo_study.scenario import StudyScenario


CONFIG = Path(__file__).parents[1] / "configs" / "base.yaml"


class ChannelTimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG)

    def test_shadowed_rician_second_moment(self):
        samples = shadowed_rician_samples(123, 9, (100_000,), 0.126, 0.835, 10.1)
        powers = np.abs(samples) ** 2
        expected = 2.0 * 0.126 + 0.835
        standard_error = float(np.std(powers, ddof=1) / math.sqrt(len(powers)))
        self.assertLess(abs(float(np.mean(powers)) - expected), 5.0 * standard_error)

    def test_channel_is_policy_independent_for_same_index(self):
        left = shadowed_rician_samples(7, 101, (8, 300), 0.126, 0.835, 10.1)
        right = shadowed_rician_samples(7, 101, (8, 300), 0.126, 0.835, 10.1)
        np.testing.assert_array_equal(left, right)
        different = shadowed_rician_samples(7, 102, (8, 300), 0.126, 0.835, 10.1)
        self.assertFalse(np.array_equal(left, different))

    def test_reference_noise_rate_and_time_units(self):
        self.assertAlmostEqual(self.config.get("derived.shadowed_rician_mean_power"), 1.087, places=12)
        self.assertAlmostEqual(self.config.get("derived.noise_power"), 2.416636088e-19, delta=1e-28)
        self.assertIsNone(self.config.get("derived.rate_required_bps_hz"))
        self.assertAlmostEqual(self.config.get("derived.handover_delay_slots"), 2.5)
        self.assertAlmostEqual(self.config.get("derived.fixed_isl_per_hop_slots"), 0.5)
        gain = float(path_gain(np.array([800.0]), 20e9, 299792.458)[0])
        snr = 1.087 * gain / self.config.get("derived.noise_power")
        self.assertAlmostEqual(10.0 * math.log10(snr), 10.0, places=10)

    def test_same_plane_twenty_degree_isl_geometry(self):
        scenario = StudyScenario(self.config)
        _, ecef = scenario.constellation.state(0.0)
        graph = build_isl_graph(ecef, 6371.0, 4000.0)
        route = shortest_route(graph, 0, 1, 299792.458, 0.005)
        expected_distance = 2.0 * 7171.0 * math.sin(math.radians(10.0))
        self.assertTrue(route.reachable)
        self.assertAlmostEqual(route.distance_km, expected_distance, places=9)
        self.assertAlmostEqual(route.geometric_delay_s * 1000.0, 8.3072878, places=6)
        self.assertAlmostEqual(route.fixed_delay_s, 0.005)


if __name__ == "__main__":
    unittest.main()
