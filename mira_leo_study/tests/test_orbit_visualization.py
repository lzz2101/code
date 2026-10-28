from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from mira_leo_study.config import load_config
from mira_leo_study.orbit_visualization import sample_eci_trajectory


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


class OrbitVisualizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(CONFIG)

    def test_samples_all_satellites_on_configured_circular_orbit(self) -> None:
        times, positions = sample_eci_trajectory(self.config, duration_s=480.0, frames=13)
        self.assertEqual(times.shape, (13,))
        self.assertEqual(positions.shape, (13, 8, 3))
        radii = np.linalg.norm(positions, axis=2)
        np.testing.assert_allclose(
            radii,
            float(self.config.get("derived.semi_major_axis_km")),
            rtol=0.0,
            atol=1e-9,
        )

    def test_invalid_sampling_arguments_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sample_eci_trajectory(self.config, duration_s=0.0, frames=10)
        with self.assertRaises(ValueError):
            sample_eci_trajectory(self.config, duration_s=10.0, frames=1)


if __name__ == "__main__":
    unittest.main()
