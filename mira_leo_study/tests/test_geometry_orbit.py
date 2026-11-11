from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from mira_leo_study.config import load_config
from mira_leo_study.geometry import (
    aeqd_forward,
    aeqd_inverse,
    ecef_to_spherical,
    generate_coverage_grid,
    link_geometry,
    radial_project_to_sphere,
)
from mira_leo_study.scenario import StudyScenario, generate_subscriptions


CONFIG = Path(__file__).parents[1] / "configs" / "base.yaml"


class GeometryOrbitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG)
        cls.scenario = StudyScenario(cls.config)

    def test_constellation_is_exactly_five_plus_three(self):
        elements = self.scenario.constellation.elements
        self.assertEqual(len(elements), 8)
        self.assertEqual([item.plane for item in elements].count(1), 5)
        self.assertEqual([item.plane for item in elements].count(2), 3)
        self.assertEqual([item.satellite_id for item in elements], [f"sat_{i:02d}" for i in range(1, 9)])

    def test_orbit_radius_and_zenith_reference(self):
        for time_s in (0.0, 120.0, 240.0, 480.0):
            eci, ecef = self.scenario.constellation.state(time_s)
            np.testing.assert_allclose(np.linalg.norm(eci, axis=1), 7171.0, atol=1e-9)
            np.testing.assert_allclose(np.linalg.norm(ecef, axis=1), 7171.0, atol=1e-9)
        center = radial_project_to_sphere(
            self.config.get("geometry.center_raw_ecef_km"), 6371.0
        )
        _, ecef = self.scenario.constellation.state(240.0)
        distance, elevation = link_geometry(ecef, center.reshape(1, 3))
        self.assertAlmostEqual(elevation[2, 0], 90.0, places=9)
        self.assertAlmostEqual(distance[2, 0], 800.0, places=9)

    def test_spherical_aeqd_round_trip_origin_and_corners(self):
        center = radial_project_to_sphere(
            self.config.get("geometry.center_raw_ecef_km"), 6371.0
        )
        lat0, lon0 = ecef_to_spherical(center)
        east = np.array([0.0, -500.0, -500.0, 500.0, 500.0])
        north = np.array([0.0, -500.0, 500.0, -500.0, 500.0])
        lat, lon = aeqd_inverse(east, north, float(lat0), float(lon0), 6371.0)
        east2, north2 = aeqd_forward(lat, lon, float(lat0), float(lon0), 6371.0)
        np.testing.assert_allclose(east2, east, atol=1e-9)
        np.testing.assert_allclose(north2, north, atol=1e-9)
        self.assertAlmostEqual(float(lat[0]), float(lat0), places=14)
        self.assertAlmostEqual(float(lon[0]), float(lon0), places=14)

    def test_devices_and_subscriptions_are_fixed_and_nonoverlapping(self):
        second = StudyScenario(self.config)
        self.assertEqual(self.scenario.devices.layout_hash, second.devices.layout_hash)
        self.assertEqual(self.scenario.subscriptions.subscription_hash, second.subscriptions.subscription_hash)
        np.testing.assert_array_equal(self.scenario.devices.ecef_km, second.devices.ecef_km)
        subscriptions = generate_subscriptions(self.config)
        self.assertEqual(len(np.unique(subscriptions.active_device_ids)), 120)
        self.assertEqual(set(subscriptions.message_members), {0, 1, 2})
        members = np.concatenate(list(subscriptions.message_members.values()))
        self.assertEqual(len(np.unique(members)), 120)
        np.testing.assert_array_equal(np.sort(members), np.arange(120))
        self.assertTrue(all(len(m) == 40 for m in subscriptions.message_members.values()))

    def test_grid_coverage_matches_discrete_reference(self):
        _, _, grid = generate_coverage_grid(self.config)
        visible_min = 99
        visible_max = -1
        center_counts = []
        second_best_minimums = []
        center = radial_project_to_sphere(
            self.config.get("geometry.center_raw_ecef_km"), 6371.0
        ).reshape(1, 3)
        for time_s in range(481):
            _, satellites = self.scenario.constellation.state(float(time_s))
            _, elevation = link_geometry(satellites, grid)
            counts = np.count_nonzero(elevation >= 10.0, axis=0)
            visible_min = min(visible_min, int(np.min(counts)))
            visible_max = max(visible_max, int(np.max(counts)))
            _, center_elevation = link_geometry(satellites, center)
            center_counts.append(int(np.count_nonzero(center_elevation >= 10.0)))
            second_best_minimums.append(float(np.sort(np.min(elevation, axis=1))[::-1][1]))
        self.assertEqual(len(grid), 2601)
        self.assertEqual((visible_min, visible_max), (2, 4))
        self.assertEqual((min(center_counts), max(center_counts)), (3, 4))
        self.assertAlmostEqual(min(second_best_minimums), 11.856573001947064, places=8)


if __name__ == "__main__":
    unittest.main()
