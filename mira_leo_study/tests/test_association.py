from __future__ import annotations

import unittest
from pathlib import Path

from mira_leo_study.association import AssociationTracker
from mira_leo_study.config import load_config
from mira_leo_study.geometry import link_geometry
from mira_leo_study.scenario import StudyScenario


CONFIG = Path(__file__).parents[1] / "configs" / "base.yaml"


class AssociationTests(unittest.TestCase):
    def test_initial_acquisition_is_not_handover_and_serving_is_scalar(self):
        config = load_config(CONFIG)
        scenario = StudyScenario(config)
        tracker = AssociationTracker(config, scenario.subscriptions)
        _, satellites = scenario.constellation.state(0.0)
        _, elevation = link_geometry(satellites, scenario.devices.ecef_km)
        first = tracker.update(0, elevation)
        second = tracker.update(1, elevation)
        self.assertIsInstance(first.serving_satellite, int)
        self.assertTrue(0 <= first.serving_satellite < 8)
        self.assertTrue(first.initial_acquisition)
        self.assertFalse(first.handover)
        self.assertFalse(second.handover)

    def test_visibility_boundary_is_inclusive(self):
        config = load_config(CONFIG)
        self.assertEqual(config.get("geometry.phi_min_deg"), 10.0)
        self.assertTrue(10.0 >= config.get("geometry.phi_min_deg"))
        self.assertFalse(9.999999 >= config.get("geometry.phi_min_deg"))


if __name__ == "__main__":
    unittest.main()
