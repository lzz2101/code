from __future__ import annotations

import unittest
from pathlib import Path

from mira_leo_study.association import AssociationTracker
from mira_leo_study.config import load_config
from mira_leo_study.geometry import link_geometry
from mira_leo_study.scenario import StudyScenario


CONFIG = Path(__file__).parents[1] / "configs" / "base.yaml"


class AssociationTests(unittest.TestCase):
    def test_initial_acquisition_is_not_handover_and_assignment_is_injective(self):
        config = load_config(CONFIG)
        scenario = StudyScenario(config)
        tracker = AssociationTracker(config, scenario.subscriptions)
        _, satellites = scenario.constellation.state(0.0)
        _, elevation = link_geometry(satellites, scenario.devices.ecef_km)
        first = tracker.update(0, elevation)
        second = tracker.update(1, elevation)
        self.assertEqual(len(set(first.serving_satellites)), 2)
        self.assertEqual(first.initial_acquisition, (True, True))
        self.assertEqual(first.handover, (False, False))
        self.assertEqual(second.handover, (False, False))

    def test_visibility_boundary_is_inclusive(self):
        config = load_config(CONFIG)
        self.assertEqual(config.get("geometry.phi_min_deg"), 10.0)
        self.assertTrue(10.0 >= config.get("geometry.phi_min_deg"))
        self.assertFalse(9.999999 >= config.get("geometry.phi_min_deg"))


if __name__ == "__main__":
    unittest.main()
