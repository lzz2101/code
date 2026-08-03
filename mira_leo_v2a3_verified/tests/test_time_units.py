from __future__ import annotations

import inspect
import unittest
from copy import deepcopy

import numpy as np

from mira_leo_v2a3_verified.algorithms.regrouping import Group
from mira_leo_v2a3_verified.aoi.metrics import update_user_aoi
from mira_leo_v2a3_verified.config import V2Config
from mira_leo_v2a3_verified.time_units import (
    blackout_slots,
    milliseconds_to_slots,
    seconds_to_slots,
)


def user(*, handover: int = 0, success: bool = False) -> dict:
    return {
        "id": 0,
        "message_id": 0,
        "aoi": 1.0,
        "serving_sat": 0,
        "handover": handover,
        "success": success,
    }


class TimeUnitTests(unittest.TestCase):
    def test_tm_01_basic_conversions(self) -> None:
        expected = [(0, 0.0, 0), (5, 0.5, 1), (10, 1.0, 1), (25, 2.5, 3), (30, 3.0, 3)]
        for delay_ms, continuous, blackout in expected:
            with self.subTest(delay_ms=delay_ms):
                self.assertAlmostEqual(milliseconds_to_slots(delay_ms, 0.01), continuous)
                self.assertEqual(blackout_slots(delay_ms / 1000.0, 0.01), blackout)

    def test_tm_02_seconds_and_milliseconds_agree(self) -> None:
        self.assertAlmostEqual(
            milliseconds_to_slots(25.0, 0.01), seconds_to_slots(0.025, 0.01)
        )

    def test_tm_03_no_handover_has_no_handover_penalty(self) -> None:
        cfg = V2Config()
        users = [user(handover=0)]
        update_user_aoi(users, [], np.zeros((1, 1)), cfg, slot=0)
        self.assertAlmostEqual(users[0]["aoi"], 2.0)

    def test_tm_04_single_handover_adds_2_5_slots_once(self) -> None:
        cfg = V2Config()
        no_event = [user(handover=0)]
        event = [user(handover=1)]
        event[0]["handover_event_id"] = "event-1"
        update_user_aoi(no_event, [], np.zeros((1, 1)), cfg, slot=0)
        update_user_aoi(event, [], np.zeros((1, 1)), cfg, slot=0)
        self.assertAlmostEqual(event[0]["aoi"] - no_event[0]["aoi"], 2.5)

    def test_tm_05_same_handover_event_is_not_charged_twice(self) -> None:
        cfg = V2Config()
        users = [user(handover=1)]
        users[0]["handover_event_id"] = "same-event"
        distance = np.zeros((1, 1))
        update_user_aoi(users, [], distance, cfg, slot=0)
        first = users[0]["aoi"]
        update_user_aoi(users, [], distance, cfg, slot=1)
        self.assertAlmostEqual(first, 4.5)
        self.assertAlmostEqual(users[0]["aoi"], 5.5)

    def test_tm_06_isl_delay_is_0_5_slots_per_hop(self) -> None:
        self.assertAlmostEqual(V2Config().isl_delay_slots, 0.5)

    def test_tm_07_propagation_and_other_delays_share_slot_units(self) -> None:
        cfg = V2Config()
        users = [user(success=False)]
        users[0]["isl_hops"] = 2
        group = Group(0, 0, 0, users, scheduled=True, success=True)
        distance = np.array([[cfg.c * cfg.slot_time_s * 0.25]])
        update_user_aoi(users, [group], distance, cfg, slot=0)
        self.assertAlmostEqual(users[0]["aoi"], 1.0 + 0.25 + 1.0)

    def test_tm_08_aoi_update_does_not_add_raw_kh_seconds(self) -> None:
        source = inspect.getsource(update_user_aoi)
        self.assertNotIn("cfg.K_h", source)
        self.assertIn("cfg.handover_delay_slots", source)


if __name__ == "__main__":
    unittest.main()
