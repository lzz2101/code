from __future__ import annotations

import argparse
import copy
import unittest
from dataclasses import replace

import numpy as np

from mira_leo_v1.channel import compute_channel_gain
from mira_leo_v1.geometry import (
    assign_serving_satellite,
    compute_distance_and_elevation,
    initialize_users,
    update_satellite_positions,
)
from mira_leo_v2.algorithms.regrouping import (
    classify_edge_groups as classify_edge_groups_v2,
)
from mira_leo_v2.algorithms.regrouping import (
    make_equal_cardinality_groups as make_groups_v2,
)
from mira_leo_v2.algorithms.scheduler import (
    handover_aware_scheduler as scheduler_v2,
)
from mira_leo_v2.main import attach_slot_observations as attach_v2
from mira_leo_v2.config import V2Config as V2A2Config
from mira_leo_v2a3_verified.algorithms.power_allocation import (
    PowerAllocationError,
    mira_leo_power_allocation,
)
from mira_leo_v2a3_verified.algorithms.regrouping import (
    Group,
    classify_edge_groups,
    make_equal_cardinality_groups,
)
from mira_leo_v2a3_verified.algorithms.scheduler import handover_aware_scheduler
from mira_leo_v2a3_verified.config import V2Config
from mira_leo_v2a3_verified.main import (
    _apply_cli_overrides,
    attach_slot_observations,
    run_mira_leo_v2a3,
)
from mira_leo_v2a3_verified.noma.transmission import transmit_noma


def make_group(
    group_id: int,
    *,
    satellite_id: int = 0,
    message_id: int = 0,
    gain: float = 1.0,
    is_edge: bool = True,
    scheduled: bool = True,
) -> Group:
    user = {"id": group_id, "message_id": message_id, "aoi": 1.0, "handover": 0}
    return Group(
        satellite_id=satellite_id,
        message_id=message_id,
        group_id=group_id,
        users=[user],
        group_gain=gain,
        group_elevation=20.0 if is_edge else 50.0,
        is_edge=is_edge,
        scheduled=scheduled,
    )


def powers(groups: list[Group]) -> list[float]:
    return [group.power for group in sorted(groups, key=lambda item: item.group_gain)]


class Algorithm3PowerAllocationTests(unittest.TestCase):
    def test_no_scheduled_group_returns_zero_summary(self) -> None:
        summary = mira_leo_power_allocation(
            [],
            P_sat=1.0,
            V_c=0.7,
            V_nc=0.3,
            V_phi=5e-4,
            slot=0,
            satellite_id=0,
        )

        self.assertEqual(summary["total_allocated_power"], 0.0)
        self.assertEqual(summary["budget_reason"], "empty")

    def test_edge_only_gets_full_power(self) -> None:
        groups = [
            make_group(0, gain=1.0, is_edge=True),
            make_group(1, gain=2.0, is_edge=True),
        ]

        summary = mira_leo_power_allocation(
            groups,
            P_sat=1.0,
            V_c=0.7,
            V_nc=0.3,
            V_phi=0.05,
        )

        self.assertEqual(summary["budget_reason"], "edge_only")
        self.assertAlmostEqual(summary["edge_power"], 1.0)
        self.assertAlmostEqual(summary["total_allocated_power"], 1.0)
        self.assertGreaterEqual(powers(groups)[0], powers(groups)[1])

    def test_nonedge_only_gets_full_power(self) -> None:
        groups = [
            make_group(0, gain=1.0, is_edge=False),
            make_group(1, gain=2.0, is_edge=False),
        ]

        summary = mira_leo_power_allocation(
            groups,
            P_sat=1.0,
            V_c=0.7,
            V_nc=0.3,
            V_phi=0.05,
        )

        self.assertEqual(summary["budget_reason"], "nonedge_only")
        self.assertAlmostEqual(summary["nonedge_power"], 1.0)
        self.assertAlmostEqual(summary["total_allocated_power"], 1.0)
        self.assertGreaterEqual(powers(groups)[0], powers(groups)[1])

    def test_both_subsets_use_reserved_fractions(self) -> None:
        groups = [
            make_group(0, gain=0.1, is_edge=True),
            make_group(1, gain=0.2, is_edge=True),
            make_group(2, gain=0.5, is_edge=False),
            make_group(3, gain=0.8, is_edge=False),
        ]

        summary = mira_leo_power_allocation(
            groups, P_sat=1.0, V_c=0.7, V_nc=0.3, V_phi=0.05
        )

        self.assertEqual(summary["budget_reason"], "both_subsets")
        self.assertAlmostEqual(summary["edge_power"], 0.7)
        self.assertAlmostEqual(summary["nonedge_power"], 0.3)
        self.assertAlmostEqual(summary["total_allocated_power"], 1.0)
        self.assertGreaterEqual(powers(groups[:2])[0], powers(groups[:2])[1])
        self.assertGreaterEqual(powers(groups[2:])[0], powers(groups[2:])[1])

    def test_paper_formula_numeric_example(self) -> None:
        groups = [
            make_group(0, gain=1.0, is_edge=True),
            make_group(1, gain=2.0, is_edge=True),
            make_group(2, gain=1.0, is_edge=False),
            make_group(3, gain=2.0, is_edge=False),
        ]

        mira_leo_power_allocation(
            groups,
            P_sat=1.0,
            V_c=0.6,
            V_nc=0.4,
            V_phi=0.05,
            enforce_global_sic_order=False,
        )
        edge_powers = powers([group for group in groups if group.is_edge])
        nonedge_powers = powers([group for group in groups if not group.is_edge])

        np.testing.assert_allclose(edge_powers, [0.325, 0.275])
        np.testing.assert_allclose(nonedge_powers, [0.225, 0.175])
        self.assertAlmostEqual(sum(group.power for group in groups), 1.0)

    def test_illegal_power_fractions_are_rejected(self) -> None:
        invalid_cases = [
            (0.6, 0.6),
            (0.0, 1.0),
            (-0.1, 1.1),
            (1.0, 0.0),
        ]
        groups = [make_group(0)]

        for V_c, V_nc in invalid_cases:
            with self.subTest(V_c=V_c, V_nc=V_nc):
                with self.assertRaises(PowerAllocationError):
                    mira_leo_power_allocation(
                        groups, P_sat=1.0, V_c=V_c, V_nc=V_nc, V_phi=0.05
                    )

    def test_nonfinite_power_settings_are_rejected(self) -> None:
        for field_name in ["P_sat", "V_phi", "V_c", "V_nc"]:
            with self.subTest(field_name=field_name):
                with self.assertRaises(ValueError):
                    V2Config(**{field_name: float("nan")})

    def test_cli_power_fractions_are_applied_together(self) -> None:
        args = argparse.Namespace(
            T=None,
            seed=None,
            P_sat=None,
            V_c=0.6,
            V_nc=0.4,
            V_phi=0.05,
        )

        cfg = _apply_cli_overrides(V2Config(), args)

        self.assertEqual(cfg.V_c, 0.6)
        self.assertEqual(cfg.V_nc, 0.4)
        self.assertEqual(cfg.V_phi, 0.05)

    def test_subset_floor_infeasible_is_rejected(self) -> None:
        groups = [
            make_group(0, gain=1.0, is_edge=True),
            make_group(1, gain=2.0, is_edge=True),
        ]

        with self.assertRaisesRegex(PowerAllocationError, "subset=edge"):
            mira_leo_power_allocation(
                groups,
                P_sat=1.0,
                V_c=0.5,
                V_nc=0.5,
                V_phi=0.8,
                slot=3,
                satellite_id=0,
            )

    def test_unscheduled_groups_remain_zero(self) -> None:
        groups = [
            make_group(0, gain=1.0, is_edge=True, scheduled=True),
            make_group(1, gain=2.0, is_edge=True, scheduled=False),
        ]
        groups[1].power = 99.0

        mira_leo_power_allocation(
            groups, P_sat=1.0, V_c=0.7, V_nc=0.3, V_phi=0.05
        )

        self.assertGreater(groups[0].power, 0.0)
        self.assertEqual(groups[1].power, 0.0)

    def test_multi_satellite_isolation_and_rejection(self) -> None:
        sat0_groups = [make_group(0, satellite_id=0, gain=1.0)]
        sat1_groups = [make_group(0, satellite_id=1, gain=1.0)]

        s0 = mira_leo_power_allocation(
            sat0_groups,
            P_sat=1.0,
            V_c=0.7,
            V_nc=0.3,
            V_phi=0.05,
            satellite_id=0,
        )
        s1 = mira_leo_power_allocation(
            sat1_groups,
            P_sat=1.0,
            V_c=0.7,
            V_nc=0.3,
            V_phi=0.05,
            satellite_id=1,
        )

        self.assertAlmostEqual(s0["total_allocated_power"], 1.0)
        self.assertAlmostEqual(s1["total_allocated_power"], 1.0)
        with self.assertRaises(PowerAllocationError):
            mira_leo_power_allocation(
                sat0_groups + sat1_groups,
                P_sat=1.0,
                V_c=0.7,
                V_nc=0.3,
                V_phi=0.05,
            )

    def test_global_sic_diagnostic_detects_cross_subset_violation(self) -> None:
        groups = [
            make_group(0, gain=10.0, is_edge=True),
            make_group(1, gain=1.0, is_edge=False),
        ]

        summary = mira_leo_power_allocation(
            groups,
            P_sat=1.0,
            V_c=0.7,
            V_nc=0.3,
            V_phi=0.05,
            enforce_global_sic_order=False,
        )

        self.assertGreater(summary["sic_power_order_violations"], 0)
        self.assertGreater(summary["max_sic_power_order_violation"], 0.0)

    def test_global_sic_equal_gain_tie_is_stable(self) -> None:
        groups = [
            make_group(0, gain=1.0, is_edge=True),
            make_group(1, gain=1.0, is_edge=False),
        ]

        summary = mira_leo_power_allocation(
            groups, P_sat=1.0, V_c=0.7, V_nc=0.3, V_phi=0.05
        )

        self.assertEqual(summary["sic_power_order_violations"], 0)
        self.assertEqual(
            [group.metadata["global_sic_rank"] for group in groups], [1, 2]
        )

    def test_noma_transmission_uses_algorithm3_non_equal_power(self) -> None:
        groups = [
            make_group(0, gain=1.0, is_edge=True),
            make_group(1, gain=2.0, is_edge=True),
        ]
        equal_groups = copy.deepcopy(groups)
        for group in equal_groups:
            group.power = 0.5

        mira_leo_power_allocation(
            groups, P_sat=1.0, V_c=0.7, V_nc=0.3, V_phi=0.05
        )
        transmit_noma(groups, noise_power=1e-12, rate_threshold=0.1)
        transmit_noma(equal_groups, noise_power=1e-12, rate_threshold=0.1)

        self.assertNotEqual([group.power for group in groups], [0.5, 0.5])
        self.assertNotEqual([group.rate for group in groups], [group.rate for group in equal_groups])

    def test_algorithm1_algorithm2_regression_before_power_allocation(self) -> None:
        cfg_v2 = replace(V2A2Config(), T=1, random_seed=15)
        cfg_v2a3 = replace(V2Config(), T=1, random_seed=15)
        rng_v2 = np.random.default_rng(cfg_v2.random_seed)
        rng_v2a3 = np.random.default_rng(cfg_v2a3.random_seed)
        users_v2 = initialize_users(cfg_v2, rng_v2)
        users_v2a3 = initialize_users(cfg_v2a3, rng_v2a3)

        sat_positions_v2 = update_satellite_positions(0, cfg_v2)
        sat_positions_v2a3 = update_satellite_positions(0, cfg_v2a3)
        distance_v2, elevation_v2, _ = compute_distance_and_elevation(
            users_v2, sat_positions_v2, cfg_v2
        )
        distance_v2a3, elevation_v2a3, _ = compute_distance_and_elevation(
            users_v2a3, sat_positions_v2a3, cfg_v2a3
        )
        assign_serving_satellite(users_v2, elevation_v2, cfg_v2)
        assign_serving_satellite(users_v2a3, elevation_v2a3, cfg_v2a3)
        attach_v2(
            users_v2,
            compute_channel_gain(distance_v2, cfg_v2, rng_v2, fading=False),
            elevation_v2,
            cfg_v2,
        )
        attach_slot_observations(
            users_v2a3,
            compute_channel_gain(distance_v2a3, cfg_v2a3, rng_v2a3, fading=False),
            elevation_v2a3,
            cfg_v2a3,
        )

        for sat_id in range(cfg_v2.L):
            old_groups = make_groups_v2(users_v2, sat_id, 4, cfg_v2.M)
            new_groups = make_equal_cardinality_groups(
                users_v2a3, sat_id, 4, cfg_v2a3.M
            )
            classify_edge_groups_v2(old_groups, cfg_v2.phi_theta)
            classify_edge_groups(new_groups, cfg_v2a3.phi_theta)
            old_scheduled = scheduler_v2(old_groups, cfg_v2, distance_v2)
            new_scheduled = handover_aware_scheduler(
                new_groups, cfg_v2a3, distance_v2a3
            )

            old_ids = sorted((group.message_id, group.group_id) for group in old_scheduled)
            new_ids = sorted((group.message_id, group.group_id) for group in new_scheduled)
            self.assertEqual(old_ids, new_ids)

    def test_main_loop_runs_with_algorithm3_histories(self) -> None:
        cfg = replace(V2Config(), T=20, random_seed=22)

        result = run_mira_leo_v2a3(cfg, policy="fixed", fixed_action=4, fading=False)
        histories = result["histories"]

        for key in [
            "edge_power",
            "nonedge_power",
            "total_allocated_power",
            "power_floor_infeasible_count",
            "sic_power_order_violations",
            "max_sic_power_order_violation",
        ]:
            self.assertEqual(len(histories[key]), cfg.T)
            self.assertTrue(np.all(np.isfinite(histories[key])))
        self.assertEqual(sum(histories["power_floor_infeasible_count"]), 0.0)
        self.assertLessEqual(max(histories["total_allocated_power"]), cfg.L * cfg.P_sat)


if __name__ == "__main__":
    unittest.main()
