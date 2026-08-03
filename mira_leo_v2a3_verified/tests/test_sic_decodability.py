from __future__ import annotations

import unittest

from mira_leo_v2a3_verified.algorithms.power_allocation import (
    PowerAllocationError,
    mira_leo_power_allocation,
)
from mira_leo_v2a3_verified.algorithms.regrouping import Group
from mira_leo_v2a3_verified.noma.transmission import (
    evaluate_noma_decoding,
    transmit_noma,
)
from mira_leo_v2a3_verified.validation.validators import (
    ValidationError,
    validate_sic_power_order,
    validate_sic_success,
)


def group(group_id: int, gain: float, power: float, *, edge: bool = False) -> Group:
    return Group(
        0,
        0,
        group_id,
        [{"id": group_id, "message_id": 0, "aoi": 1.0, "handover": 0}],
        group_gain=gain,
        is_edge=edge,
        scheduled=True,
        power=power,
    )


class SICDecodabilityTests(unittest.TestCase):
    def test_sic_01_legal_power_order(self) -> None:
        groups = [group(0, 0.1, 0.5), group(1, 0.3, 0.3), group(2, 0.8, 0.2)]
        self.assertEqual(validate_sic_power_order(groups, strict=False), [])

    def test_sic_02_wrong_power_order_is_rejected(self) -> None:
        groups = [group(0, 0.1, 0.3), group(1, 0.3, 0.5), group(2, 0.8, 0.2)]
        with self.assertRaisesRegex(ValidationError, "sic_power_order"):
            validate_sic_power_order(groups, strict=True)
        with self.assertRaisesRegex(ValueError, "power-order"):
            transmit_noma(groups, noise_power=1e-3, rate_threshold=0.1)

    def test_sic_03_interleaved_edge_nonedge_is_infeasible(self) -> None:
        groups = [
            group(0, 0.10, 0.0, edge=True),
            group(1, 0.80, 0.0, edge=True),
            group(2, 0.20, 0.0, edge=False),
            group(3, 0.60, 0.0, edge=False),
        ]
        with self.assertRaisesRegex(PowerAllocationError, "global SIC"):
            mira_leo_power_allocation(
                groups, P_sat=1.0, V_c=0.7, V_nc=0.3, V_phi=5e-4
            )

    def test_sic_04_equal_gain_tie_break_is_reproducible(self) -> None:
        groups = [group(2, 0.3, 0.5), group(1, 0.3, 0.5)]
        first = evaluate_noma_decoding(groups, 1e-3, 0.1)["ordered_group_ids"]
        second = evaluate_noma_decoding(list(reversed(groups)), 1e-3, 0.1)[
            "ordered_group_ids"
        ]
        self.assertEqual(first, second)
        self.assertEqual(first, [(0, 1), (0, 2)])

    def test_dec_01_all_required_pairs_are_decodable(self) -> None:
        groups = [group(0, 0.1, 0.7), group(1, 0.5, 0.3)]
        summary = transmit_noma(groups, noise_power=1e-3, rate_threshold=0.1)
        self.assertEqual(summary["pair_failure_count"], 0)
        self.assertTrue(all(item.success for item in groups))

    def test_dec_02_pair_failure_is_recorded_even_with_legal_power_order(self) -> None:
        groups = [group(0, 0.1, 0.5), group(1, 0.5, 0.5)]
        summary = transmit_noma(groups, noise_power=1.0, rate_threshold=1.0)
        strong_decoder_weak_message = [
            record
            for record in summary["pair_records"]
            if record["decoder_group_id"] == 1 and record["target_group_id"] == 0
        ][0]
        self.assertFalse(strong_decoder_weak_message["decoded"])

    def test_dec_03_sic_failure_propagates_along_decoder_chain(self) -> None:
        groups = [
            group(0, 0.1, 0.5),
            group(1, 0.2, 0.3),
            group(2, 0.3, 0.2),
        ]
        transmit_noma(groups, noise_power=1.0, rate_threshold=0.5)
        self.assertFalse(groups[0].success)
        self.assertFalse(groups[1].success)
        self.assertFalse(groups[2].success)

    def test_dec_04_tampered_success_is_detected_by_recomputation(self) -> None:
        groups = [group(0, 0.1, 0.5), group(1, 0.5, 0.5)]
        transmit_noma(groups, noise_power=1.0, rate_threshold=1.0)
        groups[1].success = True
        with self.assertRaisesRegex(ValidationError, "sic_decoding"):
            validate_sic_success(
                groups, noise_power=1.0, rate_threshold=1.0, strict=True
            )


if __name__ == "__main__":
    unittest.main()
