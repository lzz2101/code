from __future__ import annotations

import numpy as np

from mira_leo_v2a3_verified.algorithms.regrouping import Group


def _group_order_key(group: Group) -> tuple[float, int, int]:
    return (float(group.group_gain), int(group.message_id), int(group.group_id))


def validate_noma_power_order(
    scheduled_groups: list[Group], tolerance: float = 1e-9
) -> None:
    ordered = sorted(scheduled_groups, key=_group_order_key)
    for weaker, stronger in zip(ordered, ordered[1:]):
        if stronger.power > weaker.power + tolerance:
            raise ValueError(
                "NOMA SIC power-order violation: "
                f"satellite={weaker.satellite_id}, "
                f"weaker=(message={weaker.message_id}, group={weaker.group_id}, "
                f"gain={weaker.group_gain}, power={weaker.power}), "
                f"stronger=(message={stronger.message_id}, group={stronger.group_id}, "
                f"gain={stronger.group_gain}, power={stronger.power})"
            )


def evaluate_noma_decoding(
    scheduled_groups: list[Group], noise_power: float, rate_threshold: float
) -> dict[str, object]:
    """Evaluate every decoder-message pair required by the global SIC chain."""
    if not np.isfinite(noise_power) or noise_power < 0.0:
        raise ValueError("noise_power must be finite and non-negative")
    if not np.isfinite(rate_threshold) or rate_threshold < 0.0:
        raise ValueError("rate_threshold must be finite and non-negative")

    ordered = sorted(scheduled_groups, key=_group_order_key)
    pair_records: list[dict[str, object]] = []
    group_results: dict[tuple[int, int], dict[str, object]] = {}

    for decoder_index, decoder in enumerate(ordered):
        chain_success = True
        own_sinr = 0.0
        own_rate = 0.0
        decoder_records = []

        for target_index in range(decoder_index + 1):
            target = ordered[target_index]
            remaining_power = sum(
                item.power for item in ordered[target_index + 1 :]
            )
            denominator = decoder.group_gain * remaining_power + noise_power
            numerator = decoder.group_gain * target.power
            sinr = numerator / denominator if denominator > 0.0 else np.inf
            rate = float(np.log2(1.0 + sinr))
            decoded = bool(np.isfinite(rate) and rate >= rate_threshold)
            chain_success = chain_success and decoded

            record = {
                "satellite_id": int(decoder.satellite_id),
                "decoder_message_id": int(decoder.message_id),
                "decoder_group_id": int(decoder.group_id),
                "target_message_id": int(target.message_id),
                "target_group_id": int(target.group_id),
                "decoder_gain": float(decoder.group_gain),
                "target_power": float(target.power),
                "remaining_power": float(remaining_power),
                "noise_power": float(noise_power),
                "sinr": float(sinr),
                "rate": rate,
                "rate_threshold": float(rate_threshold),
                "decoded": decoded,
            }
            pair_records.append(record)
            decoder_records.append(record)

            if target_index == decoder_index:
                own_sinr = float(sinr)
                own_rate = rate

        group_results[(int(decoder.message_id), int(decoder.group_id))] = {
            "sinr": own_sinr,
            "rate": own_rate,
            "success": bool(chain_success),
            "pair_records": decoder_records,
        }

    return {
        "ordered_group_ids": [
            (int(group.message_id), int(group.group_id)) for group in ordered
        ],
        "pair_records": pair_records,
        "pair_failure_count": sum(not record["decoded"] for record in pair_records),
        "group_results": group_results,
    }


def transmit_noma(
    scheduled_groups: list[Group], noise_power: float, rate_threshold: float
) -> dict[str, object]:
    """Compute NOMA SINR, rate, and SIC success for one satellite."""
    if not scheduled_groups:
        return {
            "ordered_group_ids": [],
            "pair_records": [],
            "pair_failure_count": 0,
            "group_results": {},
            "success_count": 0,
        }

    for group in scheduled_groups:
        if not np.isfinite(group.power) or group.power <= 0.0:
            raise ValueError(
                f"scheduled group power must be finite and positive before NOMA transmission: "
                f"satellite={group.satellite_id}, message={group.message_id}, "
                f"group={group.group_id}, power={group.power}"
            )

    validate_noma_power_order(scheduled_groups)
    evaluation = evaluate_noma_decoding(
        scheduled_groups, noise_power=noise_power, rate_threshold=rate_threshold
    )
    group_results = evaluation["group_results"]

    for group in scheduled_groups:
        result = group_results[(int(group.message_id), int(group.group_id))]
        group.sinr = float(result["sinr"])
        group.rate = float(result["rate"])
        group.success = bool(result["success"])
        group.metadata["sic_pair_records"] = result["pair_records"]

    evaluation["success_count"] = sum(group.success for group in scheduled_groups)
    return evaluation
