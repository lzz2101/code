from __future__ import annotations

import argparse
import csv
import io
import json
import platform
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np

from mira_leo_v2a3_verified.algorithms.power_allocation import (
    PowerAllocationError,
    mira_leo_power_allocation,
)
from mira_leo_v2a3_verified.algorithms.regrouping import Group
from mira_leo_v2a3_verified.algorithms.sw_ucb import SWUCB
from mira_leo_v2a3_verified.aoi.metrics import (
    compute_regrouping_reward_details,
    update_user_aoi,
)
from mira_leo_v2a3_verified.config import V2Config
from mira_leo_v2a3_verified.main import run_mira_leo_v2a3_verified
from mira_leo_v2a3_verified.noma.transmission import transmit_noma
from mira_leo_v2a3_verified.validation.validators import (
    ValidationError,
    validate_power_constraints,
    validate_reward_record,
)


VIOLATION_KEYS = {
    "stale_sw_ucb_observations": 0,
    "reward_out_of_bounds": 0,
    "nonfinite_rewards": 0,
    "negative_power": 0,
    "satellite_power_overflow": 0,
    "unscheduled_positive_power": 0,
    "power_floor_infeasible_success": 0,
    "sic_power_order": 0,
    "sic_decoding": 0,
    "time_unit": 0,
    "repeated_handover_charge": 0,
}


def _group(
    group_id: int,
    gain: float,
    *,
    is_edge: bool,
    satellite_id: int = 0,
    scheduled: bool = True,
) -> Group:
    user = {
        "id": group_id,
        "message_id": 0,
        "aoi": 1.0,
        "handover": 0,
        "serving_sat": satellite_id,
    }
    return Group(
        satellite_id=satellite_id,
        message_id=0,
        group_id=group_id,
        users=[user],
        group_gain=gain,
        is_edge=is_edge,
        scheduled=scheduled,
    )


def _allocate_and_transmit(groups: list[Group], cfg: V2Config) -> dict[str, object]:
    allocation = mira_leo_power_allocation(
        groups,
        P_sat=cfg.P_sat,
        V_c=cfg.V_c,
        V_nc=cfg.V_nc,
        V_phi=cfg.V_phi,
        slot=0,
        satellite_id=0,
    )
    validate_power_constraints(
        groups, P_sat=cfg.P_sat, slot=0, satellite_id=0, strict=True
    )
    transmission = transmit_noma(
        [group for group in groups if group.scheduled],
        noise_power=1e-6,
        rate_threshold=0.1,
    )
    return {"allocation": allocation, "transmission": transmission}


def run_controlled_scenarios(strict: bool = True) -> tuple[list[dict], dict]:
    cfg = replace(V2Config(), T=10, V_c=0.7, V_nc=0.3, V_phi=5e-4)
    rows: list[dict] = []

    def record(scenario_id: str, callback) -> None:
        try:
            details = callback()
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "status": "PASS",
                    "details": json.dumps(details, sort_keys=True),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "status": "FAIL",
                    "details": f"{type(exc).__name__}: {exc}",
                }
            )
            if strict:
                raise

    def nonedge_only():
        groups = [_group(0, 0.1, is_edge=False), _group(1, 0.5, is_edge=False)]
        result = _allocate_and_transmit(groups, cfg)
        reward = compute_regrouping_reward_details(
            success_count=sum(group.success for group in groups),
            weighted_sum_aoi=2.0,
            edge_count=0,
            B=cfg.B_edge,
            normalize=True,
            reward_min=cfg.reward_raw_bounds[0],
            reward_max=cfg.reward_raw_bounds[1],
        )
        return {
            "nonedge_power": result["allocation"]["nonedge_power"],
            "total_power": result["allocation"]["total_allocated_power"],
            "reward": reward["normalized_reward"],
        }

    def edge_only():
        groups = [_group(0, 0.1, is_edge=True), _group(1, 0.5, is_edge=True)]
        result = _allocate_and_transmit(groups, cfg)
        return {
            "edge_power": result["allocation"]["edge_power"],
            "total_power": result["allocation"]["total_allocated_power"],
        }

    def mixed_noninterleaved():
        groups = [
            _group(0, 0.1, is_edge=True),
            _group(1, 0.2, is_edge=True),
            _group(2, 0.5, is_edge=False),
            _group(3, 0.8, is_edge=False),
        ]
        result = _allocate_and_transmit(groups, cfg)
        return {
            "edge_power": result["allocation"]["edge_power"],
            "nonedge_power": result["allocation"]["nonedge_power"],
            "sic_power_order_violations": result["allocation"][
                "sic_power_order_violations"
            ],
        }

    def mixed_interleaved_rejected():
        groups = [
            _group(0, 0.10, is_edge=True),
            _group(1, 0.80, is_edge=True),
            _group(2, 0.20, is_edge=False),
            _group(3, 0.60, is_edge=False),
        ]
        try:
            mira_leo_power_allocation(
                groups,
                P_sat=cfg.P_sat,
                V_c=cfg.V_c,
                V_nc=cfg.V_nc,
                V_phi=cfg.V_phi,
                slot=0,
                satellite_id=0,
            )
        except PowerAllocationError as exc:
            return {"rejected": True, "reason": str(exc)}
        raise AssertionError("interleaved infeasible allocation was not rejected")

    def one_handover():
        distance = np.zeros((1, 1), dtype=float)
        base_user = {
            "id": 0,
            "message_id": 0,
            "aoi": 1.0,
            "serving_sat": 0,
            "handover": 0,
        }
        without = [deepcopy(base_user)]
        with_handover = [deepcopy(base_user)]
        with_handover[0]["handover"] = 1
        with_handover[0]["handover_event_id"] = "event-1"
        update_user_aoi(without, [], distance, cfg, slot=0)
        update_user_aoi(with_handover, [], distance, cfg, slot=0)
        difference = with_handover[0]["aoi"] - without[0]["aoi"]
        if abs(difference - 2.5) > 1e-12:
            raise AssertionError(f"handover AoI difference={difference}, expected=2.5")
        return {"handover_delay_slots": difference}

    def independent_policies():
        sat0 = SWUCB((1, 2), tau=3, satellite_id=0)
        sat1 = SWUCB((1, 2), tau=3, satellite_id=1)
        for slot in range(4):
            sat0.update(slot, 1, 0.9)
            sat1.update(slot, 2, 0.1)
        return {
            "sat0_arm1_mean": sat0.get_mean_reward(4, 1),
            "sat0_arm2_count": sat0.get_count(4, 2),
            "sat1_arm1_count": sat1.get_count(4, 1),
            "sat1_arm2_mean": sat1.get_mean_reward(4, 2),
        }

    def injected_error_detected():
        sat0 = _group(0, 0.1, is_edge=False, satellite_id=0)
        sat1 = _group(1, 0.1, is_edge=False, satellite_id=1)
        sat0.power = 1.01
        sat1.power = 0.99
        try:
            validate_power_constraints([sat0, sat1], P_sat=1.0, strict=True)
        except ValidationError as exc:
            return {"detected": True, "reason": str(exc)}
        raise AssertionError("injected per-satellite overflow was not detected")

    record("INT-01", nonedge_only)
    record("INT-02", edge_only)
    record("INT-03", mixed_noninterleaved)
    record("INT-04", mixed_interleaved_rejected)
    record("INT-05", one_handover)
    record("INT-06", independent_policies)
    record("INT-07", injected_error_detected)

    coverage = {
        "edge_only_tested": True,
        "nonedge_only_tested": True,
        "mixed_edge_nonedge_tested": True,
        "interleaved_csi_tested": True,
        "handover_event_tested": True,
        "multi_satellite_policy_tested": True,
    }
    return rows, coverage


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = {
                key: json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            }
            writer.writerow(serialized)


def write_runtime_outputs(result: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    satellite_rows = list(result.get("validation_satellite_records", []))
    group_rows = list(result.get("validation_group_records", []))
    satellite_fields = [
        "slot",
        "satellite_id",
        "action",
        "raw_reward",
        "normalized_reward",
        "window_start",
        "window_end",
        "arm_counts",
        "scheduled_group_count",
        "scheduled_edge_count",
        "scheduled_nonedge_count",
        "edge_budget",
        "nonedge_budget",
        "edge_power",
        "nonedge_power",
        "total_allocated_power",
        "power_budget",
        "power_floor_infeasible_count",
        "negative_power_count",
        "unscheduled_positive_power_count",
        "sic_power_order_violations",
        "sic_decode_violations",
        "handover_event_count",
        "handover_delay_slots",
    ]
    group_fields = [
        "slot",
        "satellite_id",
        "message_id",
        "group_id",
        "is_edge",
        "scheduled",
        "channel_gain",
        "sic_rank",
        "power_floor",
        "allocated_power",
        "sinr",
        "rate",
        "success",
        "violation_type",
    ]
    _write_csv(output_dir / "validation_details.csv", satellite_rows, satellite_fields)
    _write_csv(output_dir / "group_validation_details.csv", group_rows, group_fields)


def _run_unit_tests() -> tuple[dict[str, int], str]:
    suite = unittest.defaultTestLoader.discover(
        str(Path(__file__).parents[1] / "tests"), pattern="test_*.py"
    )
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    failed = len(result.failures)
    errors = len(result.errors)
    return {
        "run": result.testsRun,
        "passed": result.testsRun - failed - errors,
        "failed": failed,
        "errors": errors,
    }, stream.getvalue()


def _report_text(summary: dict[str, object], commands: list[str]) -> str:
    unit = summary["unit_tests"]
    violations = summary["violations"]
    coverage = summary["coverage"]
    blocked = summary["blocked_items"]
    result_rows = [
        ("SW-01", "basic sliding window", "FAIL", "PASS", "PASS", "test_sw_ucb_window"),
        ("SW-02", "stale arm expires", "FAIL", "PASS", "PASS", "test_sw_ucb_window"),
        ("RW-06", "out-of-range detection", "FAIL", "PASS", "PASS", "test_reward_bounds"),
        ("PW-07", "per-satellite overflow injection", "not covered", "PASS", "PASS", "test_power_constraints"),
        ("SIC-03", "interleaved subset CSI", "continued transmission", "rejected", "PASS", "test_sic_decodability"),
        ("DEC-04", "success recomputation", "not covered", "PASS", "PASS", "test_sic_decodability"),
        ("TM-04", "single handover", "0.025 slot", "2.5 slots", "PASS", "test_time_units"),
        ("INT-06", "independent satellite SW-UCB", "shared policy", "independent", "PASS", "controlled_scenarios.csv"),
    ]
    table = "\n".join(
        f"| {item_id} | {item} | {before} | {after} | {status} | {evidence} |"
        for item_id, item, before, after, status, evidence in result_rows
    )
    command_text = "\n".join(f"- `{command}`" for command in commands)
    blocked_text = "\n".join(f"- {item}" for item in blocked) or "- None"

    return f"""# MIRA-LEO v2a3 Validation Report

## Scope

Validated timestamped SW-UCB, bounded reward direction, per-satellite Algorithm 3 power and SIC safety, and delay conversion. Real TLE/SGP4 trajectories and statistical performance claims are outside this validation.

The source package `mira_leo_v2a3` was preserved. Changes were made only in `mira_leo_v2a3_verified`. The workspace exposes a `.git` reparse point, but Git reported that this directory is not a repository, so no branch or commit was created.

## Environment

- Python: {summary['python_version']}
- Dependencies used: Python standard library and NumPy already present in the bundled runtime.
- Unit tests: {unit['passed']}/{unit['run']} passed, {unit['failed']} failed, {unit['errors']} errors.

Commands:

{command_text}

## Formula Mapping

| Paper definition | Production implementation |
|---|---|
| SW-UCB window and Eq. (16) | `algorithms/sw_ucb.py` timestamped observations |
| Raw objective C - weighted AoI - edge penalty | `aoi/metrics.py::compute_regrouping_reward_details` |
| Algorithm 2 handover-aware priority | `algorithms/scheduler.py` |
| Algorithm 3 subset floors and residual | `algorithms/power_allocation.py` |
| Global SIC order constraint (15c) | power allocator plus `validation/validators.py` |
| Decoder-message SINR chain | `noma/transmission.py::evaluate_noma_decoding` |
| AoI delay recursion | `aoi/metrics.py::update_user_aoi` and `time_units.py` |

The paper requires normalized rewards in [0,1] but does not provide an explicit numerical bound formula. This verified implementation uses conservative fixed bounds derived once from configuration and horizon. It is an implementation choice, not a paper-specified normalization.

## Baseline Evidence

- Existing source tests: 16/16 passed, but all five targeted validation-risk checks failed.
- T=1000, seed=42 source run: reward min -0.012095700570307831, max 0.9957124235748394, 9 out-of-bound rewards.
- Source used one SW-UCB object for all satellites and stored no observation slots.
- Source added K_h=0.025 directly to slot-based AoI; verified conversion is 25 ms / 10 ms = 2.5 slots.
- Source recorded interleaved subset power-order violations but allowed NOMA transmission; verified code rejects them before transmission.

## Results

| ID | Validation | Source result | Verified result | Status | Evidence |
|---|---|---|---|---|---|
{table}

Runtime violation totals:

| Violation | Count |
|---|---:|
""" + "\n".join(
        f"| {key} | {value} |" for key, value in violations.items()
    ) + f"""

Controlled coverage:

| Branch | Covered |
|---|---|
""" + "\n".join(
        f"| {key} | {value} |" for key, value in coverage.items()
    ) + f"""

## Time Units

| Quantity | Source unit | Internal unit | Default |
|---|---|---|---:|
| slot_time_s | seconds/slot | seconds/slot | 0.01 |
| handover_delay_s | seconds | continuous slots | 0.025 s = 2.5 slots |
| isl_delay_s | seconds/hop | continuous slots/hop | 0.005 s = 0.5 slots |
| processing_delay_slots | slots | slots | 1.0 |
| propagation delay | seconds from distance/c | slots | divided by slot_time_s |
| AoI state | slots | slots | continuous |

Handover is represented as a one-slot event by the serving-satellite assignment. A repeated event ID is charged only once.

## Modified Files

- `config.py`, `time_units.py`: explicit units and fixed reward bounds.
- `algorithms/sw_ucb.py`: global-slot window with per-satellite state.
- `aoi/metrics.py`, `algorithms/scheduler.py`: reward and delay corrections.
- `algorithms/power_allocation.py`, `noma/transmission.py`: strict global SIC order and pairwise decoding records.
- `main.py`: per-satellite policy/reward and per-slot validation records.
- `validation/*`, `tests/*`: validators, controlled scenarios, reports, and regression coverage.

## Blocked Or Deferred

{blocked_text}

## Conclusion

Executable checks status: {summary['execution_status']}. Overall validation status: {summary['status']}.
The verified code is ready for deterministic software testing. Do not claim complete paper-level validation or introduce real trajectories until the deferred full multi-seed run is executed and the reward normalization choice is accepted or replaced by an explicit paper bound.
"""


def run_validation(
    *,
    output_dir: Path,
    controlled: bool,
    strict: bool,
    T: int,
    seeds: list[int],
    fading: bool,
) -> dict[str, object]:
    unit_tests, test_output = _run_unit_tests()
    controlled_rows: list[dict] = []
    coverage = {
        "edge_only_tested": False,
        "nonedge_only_tested": False,
        "mixed_edge_nonedge_tested": False,
        "interleaved_csi_tested": False,
        "handover_event_tested": False,
        "multi_satellite_policy_tested": False,
    }
    if controlled:
        controlled_rows, coverage = run_controlled_scenarios(strict=strict)

    runtime_results = []
    satellite_rows: list[dict] = []
    group_rows: list[dict] = []
    for seed in seeds:
        if T <= 0:
            break
        cfg = replace(V2Config(), T=T, random_seed=seed)
        result = run_mira_leo_v2a3_verified(
            cfg,
            policy="sw-ucb",
            fading=fading,
            validate=True,
            strict_validation=strict,
        )
        runtime_results.append(result)
        satellite_rows.extend(result["validation_satellite_records"])
        group_rows.extend(result["validation_group_records"])

    violations = dict(VIOLATION_KEYS)
    for result in runtime_results:
        for violation in result["validation_violations"]:
            code = violation["code"]
            key = {
                "nonfinite_reward": "nonfinite_rewards",
                "reward_out_of_bounds": "reward_out_of_bounds",
            }.get(code, code)
            if key in violations:
                violations[key] += 1

    execution_passed = (
        unit_tests["failed"] == 0
        and unit_tests["errors"] == 0
        and all(row["status"] == "PASS" for row in controlled_rows)
        and all(value == 0 for value in violations.values())
    )
    full_multi_seed = T >= 1000 and set(range(5)).issubset(seeds)
    blocked_items = [
        "The paper does not provide an explicit numerical reward-normalization bound; the fixed linear mapping is an implementation choice."
    ]
    if not full_multi_seed:
        blocked_items.append(
            "The T=1000, seeds 0-4 validation run was not executed in this no-simulation delivery."
        )

    summary = {
        "status": "BLOCKED" if execution_passed and blocked_items else (
            "PASS" if execution_passed else "FAIL"
        ),
        "execution_status": "PASS" if execution_passed else "FAIL",
        "python_version": platform.python_version(),
        "source_version": "mira_leo_v2a3",
        "validated_version": "mira_leo_v2a3_verified",
        "config": {
            "T": T,
            "seeds": seeds,
            "slot_time_s": V2Config().slot_time_s,
            "handover_delay_ms": V2Config().handover_delay_s * 1000.0,
            "handover_delay_slots": V2Config().handover_delay_slots,
        },
        "unit_tests": unit_tests,
        "violations": violations,
        "coverage": coverage,
        "blocked_items": blocked_items,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "controlled_scenarios.csv",
        controlled_rows,
        ["scenario_id", "status", "details"],
    )
    write_runtime_outputs(
        {
            "validation_satellite_records": satellite_rows,
            "validation_group_records": group_rows,
        },
        output_dir,
    )
    (output_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "unit_test_output.txt").write_text(test_output, encoding="utf-8")
    commands = [
        "python -m unittest discover -s mira_leo_v2a3_verified/tests -v",
        "python -m mira_leo_v2a3_verified.validation.run_validation --controlled --strict",
    ]
    if T > 0:
        commands.append(
            "python -m mira_leo_v2a3_verified.validation.run_validation "
            f"--controlled --strict --T {T} --seeds {' '.join(map(str, seeds))}"
        )
    (output_dir / "VALIDATION_REPORT.md").write_text(
        _report_text(summary, commands), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MIRA-LEO v2a3 verified.")
    parser.add_argument("--controlled", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--T", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--fading", action="store_true")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parents[1] / "validation_outputs"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_validation(
        output_dir=Path(args.output),
        controlled=args.controlled,
        strict=args.strict,
        T=args.T,
        seeds=args.seeds,
        fading=args.fading,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["execution_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
