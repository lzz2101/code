from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .storage import atomic_write_json, write_csv


def parameter_identifiability(edge_group_counts: list[int], legal_run_count: int) -> dict:
    if legal_run_count <= 0:
        return {"status": "NOT_IDENTIFIABLE", "reason": "no_legal_power_configuration"}
    if not edge_group_counts or max(edge_group_counts) == 0:
        return {"status": "NOT_IDENTIFIABLE", "reason": "no_edge_groups"}
    if len(set(edge_group_counts)) == 1:
        return {"status": "NOT_IDENTIFIABLE", "reason": "no_edge_count_variation"}
    return {"status": "IDENTIFIABLE", "reason": None}


def discover_runs(output_directory: str | Path) -> list[dict]:
    root = Path(output_directory)
    rows = []
    for summary_path in root.glob("runs/*/seed_*/summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config_path = summary_path.parent / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        rows.append(
            {
                **summary,
                "summary_path": str(summary_path),
                "config_path": str(config_path),
                "phi_theta_deg": config.get("algorithm", {}).get("phi_theta_deg"),
                "action_set_name": config.get("algorithm", {}).get("action_set_name"),
                "V_c": config.get("algorithm", {}).get("V_c"),
                "V_nc": config.get("algorithm", {}).get("V_nc"),
                "R_theta": config.get("algorithm", {}).get("R_theta"),
                "B_penalty": config.get("algorithm", {}).get("B_penalty"),
                "power_mode": config.get("algorithm", {}).get("power_mode"),
                "handover_mode": config.get("delay", {}).get("handover_mode"),
                "noise_model": config.get("channel", {}).get("noise_model"),
            }
        )
    return sorted(rows, key=lambda row: (row["config_id"], row["channel_seed"]))


def _mean_ci(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    data = np.asarray(values, dtype=float)
    mean = float(np.mean(data))
    if len(data) < 2:
        return mean, None, None
    half = 1.96 * float(np.std(data, ddof=1)) / math.sqrt(len(data))
    return mean, mean - half, mean + half


def aggregate_runs(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["config_id"]].append(row)
    aggregate = []
    for config_id, group in sorted(grouped.items()):
        first = group[0]
        legal = all(row["status"] == "COMPLETED" for row in group)
        aoi = [float(row["weighted_mean_aoi_ms"]) for row in group]
        goodput = [float(row.get("capacity_goodput_bps", 0.0)) for row in group]
        tail = [float(row["aoi_p95_ms"]) for row in group]
        overhead = [float(row["grouping_membership_changes"]) for row in group]
        aoi_mean, aoi_low, aoi_high = _mean_ci(aoi)
        goodput_mean, _, _ = _mean_ci(goodput)
        tail_mean, _, _ = _mean_ci(tail)
        overhead_mean, _, _ = _mean_ci(overhead)
        aggregate.append(
            {
                "config_id": config_id,
                "run_count": len(group),
                "legal_for_ranking": int(legal),
                "statuses": ";".join(sorted({row["status"] for row in group})),
                "weighted_mean_aoi_ms": aoi_mean,
                "weighted_mean_aoi_ci95_low_ms": aoi_low,
                "weighted_mean_aoi_ci95_high_ms": aoi_high,
                "capacity_goodput_bps": goodput_mean,
                "aoi_p95_ms": tail_mean,
                "grouping_membership_changes": overhead_mean,
                "phi_theta_deg": first.get("phi_theta_deg"),
                "action_set_name": first.get("action_set_name"),
                "V_c": first.get("V_c"),
                "R_theta": first.get("R_theta"),
                "B_penalty": first.get("B_penalty"),
                "power_mode": first.get("power_mode"),
                "handover_mode": first.get("handover_mode"),
                "noise_model": first.get("noise_model"),
                "config_path": first.get("config_path"),
            }
        )
    return aggregate


def paired_differences(rows: list[dict], baseline_config_id: str, bootstrap_seed: int = 20260930, samples: int = 10000) -> list[dict]:
    by_config_seed = {
        (row["config_id"], int(row["channel_seed"])): row for row in rows
    }
    baseline_seeds = sorted(
        seed for config_id, seed in by_config_seed if config_id == baseline_config_id
    )
    rng = np.random.default_rng(bootstrap_seed)
    output = []
    for config_id in sorted({row["config_id"] for row in rows}):
        differences = []
        for seed in baseline_seeds:
            baseline = by_config_seed.get((baseline_config_id, seed))
            candidate = by_config_seed.get((config_id, seed))
            if baseline and candidate:
                differences.append(
                    float(candidate["weighted_mean_aoi_ms"])
                    - float(baseline["weighted_mean_aoi_ms"])
                )
        if not differences:
            continue
        data = np.asarray(differences, dtype=float)
        indices = rng.integers(0, len(data), size=(samples, len(data)))
        means = np.mean(data[indices], axis=1)
        output.append(
            {
                "baseline_config_id": baseline_config_id,
                "candidate_config_id": config_id,
                "paired_seed_count": len(data),
                "mean_aoi_difference_ms": float(np.mean(data)),
                "bootstrap_ci95_low_ms": float(np.quantile(means, 0.025)),
                "bootstrap_ci95_high_ms": float(np.quantile(means, 0.975)),
            }
        )
    return output


def pareto_rows(aggregate: list[dict]) -> list[dict]:
    legal = [row for row in aggregate if row["legal_for_ranking"]]
    output = []
    for candidate in legal:
        dominated = False
        for other in legal:
            if other is candidate:
                continue
            no_worse = (
                other["weighted_mean_aoi_ms"] <= candidate["weighted_mean_aoi_ms"]
                and other["aoi_p95_ms"] <= candidate["aoi_p95_ms"]
                and other["grouping_membership_changes"] <= candidate["grouping_membership_changes"]
                and other["capacity_goodput_bps"] >= candidate["capacity_goodput_bps"]
            )
            strictly = (
                other["weighted_mean_aoi_ms"] < candidate["weighted_mean_aoi_ms"]
                or other["aoi_p95_ms"] < candidate["aoi_p95_ms"]
                or other["grouping_membership_changes"] < candidate["grouping_membership_changes"]
                or other["capacity_goodput_bps"] > candidate["capacity_goodput_bps"]
            )
            if no_worse and strictly:
                dominated = True
                break
        if not dominated:
            output.append(candidate)
    return output


def build_comparison_tables(output_directory: str | Path, baseline_config_id: str | None = None) -> dict:
    output = Path(output_directory)
    comparison = output / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    rows = discover_runs(output)
    aggregate = aggregate_runs(rows)
    write_csv(comparison / "all_runs.csv", rows)
    write_csv(comparison / "aggregate_by_config.csv", aggregate)
    pareto = pareto_rows(aggregate)
    write_csv(comparison / "pareto.csv", pareto)
    paired = []
    if baseline_config_id is not None:
        paired = paired_differences(rows, baseline_config_id)
    write_csv(comparison / "paired_differences.csv", paired)
    return {"runs": rows, "aggregate": aggregate, "pareto": pareto, "paired": paired}


def select_finalists(aggregate: list[dict], maximum: int = 3) -> list[dict]:
    legal = [row for row in aggregate if row["legal_for_ranking"]]
    legal.sort(
        key=lambda row: (
            row["weighted_mean_aoi_ms"],
            len(str(row["action_set_name"])),
            row["config_id"],
        )
    )
    return legal[:maximum]
