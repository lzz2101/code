"""Legacy parameter-search APIs, retained for historical configurations only.

The fixed phi/B CLI and paper-results pipeline never invoke these searches.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from .analysis import build_comparison_tables, select_finalists
from .calibration import run_calibration
from .config import ACTION_SETS, StudyConfig, load_config
from .simulation import run_simulation
from .storage import atomic_write_json


def _apply_candidate(config: StudyConfig, dimension: str, value) -> StudyConfig:
    if dimension == "phi_theta_deg":
        return config.with_overrides({"algorithm.phi_theta_deg": float(value)})
    if dimension == "action_set_name":
        name = str(value)
        return config.with_overrides(
            {"algorithm.action_set_name": name, "algorithm.action_set": ACTION_SETS[name]}
        )
    if dimension == "V_c":
        fraction = float(value)
        return config.with_overrides(
            {"algorithm.V_c": fraction, "algorithm.V_nc": 1.0 - fraction}
        )
    if dimension == "R_theta":
        return config.with_overrides({"algorithm.R_theta": float(value)})
    if dimension == "beta_B":
        return config.with_overrides({"algorithm.beta_B": float(value)})
    raise ValueError(f"unknown study dimension: {dimension}")


def _resolve_candidate(config: StudyConfig, output_root: Path) -> tuple[StudyConfig, Path]:
    calibration_root = output_root / "candidate_calibrations" / config.config_id
    resolved_path = calibration_root / "calibration" / "resolved_config.json"
    if resolved_path.exists():
        return load_config(resolved_path), resolved_path
    result = run_calibration(config, calibration_root)
    resolved = result["resolved_config"]
    if resolved is None:
        status = result["calibration"]["B_status"]
        raise ValueError(
            f"candidate {config.config_id} cannot run because pilot calibration ended with {status}"
        )
    return resolved, resolved_path


def _run_resolved(
    config: StudyConfig,
    seeds: list[int],
    output_root: Path,
    resume: bool,
    save_group_trace: bool = False,
) -> None:
    for seed in seeds:
        summary_path = output_root / "runs" / config.config_id / f"seed_{seed}" / "summary.json"
        if resume and summary_path.exists():
            continue
        run_simulation(
            config,
            channel_seed=seed,
            policy="sw-ucb",
            output_directory=output_root,
            resume=resume,
            save_group_trace=save_group_trace,
        )


def screen_parameters(config: StudyConfig, output_directory: str | Path, resume: bool = False) -> dict:
    output_root = Path(output_directory)
    candidates: dict[str, tuple[StudyConfig, dict]] = {}
    for dimension in ("phi_theta_deg", "action_set_name", "V_c", "R_theta", "beta_B"):
        for value in config.get(f"search.{dimension}"):
            candidate = _apply_candidate(config, dimension, value)
            candidates.setdefault(
                candidate.config_id,
                (candidate, {"dimension": dimension, "value": value}),
            )
    manifest = []
    configured = config.get("randomness.validation_channel_seeds")
    seeds = (
        [int(value) for value in configured[:3]]
        if configured is not None
        else [int(config.get("randomness.fixed_channel_seed"))]
    )
    for initial_id, (candidate, metadata) in sorted(candidates.items()):
        resolved, resolved_path = _resolve_candidate(candidate, output_root)
        _run_resolved(resolved, seeds, output_root, resume)
        manifest.append(
            {
                "initial_config_id": initial_id,
                "resolved_config_id": resolved.config_id,
                "resolved_config_path": str(resolved_path),
                **metadata,
            }
        )
    atomic_write_json(output_root / "screen_manifest.json", manifest)
    baseline, _ = _resolve_candidate(config, output_root)
    comparison = build_comparison_tables(output_root, baseline.config_id)
    return {"manifest": manifest, "comparison": comparison}


def _top_values_from_screen(output_root: Path) -> dict[str, list]:
    manifest = json.loads((output_root / "screen_manifest.json").read_text(encoding="utf-8"))
    comparison = build_comparison_tables(output_root)
    aggregate = {row["config_id"]: row for row in comparison["aggregate"]}
    dimensions: dict[str, list[tuple[float, object]]] = {}
    for item in manifest:
        row = aggregate.get(item["resolved_config_id"])
        if not row or not row["legal_for_ranking"] or item["dimension"] == "R_theta":
            continue
        dimensions.setdefault(item["dimension"], []).append(
            (float(row["weighted_mean_aoi_ms"]), item["value"])
        )
    output = {}
    for dimension, values in dimensions.items():
        selected = []
        for _, value in sorted(values, key=lambda item: (item[0], str(item[1]))):
            if value not in selected:
                selected.append(value)
            if len(selected) == 2:
                break
        output[dimension] = selected
    return output


def joint_search(config: StudyConfig, output_directory: str | Path, resume: bool = False) -> dict:
    output_root = Path(output_directory)
    top = _top_values_from_screen(output_root)
    required = ("phi_theta_deg", "action_set_name", "V_c", "beta_B")
    value_lists = [top.get(name, []) for name in required]
    if any(not values for values in value_lists):
        raise ValueError("screen stage did not produce a legal candidate for every joint dimension")
    combinations = list(itertools.product(*value_lists))[:16]
    manifest = []
    configured = config.get("randomness.validation_channel_seeds")
    seeds = (
        [int(value) for value in configured]
        if configured is not None
        else [int(config.get("randomness.fixed_channel_seed"))]
    )
    for values in combinations:
        candidate = config.with_overrides({"algorithm.R_theta": 0.03})
        metadata = {}
        for dimension, value in zip(required, values):
            candidate = _apply_candidate(candidate, dimension, value)
            metadata[dimension] = value
        resolved, resolved_path = _resolve_candidate(candidate, output_root)
        _run_resolved(resolved, seeds, output_root, resume)
        manifest.append(
            {
                "resolved_config_id": resolved.config_id,
                "resolved_config_path": str(resolved_path),
                **metadata,
            }
        )
    baseline, baseline_path = _resolve_candidate(config.with_overrides({"algorithm.R_theta": 0.03}), output_root)
    _run_resolved(baseline, seeds, output_root, resume)
    if all(item["resolved_config_id"] != baseline.config_id for item in manifest):
        manifest.append(
            {
                "resolved_config_id": baseline.config_id,
                "resolved_config_path": str(baseline_path),
                "baseline": True,
            }
        )
    atomic_write_json(output_root / "joint_manifest.json", manifest)
    comparison = build_comparison_tables(output_root, baseline.config_id)
    joint_ids = {item["resolved_config_id"] for item in manifest}
    finalists = select_finalists(
        [row for row in comparison["aggregate"] if row["config_id"] in joint_ids], 3
    )
    path_by_id = {
        item["resolved_config_id"]: item["resolved_config_path"] for item in manifest
    }
    selection = {
        "dataset": "validation",
        "selection_rule": "minimum full-window weighted mean AoI among legal configurations",
        "search_scope": "at most 16 combinations from top-two legal screen candidates plus baseline",
        "finalists": [
            {**row, "resolved_config_path": path_by_id[row["config_id"]]}
            for row in finalists
        ],
    }
    atomic_write_json(output_root / "selection_validation.json", selection)
    return selection


def evaluate_selection(selection_path: str | Path, output_directory: str | Path, resume: bool = False) -> dict:
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    output_root = Path(output_directory)
    completed = []
    for finalist in selection["finalists"]:
        config = load_config(finalist["resolved_config_path"])
        configured = config.get("randomness.test_channel_seeds")
        seeds = (
            [int(value) for value in configured]
            if configured is not None
            else [int(config.get("randomness.fixed_channel_seed"))]
        )
        _run_resolved(config, seeds, output_root, resume, save_group_trace=True)
        completed.append(config.config_id)
    comparison = build_comparison_tables(output_root)
    return {"evaluated_config_ids": completed, "comparison": comparison}


def run_robustness(selection_path: str | Path, output_directory: str | Path, resume: bool = False) -> dict:
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    if not selection.get("finalists"):
        raise ValueError("selection contains no finalist")
    base = load_config(selection["finalists"][0]["resolved_config_path"])
    variants = [
        ("layout_seed_43", {"devices.layout_seed": 43}),
        ("layout_seed_44", {"devices.layout_seed": 44}),
        ("reference_snr_5_db", {"channel.reference_snr_db": 5.0}),
        ("reference_snr_15_db", {"channel.reference_snr_db": 15.0}),
        ("weights_0.5_0.3_0.2", {"services.weights": [0.5, 0.3, 0.2]}),
        ("weights_equal", {"services.weights": [1.0 / 3.0] * 3}),
        ("geometric_isl", {"delay.isl_mode": "geometric_isl"}),
        ("tau_2.5_s", {"algorithm.sw_ucb_window_s": 2.5}),
        ("tau_40_s", {"algorithm.sw_ucb_window_s": 40.0}),
        ("slot_10_ms", {"experiment.slot_dt_s": 0.01}),
        ("B_zero_diagnostic", {"algorithm.B_penalty": 0.0, "algorithm.beta_B": 0.0}),
        ("phi_10_diagnostic", {"algorithm.phi_theta_deg": 10.0}),
    ]
    output_root = Path(output_directory)
    manifest = []
    configured = base.get("randomness.validation_channel_seeds")
    seeds = (
        [int(value) for value in configured]
        if configured is not None
        else [int(base.get("randomness.fixed_channel_seed"))]
    )
    for name, overrides in variants:
        candidate = base.with_overrides(overrides)
        if name == "B_zero_diagnostic":
            resolved = candidate
            resolved_path = output_root / "robustness_configs" / f"{name}.json"
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            from .config import save_config

            save_config(resolved, resolved_path)
        else:
            resolved, resolved_path = _resolve_candidate(candidate, output_root)
        _run_resolved(resolved, seeds, output_root, resume)
        manifest.append(
            {
                "variant": name,
                "config_id": resolved.config_id,
                "config_path": str(resolved_path),
            }
        )
    atomic_write_json(output_root / "robustness_manifest.json", manifest)
    return {"manifest": manifest, "comparison": build_comparison_tables(output_root)}


def write_report(output_directory: str | Path) -> dict:
    output_root = Path(output_directory)
    comparison = build_comparison_tables(output_root)
    legal = [row for row in comparison["aggregate"] if row["legal_for_ranking"]]
    legal.sort(key=lambda row: (row["weighted_mean_aoi_ms"], row["config_id"]))
    best = legal[0] if legal else None
    search_range = None
    for resolved_path in (output_root / "candidate_calibrations").glob(
        "*/calibration/resolved_config.json"
    ) if (output_root / "candidate_calibrations").exists() else []:
        try:
            search_range = load_config(resolved_path).get("search")
            break
        except (OSError, ValueError):
            continue
    best_payload = {
        "status": "AVAILABLE" if best else "NO_LEGAL_COMPLETED_RUN",
        "selection_dataset": "validation/test as recorded in selection_validation.json",
        "scope": "synthetic constellation and finite searched ranges only",
        "global_optimum_claimed": False,
        "best": best,
        "search_range": search_range,
    }
    atomic_write_json(output_root / "best_parameters.json", best_payload)
    generated_plots = []
    plot_status = "not attempted because no completed configuration exists"
    if best and best.get("config_path"):
        try:
            from .plotting import generate_all_plots

            generated_plots = generate_all_plots(load_config(best["config_path"]), output_root)
            plot_status = f"generated {len(generated_plots)} figure sets"
        except Exception as exc:
            from .errors import OptionalDependencyError

            if isinstance(exc, OptionalDependencyError):
                plot_status = f"not generated: {exc}"
            else:
                raise
    best_payload["generated_plots"] = generated_plots
    best_payload["plot_status"] = plot_status
    atomic_write_json(output_root / "best_parameters.json", best_payload)
    lines = [
        "# MIRA-LEO Synthetic Parameter Study Report",
        "",
        "This report is generated only from completed run artifacts. No expected or mock result is inserted.",
        "",
        f"- Completed run records: {len(comparison['runs'])}",
        f"- Aggregated configurations: {len(comparison['aggregate'])}",
        f"- Legal configurations: {len(legal)}",
        f"- Plot status: {plot_status}",
        "- Claim boundary: recommendations apply only to the configured synthetic constellation, traffic, channel, slot, and search ranges.",
        "",
        "## Selection",
        "",
    ]
    if best:
        lines.extend(
            [
                f"Best legal config ID: `{best['config_id']}`.",
                f"Weighted mean AoI: {best['weighted_mean_aoi_ms']:.6g} ms.",
                "This is a finite-search recommendation, not a global or real-network optimum.",
            ]
        )
    else:
        lines.append("No legal completed configuration is available for ranking.")
    (output_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return best_payload
