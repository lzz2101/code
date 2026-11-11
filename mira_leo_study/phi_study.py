"""Archived angle-search API; unavailable from the fixed formal CLI.

Historical artifacts and regression tests may still import this module.
Use p1_ablation.py for the current six-run experiment.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .config import PHI_THETA_CANDIDATES_DEG, StudyConfig, validate_phi_study_config
from .plotting import plot_phi_study_results
from .simulation import BLOCKED_POLICIES, run_simulation
from .storage import atomic_write_json, write_csv


AVAILABLE_PHI_POLICIES = ("mira_sw_ucb", "random_grouping", "fixed_grouping")
POLICY_LABELS = {
    "mira_sw_ucb": "MIRA-LEO",
    "random_grouping": "Random grouping",
    "fixed_grouping": "Fixed(a=4)",
    "aim_static": "AIM-static",
    "ppo_multicast": "PPO-multicast",
    "ga_grouping": "GA-grouping",
}


def load_B_by_phi(path: str | Path) -> dict[int, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("fixed_channel_seed", -1)) != 0:
        raise ValueError("B-by-phi artifact must use fixed_channel_seed=0")
    source = payload.get("B_penalty_star_by_phi")
    if not isinstance(source, dict):
        raise ValueError("B-by-phi artifact is missing B_penalty_star_by_phi")
    all_entries = {int(key): value for key, value in source.items()}
    missing = [phi for phi in PHI_THETA_CANDIDATES_DEG if phi not in all_entries]
    if missing:
        raise ValueError(f"B-by-phi artifact is missing configured angles: {missing}")
    return {phi: all_entries[phi] for phi in PHI_THETA_CANDIDATES_DEG}


def resolved_phi_config(config: StudyConfig, phi: int, entry: dict) -> StudyConfig:
    if entry.get("status") != "OK" or entry.get("B_penalty_star") is None:
        raise ValueError(f"phi={phi} has no identifiable pilot B and cannot run formally")
    identity = entry.get("reward_bounds_identity")
    lower = entry.get("reward_lower")
    upper = entry.get("reward_upper")
    if identity is None or lower is None or upper is None:
        raise ValueError(f"phi={phi} is missing frozen reward bounds")
    return config.with_overrides(
        {
            "algorithm.phi_theta_deg": float(phi),
            "algorithm.B_penalty": float(entry["B_penalty_star"]),
            "algorithm.reward_bounds": {
                "lower": float(lower),
                "upper": float(upper),
                "identity": str(identity),
            },
        }
    )


def _summary_row(phi: int, policy: str, summary: dict) -> dict:
    return {
        "phi_theta_deg": phi,
        "policy": policy,
        "policy_label": POLICY_LABELS[policy],
        "status": summary.get("status"),
        "fixed_channel_seed": summary.get("channel_seed"),
        "phi_theta_applicable": True,
        "B_penalty_star": summary.get("B_penalty_star"),
        "reward_bounds_identity": summary.get("reward_bounds_identity"),
        "slots_completed": summary.get("slots_completed"),
        "weighted_mean_aoi_after_warmup_ms": summary.get(
            "weighted_mean_aoi_after_warmup_ms"
        ),
        "weighted_sum_aoi_p50_ms": summary.get("weighted_sum_aoi_p50_ms"),
        "weighted_sum_aoi_p95_ms": summary.get("weighted_sum_aoi_p95_ms"),
        "weighted_sum_aoi_p99_ms": summary.get("weighted_sum_aoi_p99_ms"),
        "successful_group_count": summary.get("successful_group_count"),
        "scheduled_group_count": summary.get("scheduled_group_count"),
        "conditional_group_success_ratio": summary.get(
            "conditional_group_success_ratio"
        ),
        "edge_group_count": summary.get("edge_group_count"),
        "mean_interference_power": summary.get("mean_interference_power"),
        "p95_interference_power": summary.get("p95_interference_power"),
        "regrouping_events_total": summary.get("regrouping_events_total"),
        "regrouping_events_per_device_per_minute": summary.get(
            "regrouping_events_per_device_per_minute"
        ),
        "action_histogram": json.dumps(summary.get("action_histogram", {}), sort_keys=True),
    }


def select_phi_theta(rows: list[dict], config: StudyConfig, b_entries: dict[int, dict]) -> dict:
    candidates = [
        row
        for row in rows
        if row["policy"] == "mira_sw_ucb"
        and row["status"] == "COMPLETED"
        and int(row["fixed_channel_seed"]) == 0
        and int(row["slots_completed"]) == config.slot_count
        and row["weighted_mean_aoi_after_warmup_ms"] is not None
    ]
    if {int(row["phi_theta_deg"]) for row in candidates} != set(PHI_THETA_CANDIDATES_DEG):
        raise ValueError("selection requires completed MIRA runs for every configured phi candidate")
    candidates.sort(
        key=lambda row: (
            float(row["weighted_mean_aoi_after_warmup_ms"]),
            int(row["phi_theta_deg"]),
        )
    )
    minimum_aoi = float(candidates[0]["weighted_mean_aoi_after_warmup_ms"])
    tied = [
        row
        for row in candidates
        if math.isclose(
            float(row["weighted_mean_aoi_after_warmup_ms"]),
            minimum_aoi,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    ]
    tied.sort(
        key=lambda row: (
            math.inf
            if row["mean_interference_power"] is None
            else float(row["mean_interference_power"]),
            math.inf
            if row["regrouping_events_total"] is None
            else int(row["regrouping_events_total"]),
            int(row["phi_theta_deg"]),
        )
    )
    best = tied[0]
    selected_phi = int(best["phi_theta_deg"])
    return {
        "selected_phi_theta_deg": selected_phi,
        "selection_metric": "weighted_mean_aoi_after_warmup_ms",
        "fixed_channel_seed": 0,
        "B_penalty_star_for_selected_phi": float(
            b_entries[selected_phi]["B_penalty_star"]
        ),
        "B_penalty_star_by_phi": {
            str(phi): b_entries[phi]["B_penalty_star"]
            for phi in PHI_THETA_CANDIDATES_DEG
        },
        "R_theta_bps_hz": float(config.get("algorithm.R_theta")),
        "slot_dt_s": float(config.get("experiment.slot_dt_s")),
        "bandwidth_hz": float(config.get("services.bandwidth_hz")),
        "scs_hz": float(config.get("radio.scs_hz")),
        "fc_hz": float(config.get("channel.carrier_hz")),
        "M": int(config.get("services.messages_per_active_satellite")),
        "N": int(config.get("services.subscribers_per_message")),
        "weights": list(config.get("services.weights")),
        "candidate_table": candidates,
        "tie_note": (
            None
            if len(tied) == 1
            else (
                "AoI values were numerically tied; lower interference, then lower "
                "regrouping overhead, then lower phi_theta were deterministic tie-break diagnostics."
            )
        ),
        "selection_reason": (
            "lowest fixed-seed post-warmup aggregated weighted-sum AoI"
            if len(tied) == 1
            else "numerically tied AoI resolved by deterministic diagnostics"
        ),
        "interpretation": (
            "Single fixed-randomness comparison of angle-specific calibrated "
            "(phi_theta, B_star(phi_theta)) configurations; not a multi-seed estimate."
        ),
    }


def run_phi_validation_sweep(
    config: StudyConfig,
    b_by_phi_path: str | Path,
    output_directory: str | Path,
    *,
    resume: bool = False,
    max_slots: int | None = None,
) -> dict:
    validate_phi_study_config(config)
    if max_slots is not None and int(max_slots) <= 0:
        raise ValueError("max_slots must be positive")
    entries = load_B_by_phi(b_by_phi_path)
    fixed_seed = int(config.get("randomness.fixed_channel_seed"))
    root = Path(output_directory)
    is_formal_length = max_slots is None or int(max_slots) >= config.slot_count
    artifact_root = (
        root
        if is_formal_length
        else root / "debug" / f"max_slots_{int(max_slots)}"
    )
    rows = []
    baseline_rows = []
    for phi in PHI_THETA_CANDIDATES_DEG:
        resolved = resolved_phi_config(config, phi, entries[phi])
        for policy in AVAILABLE_PHI_POLICIES:
            run_dir = (
                artifact_root
                / "validation"
                / f"phi_{phi}"
                / policy
                / "fixed_seed_0"
            )
            summary_path = run_dir / "summary.json"
            summary = None
            if resume and summary_path.exists():
                candidate_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                expected_slots = config.slot_count if max_slots is None else min(
                    int(max_slots), config.slot_count
                )
                if (
                    candidate_summary.get("config_id") == resolved.config_id
                    and candidate_summary.get("policy") == policy
                    and int(candidate_summary.get("channel_seed", -1)) == fixed_seed
                    and int(candidate_summary.get("slots_completed", -1)) == expected_slots
                ):
                    summary = candidate_summary
            if summary is None:
                summary = run_simulation(
                    resolved,
                    channel_seed=fixed_seed,
                    policy=policy,
                    fixed_action=4,
                    max_slots=max_slots,
                    run_directory=run_dir,
                    resume=resume,
                    save_group_trace=True,
                )["summary"]
            row = _summary_row(phi, policy, summary)
            rows.append(row)
            if policy != "mira_sw_ucb":
                baseline_rows.append(row)
        for policy, status in BLOCKED_POLICIES.items():
            baseline_rows.append(
                {
                    "phi_theta_deg": phi,
                    "policy": policy,
                    "policy_label": POLICY_LABELS[policy],
                    "status": status,
                    "fixed_channel_seed": fixed_seed,
                    "phi_theta_applicable": policy != "aim_static",
                }
            )
    comparison = artifact_root / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    write_csv(comparison / "phi_fixed_seed_summary.csv", rows)
    write_csv(comparison / "phi_validation_summary.csv", rows)
    write_csv(comparison / "baseline_summary_by_phi.csv", baseline_rows)
    selected = None
    if is_formal_length:
        selected = select_phi_theta(rows, config, entries)
        atomic_write_json(comparison / "selected_phi.json", selected)
    return {
        "completed_run_count": len(rows),
        "blocked_baseline_count": len(BLOCKED_POLICIES) * len(PHI_THETA_CANDIDATES_DEG),
        "selected_phi": selected,
        "debug_truncation": None if is_formal_length else int(max_slots),
        "artifact_root": str(artifact_root),
    }


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_phi_report(output_directory: str | Path) -> dict:
    root = Path(output_directory)
    comparison = root / "comparison"
    selected_path = comparison / "selected_phi.json"
    selected = (
        json.loads(selected_path.read_text(encoding="utf-8"))
        if selected_path.exists()
        else None
    )
    rows = _read_csv(comparison / "phi_fixed_seed_summary.csv")
    baselines = _read_csv(comparison / "baseline_summary_by_phi.csv")
    b_rows = _read_csv(root / "pilot_B" / "B_by_phi.csv")
    b_json_path = root / "pilot_B" / "B_by_phi.json"
    if not b_rows and b_json_path.exists():
        b_payload = json.loads(b_json_path.read_text(encoding="utf-8"))
        b_rows = list(b_payload.get("B_penalty_star_by_phi", {}).values())
    snapshot_path = root / "pilot_B" / "pilot_parameter_snapshot.json"
    snapshot = (
        json.loads(snapshot_path.read_text(encoding="utf-8"))
        if snapshot_path.exists()
        else None
    )
    figures = plot_phi_study_results(root)
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    def display(value) -> str:
        if value is None or value == "":
            return "NOT_AVAILABLE"
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        if isinstance(value, float):
            return f"{value:.12g}"
        return str(value)

    lines = [
        "# MIRA-LEO Phi-Theta Study Report",
        "",
        "This report reads completed fixed-seed artifacts only; it never inserts mock curves or values.",
        "",
        "## Study Interpretation",
        "",
        "- One channel realization is fixed at seed 0; no multi-seed confidence interval is computed.",
        "- Each angle uses its own pilot-calibrated B_penalty_star(phi), frozen before its formal run.",
        "- SW-UCB adapts regrouping action a(t), never B_penalty.",
        "- Radio bandwidth is 50 MHz and is distinct from the edge-count penalty coefficient.",
        "- The result compares angle-specific calibrated configurations, not a pure one-factor causal effect.",
        "",
        "## Exact Parameters",
        "",
    ]
    parameter_fields = (
        ("Decision slot (s)", "slot_dt_s"),
        ("Duration (s)", "duration_s"),
        ("Warmup (s)", "warmup_report_s"),
        ("Carrier frequency (Hz)", "carrier_hz"),
        ("SCS metadata (Hz)", "scs_hz"),
        ("Radio bandwidth (Hz)", "bandwidth_hz"),
        ("Reference bandwidth (Hz)", "reference_bandwidth_hz"),
        ("R_theta (bit/s/Hz)", "R_theta_bps_hz"),
        ("M", "messages_per_active_satellite"),
        ("N per message", "subscribers_per_message"),
        ("Weights", "weights"),
        ("V_c", "V_c"),
        ("V_nc", "V_nc"),
        ("V_phi", "V_phi"),
        ("Normalized satellite power", "normalized_satellite_power"),
        ("phi_min (degree)", "phi_min_deg"),
        ("phi_theta candidates (degree)", "phi_theta_candidates_deg"),
        ("Legal actions", "action_set"),
        ("SW-UCB xi", "sw_ucb_xi"),
        ("SW-UCB window (s)", "sw_ucb_window_s"),
        ("Shadowed-Rician", "shadowed_rician"),
        ("Capacity, continuous", "schedule_capacity_continuous"),
        ("Capacity, integer feasible", "schedule_capacity_integer"),
    )
    if snapshot is None:
        lines.append("- NOT_AVAILABLE: pilot parameter snapshot has not been generated.")
    else:
        lines.extend(["| Parameter | Value |", "|---|---:|"])
        lines.extend(
            f"| {label} | {display(snapshot.get(key))} |"
            for label, key in parameter_fields
        )

    lines.extend(
        [
            "",
            "## Per-Angle B Calibration",
            "",
            "The q75 rule is an empirical implementation choice, not a value specified by the paper.",
            "Each identifiable B value and reward range is frozen before its matching formal run.",
            "",
        ]
    )
    if not b_rows:
        lines.append("- NOT_AVAILABLE: angle-specific pilot artifacts have not been generated.")
    else:
        lines.extend(
            [
                "| phi (deg) | B* | q50 | q75 | q90 | q95 | max | samples | status |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in sorted(b_rows, key=lambda item: int(float(item["phi_theta_deg"]))):
            lines.append(
                "| {phi} | {b} | {q50} | {q75} | {q90} | {q95} | {maximum} | {count} | {status} |".format(
                    phi=display(row.get("phi_theta_deg")),
                    b=display(row.get("B_penalty_star")),
                    q50=display(row.get("ratio_q50")),
                    q75=display(row.get("ratio_q75")),
                    q90=display(row.get("ratio_q90")),
                    q95=display(row.get("ratio_q95")),
                    maximum=display(row.get("ratio_max")),
                    count=display(
                        row.get("ratio_sample_count", row.get("B_ratio_sample_count"))
                    ),
                    status=display(row.get("status", row.get("B_status"))),
                )
            )

    mira_rows = sorted(
        (row for row in rows if row.get("policy") == "mira_sw_ucb"),
        key=lambda item: int(float(item["phi_theta_deg"])),
    )
    lines.extend(
        [
            "",
            "## Fixed-Seed MIRA Results",
            "",
        ]
    )
    if not mira_rows:
        lines.append("- NOT_AVAILABLE: no formal MIRA phi runs were found.")
    else:
        lines.extend(
            [
                "| phi | status | slots | post-warmup AoI (ms) | p50 | p95 | p99 | successful/scheduled groups | edge groups | mean interference | regroup events/device/min |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in mira_rows:
            group_ratio = "{}/{}".format(
                display(row.get("successful_group_count")),
                display(row.get("scheduled_group_count")),
            )
            lines.append(
                "| {phi} | {status} | {slots} | {aoi} | {p50} | {p95} | {p99} | {groups} | {edge} | {interference} | {regroup} |".format(
                    phi=display(row.get("phi_theta_deg")),
                    status=display(row.get("status")),
                    slots=display(row.get("slots_completed")),
                    aoi=display(row.get("weighted_mean_aoi_after_warmup_ms")),
                    p50=display(row.get("weighted_sum_aoi_p50_ms")),
                    p95=display(row.get("weighted_sum_aoi_p95_ms")),
                    p99=display(row.get("weighted_sum_aoi_p99_ms")),
                    groups=group_ratio,
                    edge=display(row.get("edge_group_count")),
                    interference=display(row.get("mean_interference_power")),
                    regroup=display(row.get("regrouping_events_per_device_per_minute")),
                )
            )

    lines.extend(["", "## Rate, SIC, and Control Diagnostics", ""])
    if not mira_rows:
        lines.append("- NOT_AVAILABLE: no formal MIRA diagnostics were found.")
    else:
        lines.extend(
            [
                "| phi | group success ratio | mean R | p05 R | p50 R | p95 R | rate failures | SIC failures | p95 interference | action histogram |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in mira_rows:
            lines.append(
                "| {phi} | {ratio} | {mean_r} | {p05} | {p50} | {p95} | {rate_fail} | {sic_fail} | {interference} | {actions} |".format(
                    phi=display(row.get("phi_theta_deg")),
                    ratio=display(row.get("conditional_group_success_ratio")),
                    mean_r=display(row.get("mean_scheduled_R_bps_hz")),
                    p05=display(row.get("p05_scheduled_R_bps_hz")),
                    p50=display(row.get("p50_scheduled_R_bps_hz")),
                    p95=display(row.get("p95_scheduled_R_bps_hz")),
                    rate_fail=display(row.get("rate_threshold_failure_count")),
                    sic_fail=display(row.get("sic_failure_count")),
                    interference=display(row.get("p95_interference_power")),
                    actions=display(row.get("action_histogram")),
                )
            )

    lines.extend(
        [
            "",
            "## Selection",
            "",
        ]
    )
    if selected is None:
        lines.extend(
            [
                "- Selected phi: NOT_AVAILABLE_INCOMPLETE_FORMAL_RUNS",
                "- No selection is reported until every configured full-length MIRA run is present.",
            ]
        )
    else:
        lines.extend(
            [
                f"- Selected phi: {selected['selected_phi_theta_deg']} deg",
                f"- Selection metric: {selected['selection_metric']}",
                f"- Selection reason: {selected.get('selection_reason', 'lowest metric')}",
                f"- Frozen B*: {selected['B_penalty_star_for_selected_phi']}",
                f"- Tie note: {display(selected.get('tie_note'))}",
            ]
        )

    lines.extend(["", "## Baseline Status", ""])
    if not baselines:
        lines.append("- NOT_AVAILABLE: no baseline status records were found.")
    else:
        by_policy: dict[str, set[str]] = {}
        applicability: dict[str, set[str]] = {}
        for row in baselines:
            name = row.get("policy_label", row.get("policy", "UNKNOWN"))
            by_policy.setdefault(name, set()).add(row.get("status", "UNKNOWN"))
            applicability.setdefault(name, set()).add(
                row.get("phi_theta_applicable", "UNKNOWN")
            )
        lines.extend(
            [
                "| Policy | Status | phi_theta applicable |",
                "|---|---|---|",
            ]
        )
        for name in sorted(by_policy):
            lines.append(
                f"| {name} | {', '.join(sorted(by_policy[name]))} | "
                f"{', '.join(sorted(applicability[name]))} |"
            )

    lines.extend(["", "## Figures", ""])
    if figures:
        lines.extend(f"- `{path}`" for path in figures)
    else:
        lines.append("- NOT_AVAILABLE: no complete slot traces were found.")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Results, when present, describe one fixed random realization of this synthetic scenario only.",
            "- No cross-seed uncertainty, confidence interval, p-value, or global-optimality claim is made.",
            "- AIM-static, PPO-multicast, and GA-grouping remain blocked unless faithful source definitions are supplied.",
            "- Capacity goodput is diagnostic; deadline goodput is not evaluated because no deadline range is specified.",
            "",
            "## Artifact Completion",
            "",
            f"- Completed summary rows: {len(rows)}",
            f"- Baseline status rows: {len(baselines)}",
            f"- Generated figures: {len(figures)}",
        ]
    )
    report_path = report_dir / "PHI_THETA_STUDY_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "report": str(report_path),
        "selected_phi_available": selected is not None,
        "generated_figures": figures,
    }
