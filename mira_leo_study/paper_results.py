from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from .algorithms import RewardNormalizer
from .config import FIXED_PHI_THETA_DEG, FIXED_B_PENALTY, StudyConfig, save_config, validate_fixed_study_config
from .errors import ModelConsistencyError
from .metrics import running_mean
from .reward_pilot import apply_reward_bounds, reward_context, run_reward_bounds_pilot
from .simulation import CHECKPOINT_SCHEMA_VERSION, build_parameter_snapshot, run_simulation
from .storage import atomic_write_json, write_csv


POLICIES = ("mira_sw_ucb", "random_grouping", "fixed_grouping")
LABELS = {
    "mira_sw_ucb": "MIRA-LEO (SW-UCB)",
    "random_grouping": "Random grouping",
    "fixed_grouping": "Fixed grouping (a=4)",
}
TRACE_FILES = ("mira_leo_slot_metrics.csv", "random_grouping_slot_metrics.csv", "fixed_a4_slot_metrics.csv")
ENVIRONMENT_KEYS = ("trajectory_hash", "device_layout_hash", "subscription_hash", "channel_trace_hash", "environment_trace_hash")


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_trace(rows: list[dict], summary: dict) -> dict:
    values = np.asarray([float(r["weighted_sum_aoi_total"]) for r in rows])
    if not len(values):
        raise ValueError("comparison requires a complete nonempty AoI trace")
    expected = running_mean(values)
    actual = np.asarray([float(r["running_mean_weighted_sum_aoi_total"]) for r in rows])
    if not np.allclose(actual, expected, rtol=1e-12, atol=1e-10):
        raise ValueError("long-run AoI does not match the cumulative per-slot mean")
    tail = values[-max(1, math.ceil(len(values) * 0.2)):]
    return {
        "policy": summary["policy"], "label": LABELS[summary["policy"]],
        "status": summary["status"], "aoi_unit": "slots",
        "final_running_mean_aoi": float(expected[-1]),
        "mean_aoi_all_slots": float(np.mean(values)),
        "median_aoi_all_slots": float(np.median(values)),
        "mean_aoi_last_20pct_slots": float(np.mean(tail)),
        "std_aoi_last_20pct_slots": float(np.std(tail)),
        "mean_scheduled_groups": float(np.mean([int(r["scheduled_groups"]) for r in rows])),
        "mean_successful_groups": float(np.mean([int(r["successful_groups"]) for r in rows])),
        "invalid_power_or_sic_count": summary["invalid_power_or_sic_count"],
        "post_schedule_power_drop_count": summary["post_schedule_power_drop_count"],
        "max_transmitted_power_normalized": summary["max_transmitted_power_normalized"],
        "sic_projection_count": summary["sic_projection_count"],
        "power_infeasible_cohort_slots": summary["power_infeasible_cohort_slots"],
        "power_outage_slots": summary["power_outage_slots"],
        "seed": summary["channel_seed"], "slots": len(values),
    }


def check_common_environment(summaries: list[dict]) -> dict:
    identities = {}
    for key in ENVIRONMENT_KEYS:
        values = {s.get(key) for s in summaries}
        if len(values) != 1 or not next(iter(values)):
            raise RuntimeError(f"policies did not share the same {key}")
        identities[key] = next(iter(values))
    if len({s["slots_completed"] for s in summaries}) != 1:
        raise RuntimeError("policies must use the same horizon")
    return identities


def _validate_run(result: dict, rows: list[dict], cohort_rows: list[dict], slots: int, config: StudyConfig) -> None:
    if result["status"] not in {"COMPLETED", "COMPLETED_WITH_POWER_OUTAGES", "REWARD_RANGE_MISMATCH"}:
        raise RuntimeError(f"run did not complete validly: {result['status']}")
    if result["invalid_power_or_sic_count"] or result["post_schedule_power_drop_count"]:
        raise RuntimeError("scheduler/allocator consistency violation; results cannot be reported")
    if result["max_transmitted_power_normalized"] > 1.0 + 1e-10:
        raise RuntimeError("satellite normalized power exceeded one")
    if len(rows) != slots or [int(r["slot"]) for r in rows] != list(range(slots)):
        raise RuntimeError("slot trace has missing or duplicate slots")
    if result["phi_theta_deg"] != FIXED_PHI_THETA_DEG or result["B_penalty"] != FIXED_B_PENALTY:
        raise RuntimeError("fixed phi/B changed")
    legal = set(config.get("algorithm.action_set"))
    interval = int(config.get("derived.regrouping_interval_slots"))
    cap = int(config.get("derived.algorithm2_U"))
    held = {}
    cohort_slots = {}
    projections = 0
    outage_count = 0
    outage_slots = set()
    for row in cohort_rows:
        action = int(row["action"])
        slot = int(row["slot"])
        cohort = int(row["cohort_id"])
        epoch = slot // interval
        boundary = slot % interval == 0
        cohort_slots.setdefault(cohort, []).append(slot)
        if action not in legal or (result["policy"] == "fixed_grouping" and action != 4):
            raise RuntimeError("invalid grouping action")
        if float(row["transmitted_power_normalized"]) > 1.0 + 1e-10:
            raise RuntimeError("cohort power exceeded one")
        if int(row["regrouping_epoch"]) != epoch or int(row["is_regrouping_slot"]) != int(boundary):
            raise RuntimeError("regrouping epoch/boundary mismatch")
        identity = (action, row["membership_hash"], row["action_owner_satellite_id"])
        key = (cohort, epoch)
        if key in held and identity != held[key]:
            raise RuntimeError("action, partition or action owner changed inside an epoch")
        held[key] = identity
        if not boundary and (int(row["regrouping_action_change"]) or int(row["device_regrouping_events"])):
            raise RuntimeError("off-boundary regrouping event")
        if int(row["csi_slot_index"]) != slot or int(row["power_csi_slot_index"]) != slot:
            raise RuntimeError("power must use the current slot CSI")
        if row["scheduler_algorithm"] != "Algorithm2_strict" or row["power_allocator"] != "twc_aim_staircase":
            raise RuntimeError("unexpected formal scheduler or power allocator")
        outage = bool(int(row["transmission_failure_power"]))
        if int(row["candidate_groups"]) > 30 or int(row["post_schedule_power_drop_count"]) or int(row["transmitted_sic_violation_count"]):
            raise RuntimeError("compact group count or power consistency violation")
        if outage:
            outage_count += 1
            outage_slots.add(slot)
            if (int(row["power_plan_valid"]) or int(row["transmission_attempted"])
                    or int(row["transmitted_groups"]) or int(row["successful_groups"])
                    or abs(float(row["transmitted_power_normalized"])) > 1e-12
                    or int(row["power_solver_status"]) != 2):
                raise RuntimeError("infeasible power must mean no transmission, no success and zero actual power")
        elif not int(row["power_plan_valid"]) or int(row["sic_violation_after_count"]):
            raise RuntimeError("unhandled invalid power plan")
        edge = int(row["candidate_edge_groups"])
        expected_edge = min(edge, cap)
        expected_nonedge = 0 if edge > cap else min(int(row["candidate_nonedge_positive_priority"]), cap - edge)
        if int(row["scheduled_edge_groups"]) != expected_edge or int(row["scheduled_nonedge_groups"]) != expected_nonedge:
            raise RuntimeError("scheduled set does not obey strict Algorithm 2")
        projections += int(row["sic_projection_used"])
    if len(cohort_slots) != int(config.get("services.cohort_count")) or any(
        sorted(sequence) != list(range(slots)) for sequence in cohort_slots.values()
    ):
        raise RuntimeError("cohort trace has missing/duplicate slots")
    if projections != int(result["sic_projection_count"]):
        raise RuntimeError("SIC projection counter mismatch")
    if outage_count != result["power_infeasible_cohort_slots"] or len(outage_slots) != result["power_outage_slots"]:
        raise RuntimeError("power outage counter mismatch")
    decisions = len(cohort_slots) * math.ceil(slots / interval)
    if sum(result["action_decision_count"].values()) != decisions:
        raise RuntimeError("action decision count is not once per epoch")
    expected_observations = decisions if result["policy"] == "mira_sw_ucb" else 0
    if result["bandit_observation_count"] != expected_observations:
        raise RuntimeError("bandit observation count is not once per epoch")
    for row in rows:
        if int(row["successful_groups"]) > int(row["scheduled_groups"]):
            raise RuntimeError("successful group count exceeded actual admissions")


def plot_convergence(traces: dict[str, list[dict]], directory: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for field, ylabel, filename, pdf in (
        ("running_mean_weighted_sum_aoi_total", "Long-run average weighted-sum AoI (slots)", "convergence_long_run_aoi", True),
        ("weighted_sum_aoi_total", "Per-slot weighted-sum AoI (slots)", "weighted_sum_aoi_raw_trace", False),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for policy in POLICIES:
            rows = traces[policy]
            ax.plot([int(r["slot"]) for r in rows], [float(r[field]) for r in rows], label=LABELS[policy], linewidth=1.25)
        ax.set(xlabel="Simulation slot", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / f"{filename}.png", dpi=200)
        if pdf:
            fig.savefig(directory / f"{filename}.pdf")
        plt.close(fig)


def plot_scheduler_diagnostics(cohort_rows: list[dict], directory: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fields = ("candidate_groups", "scheduled_groups", "rejected_by_algorithm2_cap",
              "successful_groups", "scheduled_edge_groups", "scheduled_nonedge_groups")
    by_slot = {}
    for row in cohort_rows:
        counts = by_slot.setdefault(int(row["slot"]), dict.fromkeys(fields, 0))
        for field in fields:
            counts[field] += int(row[field])
    slots = sorted(by_slot)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for ax, selected in zip(axes, (fields[:3], fields[3:])):
        for field in selected:
            ax.plot(slots, [by_slot[s][field] for s in slots], label=field.replace("_", " "), linewidth=1)
        ax.set_ylabel("Group count")
        ax.legend()
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Simulation slot")
    fig.tight_layout()
    fig.savefig(directory / "candidate_vs_scheduled_groups.png", dpi=200)
    plt.close(fig)


def run_paper_results(
    config: StudyConfig, output_directory, *, slots: int | None = None,
    seed: int | None = None, reward_bounds_path=None, resume: bool = False,
    tau_s: list[float] | None = None, save_group_trace: bool = False,
) -> dict:
    """Run grouping-only comparisons on a shared indexed environment realization."""
    validate_fixed_study_config(config)
    seed = int(config.get("randomness.fixed_channel_seed", 0) if seed is None else seed)
    config = config.with_overrides({"randomness.fixed_channel_seed": seed})
    requested = config.slot_count if slots is None else int(slots)
    if not 1 <= requested <= config.slot_count:
        raise ValueError("slots must be between one and the configured horizon")
    points = list(dict.fromkeys([10.0] + [float(t) for t in (
        config.get("formal_results.regrouping_overhead_tau_s", [10.0]) if tau_s is None else tau_s
    )]))
    if any(not math.isfinite(t) or t < float(config.get("experiment.slot_dt_s")) for t in points):
        raise ValueError("tau must be finite and at least one slot")
    dt = float(config.get("experiment.slot_dt_s"))
    if any(abs(t / dt - round(t / dt)) > 1e-9 for t in points):
        raise ValueError("tau must be an integer multiple of slot_dt_s")
    regroup_dt = float(config.get("algorithm.regrouping_interval_s"))
    if any(t < regroup_dt or abs(t / regroup_dt - round(t / regroup_dt)) > 1e-9 for t in points):
        raise ValueError("tau must be an integer multiple of regrouping_interval_s")
    root = Path(output_directory)
    # Keep explicit truncated runs separate from the full paper artifacts.
    if requested != config.slot_count:
        root = root / "debug" / f"slots_{requested}_seed_{seed}"
    bounds = config.get("algorithm.reward_bounds", {})
    cached = Path(output_directory) / "reward_bounds" / "reward_bounds.json"
    if reward_bounds_path is not None:
        config = apply_reward_bounds(config, json.loads(Path(reward_bounds_path).read_text(encoding="utf-8")), seed)
    elif bounds.get("lower") is None or bounds.get("upper") is None or not bounds.get("identity"):
        if cached.exists():
            config = apply_reward_bounds(config, json.loads(cached.read_text(encoding="utf-8")), seed)
        else:
            config = run_reward_bounds_pilot(config, output_directory, seed=seed)["config"]
    elif bounds.get("context") != reward_context(config, seed):
        raise ValueError("frozen bounds have no matching current protocol context; rerun calibrate-reward-bounds for the 1s compact configuration")
    bounds = config.get("algorithm.reward_bounds")
    RewardNormalizer(float(bounds["lower"]), float(bounds["upper"]), str(bounds["identity"]))
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "summary.json", {
        "status": "RUNNING", "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "config_id": config.config_id, "seed": seed, "slots": requested,
    })
    save_config(config, root / "config.json")
    atomic_write_json(root / "parameter_snapshot.json", build_parameter_snapshot(config, "grouping_comparison", seed))
    convergence = root / "formal_results" / "convergence_long_run_aoi"
    overhead_dir = root / "formal_results" / "regrouping_overhead"
    convergence.mkdir(parents=True, exist_ok=True)
    overhead_dir.mkdir(parents=True, exist_ok=True)

    def run_one(run_config, policy, name):
        directory = root / "runs" / name
        summary = None
        if resume and (directory / "summary.json").exists() and all(
            (directory / filename).exists() for filename in ("slot_metrics.csv", "convergence_slot_metrics.csv")
        ):
            cached_summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            if (cached_summary.get("config_id") == run_config.config_id
                    and cached_summary.get("policy") == policy
                    and cached_summary.get("channel_seed") == seed
                    and cached_summary.get("slots_completed") == requested
                    and cached_summary.get("schema_version") == CHECKPOINT_SCHEMA_VERSION
                    and (not save_group_trace or (directory / "group_trace.csv").exists())):
                summary = cached_summary
        if summary is None:
            try:
                result = run_simulation(run_config, channel_seed=seed, policy=policy, fixed_action=4,
                                        max_slots=requested, run_directory=directory,
                                        resume=resume, save_group_trace=save_group_trace)
            except ModelConsistencyError as exc:
                atomic_write_json(root / "summary.json", {
                    "status": "MODEL_CONSISTENCY_ERROR", "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "policy": policy, "seed": seed, "slots_requested": requested,
                    "error": str(exc), "run_directory": str(directory),
                })
                (root / "report.md").write_text(
                    "# Model consistency failure\n\n" + str(exc) +
                    "\n\nNo completed comparison is reported. See the run's model_consistency_error.json.\n",
                    encoding="utf-8",
                )
                raise
            summary = result["summary"]
        rows = _read_csv(directory / "convergence_slot_metrics.csv")
        cohort_rows = _read_csv(directory / "slot_metrics.csv")
        _validate_run(summary, rows, cohort_rows, requested, run_config)
        expected_cohort_rows = requested * int(config.get("services.cohort_count"))
        if len(cohort_rows) != expected_cohort_rows:
            raise RuntimeError("incomplete cohort diagnostics")
        return summary, rows, cohort_rows

    summaries, comparison, traces = [], [], {}
    mira_cohorts = None
    for policy, filename in zip(POLICIES, TRACE_FILES):
        summary, rows, cohort_rows = run_one(config, policy, policy)
        summaries.append(summary)
        comparison.append(summarize_trace(rows, summary))
        traces[policy] = rows
        write_csv(convergence / filename, rows)
        if policy == "mira_sw_ucb":
            mira_cohorts = cohort_rows
    environment = check_common_environment(summaries)
    write_csv(convergence / "comparison_slot_metrics.csv", [row for policy in POLICIES for row in traces[policy]])
    write_csv(convergence / "comparison_summary.csv", comparison)
    atomic_write_json(convergence / "comparison_summary.json", comparison)
    plot_convergence(traces, convergence)
    plot_scheduler_diagnostics(mira_cohorts, convergence)
    overhead_rows, event_rows = [], []
    for tau in points:
        if tau == 10.0:
            summary, cohorts = summaries[0], mira_cohorts
        else:
            variant = config.with_overrides({"algorithm.sw_ucb_window_s": tau})
            validate_fixed_study_config(variant, allow_tau_sensitivity=True)
            summary, _, cohorts = run_one(variant, "mira_sw_ucb", f"mira_tau_{tau:g}")
            check_common_environment([summaries[0], summary])
        keys = ("total_regrouping_events", "eligible_device_service_count", "evaluated_duration_s",
                "evaluated_duration_min", "events_per_device_per_min", "action_change_events")
        overhead_rows.append({"policy": "mira_sw_ucb", "tau_s": tau, "status": summary["status"], **{k: summary[k] for k in keys}})
        for row in cohorts:
            event_rows.append({"policy": "mira_sw_ucb", "tau_s": tau, **{k: row[k] for k in (
                "slot", "time_s", "cohort_id", "satellite_id", "action", "regrouping_action_change",
                "device_regrouping_events", "initial_assignments", "handover_assignments",
                "regrouping_epoch", "is_regrouping_slot", "membership_hash",
            )}})
        counted = sum(int(r["device_regrouping_events"]) for r in cohorts)
        if counted != summary["total_regrouping_events"]:
            raise RuntimeError("overhead event ledger does not match the summary")
    fields = list(overhead_rows[0])
    for policy in ("aim_static", "ga_grouping"):
        overhead_rows.append(dict.fromkeys(fields) | {"policy": policy, "status": "NOT_IMPLEMENTED"})
    write_csv(overhead_dir / "regrouping_events.csv", event_rows)
    write_csv(overhead_dir / "regrouping_overhead_summary.csv", overhead_rows)
    table = ["| Policy | tau_s | Re-signalling events | Device-service instances | Duration_min | Events/device/min | Action changes | Status |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in overhead_rows:
        value = lambda key: "" if row.get(key) is None else str(row[key])
        table.append("| " + " | ".join(value(k) for k in ("policy", "tau_s", "total_regrouping_events", "eligible_device_service_count", "evaluated_duration_min", "events_per_device_per_min", "action_change_events", "status")) + " |")
    (overhead_dir / "regrouping_overhead_table.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    all_statuses = {s["status"] for s in summaries + [r for r in overhead_rows if r["policy"] == "mira_sw_ucb"]}
    overall_status = "REWARD_RANGE_MISMATCH" if "REWARD_RANGE_MISMATCH" in all_statuses else (
        "COMPLETED_WITH_POWER_OUTAGES" if "COMPLETED_WITH_POWER_OUTAGES" in all_statuses else "COMPLETED")
    result = {
        "status": overall_status,
        "phi_theta_deg": FIXED_PHI_THETA_DEG, "B_penalty": FIXED_B_PENALTY, "fixed_action": 4, "seed": seed,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "action_set_name": "compact", "action_set": list(config.get("algorithm.action_set")),
        "regrouping_interval_s": float(config.get("algorithm.regrouping_interval_s")),
        "sw_ucb_window_epochs": int(config.get("derived.sw_ucb_window_epochs")),
        "scheduler_algorithm": "Algorithm2_strict", "power_allocator": "twc_aim_staircase",
        "infeasible_power_handling": str(config.get("algorithm.infeasible_power_handling")),
        "policy_seed": config.get("randomness.policy_seed"), "slots": requested,
        "is_full_horizon": requested == config.slot_count,
        "environment_identity_equal": True, "environment": environment,
        "comparison": comparison, "regrouping_overhead": overhead_rows,
        "unavailable_baselines": {p: "NOT_IMPLEMENTED" for p in ("aim_static", "ga_grouping", "ppo_multicast")},
        "artifact_root": str(root), "convergence_directory": str(convergence),
        "overhead_directory": str(overhead_dir),
    }
    atomic_write_json(root / "summary.json", result)
    report = ["# Fixed-parameter MIRA-LEO paper results", "",
              f"phi_theta_deg={FIXED_PHI_THETA_DEG:.1f}; B_penalty={FIXED_B_PENALTY:g}; Fixed grouping a=4.",
              f"Horizon: {requested} slots; seed: {seed}; full horizon: {result['is_full_horizon']}.",
              "AoI is the constellation sum in slots; the running mean uses the entire horizon, including exploration.",
              "Per-satellite mean includes all eight satellites (inactive contributions are zero).",
              "All three policies use the same scheduler, power allocator, SIC, AoI, trajectory and indexed channel trace.",
              "Membership and actions are held for 1 second (100 physical slots); global regrouping epochs index SW-UCB, whose nominal 10-second window contains 10 epochs.",
              "Each physical slot uses current-slot error-free CSI to update worst-user gain, edge labels, strict Algorithm 2, TWC/AIM powers, global SIC and AoI.",
              "Algorithm 2 uses U=62, all edge groups when edge count <= U, otherwise top-U edge groups; only positive-benefit nonedge groups fill remaining capacity.",
              "The allocator uses fixed class pools, current-CSI staircases and continuous residuals. Cross-class SIC repair changes powers only. A proven infeasible LP mutes the satellite for that slot; all its scheduled updates fail and remain in AoI/reward statistics. The schedule is unchanged and actual power is zero.",
              "Power-outage counters are separate from channel decoding failures. invalid_power_or_sic_count counts unhandled/transmitted constraint violations, not explicitly recorded zero-power outages. Numerical/solver errors that do not prove infeasibility still stop the run.",
              "Bandit rewards average the per-slot normalized rewards in each epoch and update the agent that selected the action, including when a handover occurs mid-epoch.",
              "Physical slot t and bandit epoch e are distinct indices; the paper's slot-based regret notation requires this mapping for the two-timescale implementation.",
              "A logical G-RNTI includes service, group index and sorted membership. Only consecutive visible assignments on the same satellite count as regrouping.",
              "Initial acquisition, coverage exit and handover are excluded; the denominator is all eligible subscribed device-service instances times the full evaluated duration.",
              "Action-change counts and whole-message membership-change diagnostics are separate from device-level signalling events.",
              "Additional tau values are project sensitivity points. Other parameters and environment remain fixed.", ""]
    for row in comparison:
        report.append(f"- {row['label']}: mean AoI={row['mean_aoi_all_slots']:.8g}; last 20% mean={row['mean_aoi_last_20pct_slots']:.8g}; power-outage cohort-slots={row['power_infeasible_cohort_slots']}; status={row['status']}.")
    mira_mean = comparison[0]["mean_aoi_all_slots"]
    if any(mira_mean >= row["mean_aoi_all_slots"] for row in comparison[1:]):
        report.append("MIRA-LEO does not strictly outperform every baseline on full-horizon mean AoI; all measured values and exploration slots are retained. See the scheduler/power/action CSV diagnostics.")
    report.extend(["", "AIM-static, GA-grouping and PPO-multicast: NOT_IMPLEMENTED.", "", *table])
    (root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return result
