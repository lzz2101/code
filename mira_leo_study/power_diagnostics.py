"""Power-only audit of the pilot snapshot grid; no AoI or learning simulation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .algorithms import allocate_power, evaluate_twc_aim_power_plan, make_groups
from .association import AssociationTracker
from .channel import serving_channel_amplitude
from .config import load_config
from .geometry import link_geometry
from .scenario import StudyScenario
from .storage import atomic_write_json, write_csv


def scan_pilot_power(config, output_directory, *, seed=0):
    """Mark all candidates scheduled to identify structural power conflicts.

    This deliberately does not claim to count actual Algorithm-2 failures:
    a real schedule can omit nonpositive-priority nonedge groups.
    """
    scenario = StudyScenario(config)
    tracker = AssociationTracker(config, scenario.subscriptions)
    dt = float(config.get("experiment.slot_dt_s"))
    stride = round(float(config.get("formal_results.reward_pilot_snapshot_interval_s")) / dt)
    snapshots = set()
    for start, end in config.get("formal_results.reward_pilot_windows_s"):
        snapshots.update(range(round(start / dt), round(end / dt), stride))
    rows, details = [], []
    for slot in range(config.slot_count):
        _, ecef = scenario.constellation.state(slot * dt)
        distance, elevation = link_geometry(ecef, scenario.devices.ecef_km)
        association = tracker.update(slot, elevation)
        if slot not in snapshots:
            continue
        channel = serving_channel_amplitude(config, seed, slot, distance)
        for action in config.get("algorithm.action_set"):
            for cohort, satellite in enumerate(association.serving_satellites):
                groups = make_groups(config, satellite, cohort, action,
                                     scenario.subscriptions.message_members, channel, elevation)
                for group in groups:
                    group.scheduled = True
                diagnostic = evaluate_twc_aim_power_plan(config, groups)
                infeasible = diagnostic["projection_solver_status"] == 2
                outcome = allocate_power(config, groups) if not diagnostic["valid"] else None
                row = {
                    "slot": slot, "time_s": slot * dt, "seed": seed,
                    "phi_theta_deg": float(config.get("algorithm.phi_theta_deg")),
                    "cohort_id": cohort, "satellite_id": scenario.constellation.elements[satellite].satellite_id,
                    "action": action, "assumption": "ALL_CANDIDATE_GROUPS_SCHEDULED",
                    "candidate_groups": len(groups), "edge_count": diagnostic["edge_count"],
                    "nonedge_count": diagnostic["nonedge_count"],
                    "edge_floor_total": diagnostic["edge_floor_total"],
                    "nonedge_floor_total": diagnostic["nonedge_floor_total"],
                    "valid_power_plan": diagnostic["valid"], "lp_infeasible": infeasible,
                    "lp_status": diagnostic["projection_solver_status"],
                    "handled_as_transmission_failure": bool(outcome and outcome["transmission_failure"]),
                    "actual_power_if_failed": outcome["actual_total_power"] if outcome else None,
                }
                rows.append(row)
                if not diagnostic["valid"]:
                    details.append({**row, "solver_message": diagnostic["projection_solver_message"],
                                    "groups_by_current_csi": [
                                        {"message_id": g.message_id, "group_id": g.group_id,
                                         "gain": g.gain, "is_edge": g.is_edge}
                                        for g in sorted(groups, key=lambda g: (g.gain, g.cohort_id, g.message_id, g.group_id))]})
        if slot in {max(s for s in snapshots if s < round(end / dt)) for _, end in config.get("formal_results.reward_pilot_windows_s")}:
            print(f"Completed diagnostic window through t={slot * dt:g}s", flush=True)
    failed = [r for r in rows if r["lp_infeasible"]]
    summary = {
        "scope": "Pilot snapshot grid, all candidates scheduled, power-only diagnostic",
        "not_an_actual_policy_failure_count": True,
        "caveat": "Actual Algorithm 2 can omit nonpositive-priority nonedge groups; this is not a 480-second per-slot outage frequency.",
        "no_aoi_or_learning_or_reward_pilot_executed": True,
        "association": "Recomputed at every current physical slot using hysteresis/TTT",
        "phi_theta_deg": float(config.get("algorithm.phi_theta_deg")), "seed": seed,
        "snapshot_count": len(snapshots), "total_power_configurations": len(rows),
        "lp_infeasible_count": len(failed),
        "other_invalid_count": sum(not r["valid_power_plan"] and not r["lp_infeasible"] for r in rows),
        "valid_count": sum(r["valid_power_plan"] for r in rows),
        "handled_transmission_failure_count": sum(r["handled_as_transmission_failure"] for r in rows),
        "infeasible_distinct_times_s": sorted({r["time_s"] for r in failed}),
        "infeasible_by_action": dict(sorted(Counter(r["action"] for r in failed).items())),
        "infeasible_by_cohort": dict(sorted(Counter(r["cohort_id"] for r in failed).items())),
    }
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    write_csv(root / "power_cases.csv", rows)
    atomic_write_json(root / "infeasible_cases.json", details)
    atomic_write_json(root / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("configs") / "base.yaml"))
    parser.add_argument("--outputs", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.outputs or f"mira_leo_study/diagnostics/pilot_power_phi{float(config.get('algorithm.phi_theta_deg')):g}_seed{args.seed}_all_candidates"
    print(json.dumps(scan_pilot_power(config, output, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
