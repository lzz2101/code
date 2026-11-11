"""Artificial CSV packaging tests; both simulator and calibration are replaced.

All figures live in a temporary directory and are deleted after the test. They
are test artifacts, never experiment results. No simulation slot loop is run.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mira_leo_study.config import load_config
from mira_leo_study.geo_edge import geo_occupancy_row, snapshot_group_rows
from mira_leo_study.models import Group
from mira_leo_study.p1_ablation import (
    ENVIRONMENT_KEYS, _csv_summary, experiment_matrix, run_p1_ablation,
)
from mira_leo_study.p1_metrics import objective_summary
from mira_leo_study.protected_power import POWER_MODE, POOL_RULE_VERSION
from mira_leo_study.reward_pilot import apply_reward_bounds, reward_context
from mira_leo_study.scenario import StudyScenario
from mira_leo_study.simulation import CHECKPOINT_SCHEMA_VERSION
from mira_leo_study.storage import atomic_write_json, write_csv

CONFIG = Path(__file__).parents[1] / "configs/base.yaml"


def artificial_pilot(config, directory, *, seed):
    payload = dict(phi_theta_deg=config.get("algorithm.phi_theta_deg"), B_penalty=.05,
                   reward_context=reward_context(config, seed), reward_identity="ARTIFICIAL-test-only",
                   reward_lower=-1e6, reward_upper=1e6)
    atomic_write_json(Path(directory) / "reward_bounds/reward_bounds.json", payload)
    return {"config": apply_reward_bounds(config, payload, seed)}


def artificial_csv_run(config, *, policy, channel_seed, run_directory, max_slots,
                       save_geo_edge_trace, **kwargs):
    """Write a deliberately simple finite trace without invoking model dynamics."""
    directory = Path(run_directory)
    phi = config.get("algorithm.phi_theta_deg")
    action = 4 if policy == "fixed_grouping" else 1
    groups = 3 * action
    rows, convergence = [], []
    for slot in range(max_slots):
        delta = 40. + slot
        p1 = groups - delta
        row = dict(slot=slot, time_s=slot*.01, policy=policy, successful_groups=groups,
                   weighted_aoi_slots=delta, P1_slot_value=p1, raw_reward=p1, edge_groups=0,
                   transmission_failure_power=0, protected_scheduled_count=0,
                   ordinary_scheduled_count=groups, promoted_count=0,
                   protected_power_actual=0., ordinary_power_actual=1.,
                   power_allocator=POWER_MODE, post_schedule_power_drop_count=0,
                   transmitted_sic_violation_count=0, action=action, regrouping_epoch=slot//100,
                   is_regrouping_slot=int(slot%100 == 0), membership_hash=f"ARTIFICIAL-{slot//100}",
                   action_owner_satellite_id="sat_01", regrouping_action_change=0,
                   device_regrouping_events=0, csi_slot_index=slot, power_csi_slot_index=slot,
                   scheduled_groups=groups, geo_edge_scheduled_count=0, candidate_edge_groups=0,
                   scheduled_edge_groups=0, scheduled_nonedge_groups=groups,
                   candidate_nonedge_positive_priority=groups, transmitted_power_normalized=1.,
                   power_plan_valid=1, transmission_attempted=1, transmitted_groups=groups,
                   candidate_groups=groups, sic_projection_used=0, power_failure_reason="")
        rows.append(row)
        convergence.append(dict(slot=slot, weighted_sum_aoi_total=delta,
                                P1_slot_value=p1, successful_groups=groups))
    write_csv(directory / "slot_metrics.csv", rows)
    write_csv(directory / "convergence_slot_metrics.csv", convergence)
    warm = rows[1000:]
    summary = dict(
        policy=policy, status="COMPLETED", channel_seed=channel_seed, config_id=config.config_id,
        phi_theta_deg=phi, slots_completed=max_slots, power_allocator=POWER_MODE,
        power_pool_rule_version=POOL_RULE_VERSION, schema_version=CHECKPOINT_SCHEMA_VERSION,
        **objective_summary(sum_c=groups*max_slots, sum_delta=sum(r["weighted_aoi_slots"] for r in rows),
                            slots=max_slots, warm_c=groups*len(warm),
                            warm_delta=sum(r["weighted_aoi_slots"] for r in warm), warm_slots=len(warm),
                            warmup_s=10., slot_dt_s=.01),
        power_outage_slots=0, sic_projection_count=0, transmitted_sic_violation_count=0,
        sum_power_excess=0., action_decision_count={"sat_01:a="+str(action): math.ceil(max_slots/100)},
        bandit_observation_count=math.ceil(max_slots/100) if policy == "mira_sw_ucb" else 0,
        weighted_mean_aoi_after_warmup_ms=np.mean([r["weighted_aoi_slots"] for r in warm])*10. if warm else None,
        weighted_mean_aoi_per_weight_unit_after_warmup_ms=np.mean([r["weighted_aoi_slots"] for r in warm])*.25 if warm else None,
        successful_group_count=groups*max_slots, scheduled_group_count=groups*max_slots,
        transmitted_group_count=groups*max_slots, conditional_group_success_ratio=1., capacity_goodput_bps=100.,
        geo_edge_candidate_count=0, geo_edge_scheduled_count=0, promoted_weak_nonedge_count=0,
        protected_count=0, ordinary_count=groups*max_slots, protected_pool_fraction_realized=0.,
        power_infeasible_serving_slots=0, power_outage_rate=0., floor_infeasible_slots=0,
        cross_pool_sic_infeasible_slots=0, lp_numeric_failure_slots=0, rate_threshold_failure_count=0,
        sic_failure_count=0, propagated_sic_failure_count=0, regrouping_events_total=0,
        action_histogram={str(action): max_slots}, handover_count=0, route_unreachable_slots=0,
        reward_bounds_identity="ARTIFICIAL-test-only", reward_clipping_rate_max=0.,
    )
    summary.update({key: "ARTIFICIAL-shared-identity" for key in ENVIRONMENT_KEYS if key != "slots_completed"})
    atomic_write_json(directory / "summary.json", summary)
    if save_geo_edge_trace:
        scenario = StudyScenario(config)
        sample_groups = [Group(0, m, 0, scenario.subscriptions.message_members[m],
                               1., 50., False, scheduled=True) for m in range(3)]
        ecef = np.tile([7000., 0., 0.], (8, 1))
        snapshots, members = [], []
        for slot in range(0, max_slots, 100):
            a, b = snapshot_group_rows(config, scenario, sample_groups, slot=slot,
                                      elevation=np.full((8, 120), 50.),
                                      amplitudes=np.ones((8, 120)), satellite_ecef=ecef)
            snapshots.extend(a)
            members.extend(b)
        geo = directory / "geo_edge"
        write_csv(geo / "edge_group_snapshots.csv", snapshots)
        write_csv(geo / "edge_member_positions.csv", members)
        write_csv(geo / "geo_edge_time_distribution.csv", [
            geo_occupancy_row(sample_groups, slot=i, dt_s=.01, satellite_id="sat_01", handover=False)
            for i in range(max_slots)])
    return {"summary": summary}


class P1PackagingTests(unittest.TestCase):
    def test_six_run_csv_figures_geo_and_resume_without_simulation(self):
        config = load_config(CONFIG)
        self.assertEqual(config.slot_count, 48000)
        with tempfile.TemporaryDirectory(prefix="ARTIFICIAL_p1_") as temp:
            with patch("mira_leo_study.p1_ablation.run_reward_bounds_pilot", side_effect=artificial_pilot) as pilot, \
                 patch("mira_leo_study.p1_ablation.run_simulation", side_effect=artificial_csv_run) as simulator, \
                 patch("mira_leo_study.simulation.serving_channel_amplitude", side_effect=AssertionError("simulation forbidden")):
                result = run_p1_ablation(config, temp, debug_slots=1001)
            self.assertEqual(pilot.call_count, 4)
            self.assertEqual(simulator.call_count, 6)
            self.assertEqual([call.kwargs["policy"] for call in simulator.call_args_list],
                             [r["policy"] for r in experiment_matrix()])
            self.assertEqual([call.args[0].get("algorithm.phi_theta_deg") for call in simulator.call_args_list],
                             [13., 15., 20., 25., 15., 15.])
            self.assertEqual(sum(call.kwargs["save_geo_edge_trace"] for call in simulator.call_args_list), 1)
            self.assertFalse(result["formal_horizon"])
            root = Path(result["artifact_root"])
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["runs"]), 6)
            self.assertEqual(manifest["status"], "COMPLETED")
            self.assertEqual(len(manifest["figures"]), 19)
            for figure in manifest["figures"].values():
                self.assertTrue(Path(figure["path"]).is_file())
                self.assertTrue(Path(figure["source_csv"]).is_file())
            self.assertEqual(result["geo_edge_phi15"]["snapshot_slots"], 11)
            self.assertEqual(result["geo_edge_phi15"]["slots_with_geo_edge"], 0)
            with (root / "comparison/all_six_runs.csv").open(newline="") as handle:
                records = list(csv.DictReader(handle))
            self.assertEqual(len(records), 6)
            for row in records:
                self.assertAlmostEqual(float(row["sum_P1"]), float(row["sum_C"])-float(row["sum_weighted_delta_slots"]))
            before = (root / "runs/mira_phi15/seed_0/slot_metrics.csv").read_bytes()
            # A partial summary must dispatch to the simulator/checkpoint path.
            checkpointed = root / "runs/mira_phi20/seed_0/summary.json"
            summary = json.loads(checkpointed.read_text(encoding="utf-8"))
            summary["status"] = "CHECKPOINTED"
            atomic_write_json(checkpointed, summary)
            with patch("mira_leo_study.p1_ablation.run_reward_bounds_pilot", side_effect=AssertionError("must load bounds")), \
                 patch("mira_leo_study.p1_ablation.run_simulation", side_effect=artificial_csv_run) as simulator, \
                 patch("mira_leo_study.p1_ablation._comparison_artifacts", side_effect=lambda root, rows, arrays: ([_csv_summary(r) for r in rows], {})), \
                 patch("mira_leo_study.p1_ablation.analyze_geo_edge", return_value=result["geo_edge_phi15"]):
                resumed = run_p1_ablation(config, temp, debug_slots=1001, resume=True)
            self.assertEqual(simulator.call_count, 1)
            self.assertEqual(simulator.call_args.args[0].get("algorithm.phi_theta_deg"), 20.)
            self.assertTrue(simulator.call_args.kwargs["resume"])
            self.assertEqual(resumed["comparison"], result["comparison"])
            self.assertEqual(before, (root / "runs/mira_phi15/seed_0/slot_metrics.csv").read_bytes())


if __name__ == "__main__":
    unittest.main()
