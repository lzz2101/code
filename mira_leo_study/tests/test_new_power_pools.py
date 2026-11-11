"""Artificial pool/LP/P1/geometry tests; no orbit/channel/AoI simulation loop."""
from pathlib import Path
import copy
import csv
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from mira_leo_study.algorithms import (allocate_power, evaluate_power_plan, classify_power_pools,
    materialize_groups_from_partition, build_group_partition, decode_groups)
from mira_leo_study.config import load_config, validate_fixed_study_config, validate_p1_ablation_config
from mira_leo_study.errors import ConfigurationError, ModelConsistencyError
from mira_leo_study.geo_edge import group_spatial_statistics, snapshot_group_rows
from mira_leo_study.models import Group
from mira_leo_study.p1_ablation import experiment_matrix, check_common_environment
from mira_leo_study.p1_metrics import p1_slot_value, objective_summary
from mira_leo_study.protected_power import POWER_MODE
from mira_leo_study.scenario import StudyScenario
from mira_leo_study.simulation import run_simulation, CHECKPOINT_SCHEMA_VERSION, validate_checkpoint_identity
from mira_leo_study.storage import write_csv, csv_to_parquet, recover_csv_before_slot, recover_trace_file
from mira_leo_study.diagnostic_trace import parquet_diagnostic_types

CONFIG=Path(__file__).parents[1]/"configs/base.yaml"


def group(index,gain,edge,scheduled=True):
    return Group(0,0,index,np.array([index]),gain,12. if edge else 45.,edge,scheduled=scheduled)


class NewPowerPoolTests(unittest.TestCase):
    def setUp(self):self.config=load_config(CONFIG)

    def assert_constraints(self,groups,diagnostic):
        self.assertTrue(diagnostic["valid"],diagnostic)
        ordered=sorted((g for g in groups if g.scheduled),key=lambda g:(g.gain,g.message_id,g.group_id))
        self.assertLessEqual(sum(g.power for g in ordered),1.+1e-10)
        for left,right in zip(ordered,ordered[1:]):self.assertGreaterEqual(left.power+1e-10,right.power)
        for g in ordered:self.assertGreaterEqual(g.power+1e-10,diagnostic["floors"][id(g)])
        for label in ("protected","ordinary"):
            self.assertAlmostEqual(sum(g.power for g in ordered if g.metadata["power_pool"]==label),diagnostic[label+"_budget"],places=10)

    def test_example_uses_strongest_geo_edge_reference(self):
        groups=[group(0,2,True),group(1,10,True),group(2,4,False),group(3,11,False)]
        pools=classify_power_pools(groups)
        self.assertEqual(pools["g_ref"],10)
        self.assertEqual([g.group_id for g in pools["promoted"]],[2])
        self.assertEqual({g.group_id for g in pools["protected"]},{0,1,2})
        self.assertFalse(groups[2].is_edge)

    def test_only_scheduled_geo_groups_set_reference(self):
        groups=[group(0,100,True,False),group(1,4,False),group(2,5,False)]
        result=allocate_power(self.config,groups)
        self.assertEqual(result["protected_count"],0)
        self.assertEqual(result["ordinary_budget"],1)
        self.assertIsNone(result["g_ref"])
        self.assertEqual(groups[0].power,0)

    def test_strict_equal_gain_is_not_promoted_and_ties_are_deterministic(self):
        groups=[group(2,2,False),group(1,2,True),group(0,1,False),group(3,3,False)]
        pools=classify_power_pools(groups)
        self.assertEqual([g.group_id for g in pools["ordered"]],[0,1,2,3])
        self.assertEqual([g.group_id for g in pools["promoted"]],[0])
        self.assertEqual([g.group_id for g in pools["ordinary"]],[2,3])

    def test_no_edge_and_protected_only_and_empty(self):
        for groups,protected,ordinary in (([group(0,1,False),group(1,2,False)],0.,1.),
                                          ([group(0,1,False),group(1,2,True)],1.,0.),([],0.,0.)):
            d=allocate_power(self.config,groups)
            self.assert_constraints(groups,d)
            self.assertEqual((d["protected_budget"],d["ordinary_budget"]),(protected,ordinary))
            self.assertEqual(d["actual_total_power"],1. if groups else 0.)

    def test_feasible_mixed_targets_respect_budgets_and_global_sic(self):
        groups=[group(0,1,True),group(1,2,False),group(2,3,False)]
        d=allocate_power(self.config,groups)
        self.assert_constraints(groups,d)
        self.assertAlmostEqual(groups[0].power,.7)
        self.assertAlmostEqual(groups[1].power+groups[2].power,.3)
        self.assertFalse(d["sic_projection_used"])

    def test_lp_repairs_boundary_without_changing_geographic_flags_or_schedule(self):
        groups=[group(i,i+1,i==6) for i in range(10)]
        identities=[(g.key,g.is_edge,g.scheduled) for g in groups]
        d=allocate_power(self.config,groups)
        self.assert_constraints(groups,d)
        self.assertTrue(d["sic_projection_used"])
        self.assertGreater(len(d["sic_order_violations_before"]),0)
        self.assertEqual(d["sic_order_violations_after"],[])
        self.assertEqual(d["promoted_weak_nonedge_count"],6)
        self.assertEqual(identities,[(g.key,g.is_edge,g.scheduled) for g in groups])

    def test_incompatible_pools_zero_actual_powers_and_no_transmission(self):
        groups=[group(i,i+1,i<3) for i in range(4)]
        d=allocate_power(self.config,groups)
        self.assertFalse(d["valid"])
        self.assertEqual(d["power_infeasible_reason"],"POOL_SIC_INCOMPATIBLE")
        self.assertTrue(d["transmission_failure"])
        self.assertEqual(d["actual_total_power"],0)
        self.assertTrue(all(g.scheduled and g.power==0 and not g.success for g in groups))

    def test_floor_failure_has_distinct_reason(self):
        config=self.config.with_overrides({"algorithm.V_phi":.6})
        d=allocate_power(config,[group(0,1,False),group(1,2,False)])
        self.assertEqual(d["power_infeasible_reason"],"FLOOR_EXCEEDS_POOL")
        self.assertEqual(d["actual_total_power"],0.)

    def test_numeric_solver_failure_is_zero_tx_with_distinct_reason(self):
        with patch("scipy.optimize.linprog",return_value=SimpleNamespace(success=False,status=4,message="artificial numeric error")):
            d=allocate_power(self.config,[group(i,i+1,i==6) for i in range(10)])
        self.assertEqual(d["power_infeasible_reason"],"LP_NUMERIC_FAILURE")
        self.assertTrue(d["transmission_failure"])
        self.assertEqual(d["actual_total_power"],0.)

    def test_solver_exception_and_nonfinite_solution_cannot_transmit(self):
        groups=[group(i,i+1,i==6) for i in range(10)]
        with patch("scipy.optimize.linprog",side_effect=RuntimeError("artificial solver failure")):
            d=allocate_power(self.config,groups)
        self.assertEqual(d["power_infeasible_reason"],"LP_NUMERIC_FAILURE")
        with patch("scipy.optimize.linprog",return_value=SimpleNamespace(success=True,status=0,message="artificial",x=np.full(20,np.nan))):
            d=allocate_power(self.config,groups)
        self.assertFalse(d["valid"])
        self.assertTrue(all(g.power==0 and not g.success for g in groups))

    def test_exact_geographic_definition_and_invalid_angles(self):
        members={m:np.arange(40*m,40*(m+1)) for m in range(3)}
        channel=np.ones((8,120))
        partition=build_group_partition(self.config,0,1,members,channel)
        for last,expected in ((14.9,True),(15.,False),(16.,False)):
            elevation=np.full((8,120),10.2)
            elevation[0,39]=last
            groups=materialize_groups_from_partition(self.config,0,partition,channel,elevation)
            self.assertEqual(groups[0].is_edge,expected)
        for phi in (0.,-1.,91.,float("nan"),float("inf")):
            with self.assertRaises(ConfigurationError):
                self.config.with_overrides({"algorithm.phi_theta_deg":phi})

    def test_invalid_inputs_fail_with_clear_codes(self):
        for gain in (0.,-1.,float("nan"),float("inf")):
            with self.assertRaisesRegex(ModelConsistencyError,"INVALID_GROUP_GAIN"):
                allocate_power(self.config,[group(0,gain,True)])
        with self.assertRaisesRegex(ModelConsistencyError,"DUPLICATE_GROUP_KEY"):
            allocate_power(self.config,[group(0,1,True),group(0,2,False)])
        g=group(0,1,True);g.member_indices=np.array([],dtype=int)
        with self.assertRaisesRegex(ModelConsistencyError,"EMPTY_GROUP"):allocate_power(self.config,[g])

    def test_read_only_evaluation_does_not_mutate_groups(self):
        groups=[group(0,1,True),group(1,2,False)];before=copy.deepcopy(groups)
        d=evaluate_power_plan(self.config,groups)
        self.assertEqual(d["mode"],POWER_MODE)
        for a,b in zip(groups,before):
            self.assertEqual(a.power,b.power);self.assertEqual(a.metadata,b.metadata)

    def test_all_four_angles_keep_coverage_cadence_and_raw_csi(self):
        members={m:np.arange(40*m,40*(m+1)) for m in range(3)}
        channel=np.ones((8,120))
        for phi in (13.,15.,20.,25.):
            c=self.config.with_overrides({"algorithm.phi_theta_deg":phi})
            validate_p1_ablation_config(c)
            self.assertEqual(c.get("geometry.phi_min_deg"),10.)
            partition=build_group_partition(c,0,1,members,channel)
            e=np.full((8,120),phi-.1)
            first=materialize_groups_from_partition(c,0,partition,channel,e)
            self.assertTrue(all(g.is_edge for g in first))
            e[0,0]=phi
            second=materialize_groups_from_partition(c,0,partition,channel*2,e)
            self.assertFalse(second[0].is_edge)
            np.testing.assert_array_equal(first[0].member_indices,second[0].member_indices)
            self.assertEqual(second[0].gain,4.*first[0].gain)
            for g in second:g.scheduled=True
            d=allocate_power(c,second)
            if d["valid"]:self.assert_constraints(second,d)
            else:self.assertEqual(d["actual_total_power"],0.)

    def test_formal_config_and_runner_reject_legacy_allocator(self):
        validate_fixed_study_config(self.config)
        self.assertEqual(self.config.get("algorithm.phi_theta_deg"),15.)
        old=self.config.with_overrides({"algorithm.power_mode":"twc_aim_staircase"})
        with self.assertRaises(ConfigurationError):validate_fixed_study_config(old)
        with self.assertRaisesRegex(ValueError,"unique"):
            run_simulation(old,channel_seed=0,max_slots=1)  # Rejected before any simulation.


class P1AndGeometryTests(unittest.TestCase):
    def test_nullable_new_fields_keep_types_across_parquet_batches(self):
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            source,target=Path(tmp)/"artificial.csv",Path(tmp)/"artificial.parquet"
            rows=[dict(slot=i,policy="ARTIFICIAL",members="x"*1500,
                       power_failure_reason="LP_NUMERIC_FAILURE" if i>=1000 else "",
                       power_pool_csi_reference_gain=1e-12 if i>=1000 else None,
                       epoch_reward=-40. if i>=1000 else None) for i in range(1200)]
            write_csv(source,rows)
            csv_to_parquet(source,target,chunk_size=1,column_types=parquet_diagnostic_types(list(rows[0])))
            table=pq.read_table(target)
            self.assertEqual(table.num_rows,1200)
            self.assertEqual(table["power_failure_reason"][-1].as_py(),"LP_NUMERIC_FAILURE")
            self.assertAlmostEqual(table["power_pool_csi_reference_gain"][-1].as_py(),1e-12)
            self.assertIsNone(table["power_pool_csi_reference_gain"][0].as_py())

    def test_p1_and_bandit_penalty_are_separate(self):
        self.assertEqual(p1_slot_value(3,40.),-37.)
        result=objective_summary(sum_c=6,sum_delta=80.,slots=2,warm_c=3,warm_delta=30.,warm_slots=1,warmup_s=.01,slot_dt_s=.01)
        self.assertEqual(result["sum_P1"],-74.)
        self.assertEqual(result["post_warmup_mean_P1"],-27.)

    def test_six_run_matrix_and_environment_identity(self):
        matrix=experiment_matrix()
        self.assertEqual(len(matrix),6)
        self.assertEqual({r["phi_theta_deg"] for r in matrix if r["policy"]=="mira_sw_ucb"},{13.,15.,20.,25.})
        self.assertEqual([r["phi_theta_deg"] for r in matrix if r["policy"]!="mira_sw_ucb"],[15.,15.])
        rows=[dict(r,layout_hash="a",subscription_hash="b",trajectory_hash="c",channel_trace_hash="d",environment_trace_hash="e",slots_completed=3) for r in matrix]
        self.assertEqual(check_common_environment(rows)["status"],"PASS")
        rows[-1]["layout_hash"]="different"
        with self.assertRaises(ValueError):check_common_environment(rows)

    def test_spatial_metrics_do_not_assume_clustering(self):
        single=group_spatial_statistics([0],[0]);self.assertIsNone(single["pairwise_mean_km"])
        spread=group_spatial_statistics([0,3,0],[0,0,4])
        self.assertAlmostEqual(spread["pairwise_mean_km"],4.)
        self.assertAlmostEqual(spread["pairwise_max_km"],5.)
        self.assertAlmostEqual(spread["convex_hull_area_km2"],6.)
        line=group_spatial_statistics([0,1,2],[0,0,0]);self.assertIsNone(line["convex_hull_area_km2"])

    def test_checkpoint_recovery_removes_only_uncommitted_slots(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"trace.csv"
            write_csv(path,[dict(slot=0,value=10),dict(slot=1,value=11),dict(slot=1,value=11),dict(slot=2,value=12)])
            recover_csv_before_slot(path,1)
            with path.open(newline="") as handle:rows=list(csv.DictReader(handle))
            self.assertEqual(rows,[dict(slot="0",value="10")])
            recover_csv_before_slot(path,1)
            with path.open(newline="") as handle:self.assertEqual(list(csv.DictReader(handle)),rows)

    def test_finalization_crash_recovers_same_committed_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            partial,final=Path(tmp)/"partial.csv",Path(tmp)/"final.csv"
            write_csv(final,[dict(slot=i,value=i*2) for i in range(3)])
            expected=final.read_bytes()
            recover_trace_file(partial,final,3)
            self.assertEqual(partial.read_bytes(),expected)
            self.assertEqual(final.read_bytes(),expected)
            with self.assertRaisesRegex(ValueError,"missing committed trace"):
                recover_trace_file(Path(tmp)/"missing.csv",Path(tmp)/"absent.csv",1)

    def test_checkpoint_schema_and_geo_trace_identity(self):
        common=dict(config_id="artificial",channel_seed=0,slot_limit=48000,
                    policy="mira_sw_ucb",fixed_action=4,save_group_trace=False)
        checkpoint=dict(common,schema_version=CHECKPOINT_SCHEMA_VERSION,save_geo_edge_trace=True)
        validate_checkpoint_identity(checkpoint,**common,save_geo_edge_trace=True)
        with self.assertRaises(ValueError):validate_checkpoint_identity(checkpoint,**common)
        checkpoint["schema_version"]=6
        with self.assertRaises(ValueError):validate_checkpoint_identity(checkpoint,**common,save_geo_edge_trace=True)


if __name__=="__main__":unittest.main()
