"""Artificial component/trace fixtures only; real simulation parity is opt-in."""
from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mira_leo_study.algorithms import decode_groups, schedule_groups
from mira_leo_study.config import load_config
from mira_leo_study.diagnostic_trace import (
    ALLOWED_OUTCOMES, classify_group_outcome, serving_power_diagnostics,
    enrich_group_diagnostic_rows, parquet_diagnostic_types,
)
from mira_leo_study.diagnostics import DiagnosticDataError, generate_diagnostics, ratio
from mira_leo_study.models import Group
from mira_leo_study.regrouping import groups_for_slot, initial_epoch_state
from mira_leo_study.reward_pilot import reward_context
from mira_leo_study.simulation import run_simulation, select_grouping_action
from mira_leo_study.storage import atomic_write_json, csv_to_parquet, write_csv
from mira_leo_study.tests import simulation_test

CONFIG=Path(__file__).parents[1]/'configs/base.yaml'


def group(gid=0, members=(0,1), message=0):
    return Group(0,message,gid,np.array(members,dtype=np.int32),.01,25.,False,
                 scheduled=True,priority=1.,power=.2,rate_bps_hz=.000008)


class OutcomeTests(unittest.TestCase):
    def test_exclusive_categories_and_precedence(self):
        g=group(); d={'transmission_attempted':True}
        cases=[]
        g.scheduled=False;g.priority=0
        cases.append(classify_group_outcome(g,d))
        g.is_edge=True;g.metadata['algorithm2_U']=1
        cases.append(classify_group_outcome(g,d,edge_count=2))
        self.assertEqual(classify_group_outcome(g,d,edge_count=1),'OTHER_FAILURE')
        g.is_edge=False;g.priority=1;cases.append(classify_group_outcome(g,d))
        g.scheduled=True
        cases.append(classify_group_outcome(g,{'transmission_failure':True,'transmission_attempted':True}))
        cases.append(classify_group_outcome(g,d))
        g.rate_pass=True;cases.append(classify_group_outcome(g,d))
        g.sic_pass=True
        cases.append(classify_group_outcome(g,d,route_reachable=False))
        cases.append(classify_group_outcome(g,d,visible=False))
        cases.append(classify_group_outcome(g,d,handover_blackout=True))
        g.success=True;cases.append(classify_group_outcome(g,d))
        self.assertEqual(cases,list(ALLOWED_OUTCOMES[:-1]))
        self.assertTrue(all(c in ALLOWED_OUTCOMES for c in cases))
        self.assertTrue(np.isnan(ratio(0,0)))

    def test_sic_source_is_first_failed_predecessor(self):
        config=load_config(CONFIG);a=group(0,(0,));b=group(1,(1,));c=group(2,(2,))
        a.gain=0.;b.gain=1.;c.gain=2.
        decode_groups(config,[a,b,c])
        self.assertFalse(a.rate_pass)
        self.assertTrue(b.rate_pass);self.assertFalse(b.sic_pass)
        self.assertEqual(b.metadata['sic_first_failed_predecessor_group_id'],0)
        self.assertEqual(c.metadata['sic_first_failed_predecessor_group_id'],0)

    def test_aoi_csi_elevation_power_and_rank(self):
        config=load_config(CONFIG);g=group();h=group(1,(2,),1)
        groups=[g,h];before=np.array([10.,20.,200.]);after=np.array([11.,21.,201.])
        amp=np.array([[.1,.2,.4]]);elev=np.array([[12.,18.,25.]])
        schedule_groups(config,groups,before,np.full((1,3),800.),False,0.)
        for x in groups: x.rate_pass=x.sic_pass=False;x.power=.2
        d={'status':'OK','valid':True,'transmission_attempted':True,'transmission_failure':False,
           'target_powers':{id(x):.2 for x in groups},'floors':{id(x):.05 for x in groups},
           'edge_budget':0.,'nonedge_budget':1.}
        core=[(x.gain,x.scheduled,x.priority,x.power,x.success) for x in groups]
        saved=after.copy()
        rows,c=enrich_group_diagnostic_rows(config,groups,d,ages_after=after,amplitudes=amp,
            elevations=elev,visible=np.ones_like(elev,bool),route_reachable=True,handover_blackout=False,
            system_weighted_aoi=62.6,policy='mira_sw_ucb',slot=0,satellite_id='sat_01',action=1)
        r=rows[0]
        self.assertEqual(r['pre_group_aoi_sum_slots'],30.)
        self.assertEqual(r['weighted_pre_group_aoi_slots'],21.)
        self.assertEqual(r['post_group_aoi_sum_slots'],32.)
        self.assertAlmostEqual(r['weighted_post_group_aoi_slots'],22.4)
        self.assertEqual(r['aoi_rank_in_system'],2)
        self.assertAlmostEqual(r['power_margin_above_floor'],.15)
        self.assertAlmostEqual(r['rate_margin_bps_hz'],-.000002)
        self.assertEqual(r['outcome_category'],'TRANSMITTED_RATE_BELOW_THRESHOLD')
        self.assertEqual(core,[(x.gain,x.scheduled,x.priority,x.power,x.success) for x in groups])
        np.testing.assert_array_equal(saved,after)
        g=group(0,(0,1,2));g.gain=.01;g.max_elevation_deg=25.
        schedule_groups(config,[g],before,np.full((1,3),800.),False,0.)
        d['target_powers']={id(g):.2};d['floors']={id(g):.05}
        r=enrich_group_diagnostic_rows(config,[g],d,ages_after=after,amplitudes=amp,elevations=elev,
            visible=np.ones_like(elev,bool),route_reachable=True,handover_blackout=False,
            system_weighted_aoi=float(after.sum()*.7),policy='mira_sw_ucb',slot=0,satellite_id='sat_01',action=1)[0][0]
        self.assertEqual(r['worst_user_channel_amplitude'],.1)
        self.assertAlmostEqual(r['gain'],.01)
        self.assertFalse(r['is_edge']);self.assertAlmostEqual(r['fraction_members_below_phi'],1/3)
        self.assertEqual(r['worst_user_elevation_deg'],12.)

    def test_class_interleaving(self):
        groups=[group(i,(i,)) for i in range(6)]
        for i,(g,e) in enumerate(zip(groups,[1,1,0,0,1,0])):g.gain=i+1;g.is_edge=bool(e)
        row=serving_power_diagnostics(groups,{'status':'OK'})
        self.assertEqual(row['csi_order_class_sequence'],'E E N N E N')
        self.assertEqual(row['csi_order_class_switch_count'],3)

    def test_boundary_csi_preserved_without_changing_membership_or_rng(self):
        config=load_config(CONFIG);state={**initial_epoch_state(),'agents':None,'previous_action':None,
            'action_histogram':Counter(),'action_change_events':0}
        members={m:np.arange(m*40,(m+1)*40,dtype=np.int32) for m in range(3)}
        amp=np.tile(np.arange(1,121,dtype=float),(8,1));elev=np.full_like(amp,45.)
        kwargs=dict(satellite=0,policy='fixed_grouping',policy_seed=0,fixed_action=4,
            message_members=members,elevation_deg=elev,select_action=select_grouping_action)
        random_before=np.random.get_state()
        first=groups_for_slot(config,state,slot=0,channel_amplitude=amp,diagnostics_enabled=True,**kwargs)[1]
        later=groups_for_slot(config,state,slot=1,channel_amplitude=amp[:,::-1].copy(),diagnostics_enabled=True,**kwargs)[1]
        for a,b in zip(first,later):
            np.testing.assert_array_equal(a.member_indices,b.member_indices)
            self.assertEqual(b.metadata['boundary_group_gain'],a.gain)
        self.assertTrue(any(a.gain!=b.gain for a,b in zip(first,later)))
        random_after=np.random.get_state()
        np.testing.assert_array_equal(random_before[1],random_after[1])

    def test_nullable_arrow_types(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'trace.csv'
            rows=[dict(policy='mira_sw_ucb',slot=0,sic_first_failed_predecessor_rate=None),
                  dict(policy='mira_sw_ucb',slot=1,sic_first_failed_predecessor_rate=.2)]
            write_csv(path,rows)
            csv_to_parquet(path,path.with_suffix('.parquet'),column_types=parquet_diagnostic_types(rows[0]))
            import pyarrow.parquet as pq
            self.assertEqual(pq.read_table(path.with_suffix('.parquet'))['sic_first_failed_predecessor_rate'].to_pylist(),[None,.2])


def artificial_trace(root, mismatch=False, other=False):
    """Explicitly artificial 3-slot records, no orbit, channel generator or run_simulation."""
    config=load_config(CONFIG)
    for policy in ('mira_sw_ucb','random_grouping','fixed_grouping'):
        rows=[];serving_rows=[]
        for slot in range(3):
            current=[]
            for population in range(1):
                groups=[];amp=np.full((8,12),.1);elev=np.tile(np.linspace(12,28,12),(8,1))
                for m in range(3):
                    g=group(0,(population*6+m*2,population*6+m*2+1),m)
                    g.max_elevation_deg=float(elev[0,g.member_indices].max());g.is_edge=g.max_elevation_deg<20
                    groups.append(g)
                ages=np.full(12,10.+slot)
                schedule_groups(config,groups,ages,np.full((8,12),800.),False,0.)
                fail=policy=='mira_sw_ucb' and slot==1 and population==0
                d=dict(status='POWER_INFEASIBLE_NO_TRANSMISSION' if fail else 'OK',valid=not fail,
                    transmission_attempted=not fail,transmission_failure=fail,
                    projection_solver_status=2 if fail else None,projection_solver_message='artificial fixture' if fail else '',
                    target_powers={id(g):1/3 for g in groups},floors={id(g):.05 for g in groups},
                    edge_budget=sum(g.is_edge for g in groups)/3,nonedge_budget=sum(not g.is_edge for g in groups)/3)
                for g in groups:
                    g.power=0 if fail else 1/3;g.rate_pass=g.sic_pass=g.success=not fail;g.rate_bps_hz=.1 if not fail else 0
                    g.metadata.update(boundary_group_gain=g.gain,boundary_group_csi_rank=g.message_id+1)
                if policy=='mira_sw_ucb' and slot==2 and population==0:
                    groups[0].rate_pass=groups[0].sic_pass=groups[0].success=False
                    groups[0].rate_bps_hz=0
                    groups[1].sic_pass=groups[1].success=False
                    groups[1].metadata.update(sic_first_failed_predecessor_message_id=0,sic_first_failed_predecessor_group_id=0,
                        sic_first_failed_predecessor_gain_db=-20.,sic_first_failed_predecessor_power=1/3,sic_first_failed_predecessor_rate=0.)
                after=ages.copy()
                for g in groups: after[g.member_indices]=1.5 if g.success else ages[g.member_indices]+1
                weighted=sum(float(after[g.member_indices].sum())*config.get('services.weights')[g.message_id] for g in groups)
                grows,stats=enrich_group_diagnostic_rows(config,groups,d,ages_after=after,amplitudes=amp,elevations=elev,
                    visible=np.ones_like(elev,bool),route_reachable=True,handover_blackout=False,
                    system_weighted_aoi=weighted,policy=policy,slot=slot,satellite_id=f'sat_0{population+1}',action=1)
                if other and policy=='mira_sw_ucb' and slot==0 and population==0:grows[0]['outcome_category']='OTHER_FAILURE'
                rows.extend(grows)
                c=dict(policy=policy,slot=slot,time_s=slot*.01,satellite_id=f'sat_0{population+1}',
                    action=1,is_regrouping_slot=int(slot==0),candidate_groups=3,candidate_edge_groups=sum(g.is_edge for g in groups),
                    candidate_nonedge_groups=sum(not g.is_edge for g in groups),scheduled_edge_groups=sum(g.is_edge for g in groups),
                    scheduled_nonedge_groups=sum(not g.is_edge for g in groups),weighted_aoi_slots=weighted,
                    edge_budget=d['edge_budget'],nonedge_budget=d['nonedge_budget'],edge_floor_total=.05,nonedge_floor_total=.05)
                c.update(stats);current.append(c)
            total=sum(c['weighted_aoi_slots'] for c in current)
            for c in current:c['aggregated_weighted_sum_aoi_slots']=total
            serving_rows.extend(current)
        directory=root/'runs'/policy;directory.mkdir(parents=True)
        write_csv(directory/'group_trace.csv',rows);write_csv(directory/'slot_metrics.csv',serving_rows)
        atomic_write_json(directory/'summary.json',dict(policy=policy,channel_seed=0,slots_completed=3,status='COMPLETED',
            environment_trace_hash='artificial-environment' if not mismatch or policy!='fixed_grouping' else 'different',
            channel_trace_hash='artificial-channel',reward_clipping_rate_max=0.))
    return config


class DiagnosticPipelineTests(unittest.TestCase):
    def test_missing_trace_never_runs_simulator(self):
        with tempfile.TemporaryDirectory() as temp,patch('mira_leo_study.simulation.run_simulation',side_effect=AssertionError('simulation forbidden')):
            with self.assertRaises(DiagnosticDataError):generate_diagnostics(load_config(CONFIG),temp)

    def test_reject_old_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);d=root/'runs/mira_sw_ucb';d.mkdir(parents=True)
            write_csv(d/'group_trace.csv',[dict(slot=0,gain=.1)])
            with self.assertRaisesRegex(DiagnosticDataError,'Rerun'):generate_diagnostics(load_config(CONFIG),root)

    def test_environment_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);config=artificial_trace(root,mismatch=True)
            with self.assertRaisesRegex(DiagnosticDataError,'environment_trace_hash'):generate_diagnostics(config,root)

    def test_other_failure_is_explicit_warning(self):
        with tempfile.TemporaryDirectory() as temp,patch('mira_leo_study.diagnostic_plots.generate_plots',return_value=[]):
            root=Path(temp);config=artificial_trace(root,other=True)
            result=generate_diagnostics(config,root)
            self.assertEqual(result['status'],'WARNING_OTHER_FAILURE')
            self.assertEqual(result['other_failure_count'],1)

    def test_aoi_conservation_failure_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);config=artificial_trace(root)
            path=root/'runs/mira_sw_ucb/group_trace.csv'
            with path.open(encoding='utf-8',newline='') as handle:rows=list(csv.DictReader(handle))
            rows[0]['weighted_post_group_aoi_slots']='999999'
            write_csv(path,rows)
            with self.assertRaisesRegex(DiagnosticDataError,'reconcile'):generate_diagnostics(config,root)

    def test_diagnostic_cli_dispatch_does_not_simulate(self):
        from mira_leo_study.__main__ import main
        with patch('mira_leo_study.diagnostics.generate_diagnostics',return_value={'status':'artificial-cli-fixture'}) as generate, \
             patch('mira_leo_study.__main__.run_simulation',side_effect=AssertionError('simulation forbidden')):
            self.assertEqual(main(['diagnose-results','--outputs','artificial-missing-path','--seed','0']),0)
            self.assertEqual(generate.call_args.kwargs['seed'],0)

    def test_artificial_pipeline_all_plots_and_summaries(self):
        with tempfile.TemporaryDirectory() as temp,patch('mira_leo_study.simulation.run_simulation',side_effect=AssertionError('simulation forbidden')):
            root=Path(temp);config=artificial_trace(root)
            result=generate_diagnostics(config,root)
            out=Path(result['diagnostics_directory'])
            self.assertEqual(result['status'],'DIAGNOSTICS_COMPLETED')
            self.assertEqual(result['other_failure_count'],0)
            self.assertEqual(len(result['plots']),36)
            self.assertTrue(all((out/name).stat().st_size>100 for name in result['plots']))
            for name in ['policy_diagnostic_summary.csv','failure_attribution_summary.csv','elevation_bin_summary.csv',
                         'action_diagnostic_summary.csv','top_aoi_group_events.csv','group_outcomes.csv',
                         'worst_aoi_episodes.csv','worst_power_failure_episodes.csv','worst_rate_failure_groups.csv',
                         'policy_comparison_conditioned_on_elevation.csv','policy_comparison_conditioned_on_csi.csv',
                         'policy_comparison_conditioned_on_action.csv','diagnostic_report.md']:
                self.assertTrue((out/name).exists(),name)
            with (out/'failure_attribution_summary.csv').open(encoding='utf-8',newline='') as handle:rows=list(csv.DictReader(handle))
            self.assertEqual(sum(int(r['count']) for r in rows),27)
            with (out/'policy_diagnostic_summary.csv').open(encoding='utf-8',newline='') as handle:summary=list(csv.DictReader(handle))
            mira=next(r for r in summary if r['policy']=='mira_sw_ucb')
            self.assertAlmostEqual(float(mira['transmitted_rate_failure_rate']),1/6)
            self.assertAlmostEqual(float(mira['sic_propagated_failure_rate']),1/6)
            self.assertAlmostEqual(float(mira['power_infeasible_group_opportunity_rate']),3/9)
            with (out/'worst_aoi_episodes.csv').open(encoding='utf-8',newline='') as handle:episodes=list(csv.DictReader(handle))
            self.assertIn('action_by_serving_satellite',episodes[0])
            self.assertIn('serving_gain_statistics',episodes[0])
            self.assertIn('rate_failure_rate',mira)
            self.assertIn('Observational',(out/'diagnostic_report.md').read_text(encoding='utf-8'))

    @simulation_test
    def test_real_on_off_parity_opt_in(self):
        config=load_config(CONFIG)
        config=config.with_overrides({'algorithm.reward_bounds':{'lower':-1e9,'upper':1e9,'identity':'opt-in-parity','context':reward_context(config,0)}})
        with tempfile.TemporaryDirectory() as temp:
            left=Path(temp)/'off';right=Path(temp)/'on'
            from mira_leo_study.algorithms import allocate_power
            captured=[]
            def record(cfg,groups):
                diagnostic=allocate_power(cfg,groups);captured.append(groups);return diagnostic
            with patch('mira_leo_study.simulation.allocate_power',side_effect=record):
                a=run_simulation(config,channel_seed=0,policy='mira_sw_ucb',max_slots=200,run_directory=left)
            off_groups=captured[:];captured.clear()
            with patch('mira_leo_study.simulation.allocate_power',side_effect=record):
                b=run_simulation(config,channel_seed=0,policy='mira_sw_ucb',max_slots=200,run_directory=right,save_group_trace=True)
            for off_slot,on_slot in zip(off_groups,captured):
                self.assertEqual(len(off_slot),len(on_slot))
                for x,y in zip(off_slot,on_slot):
                    np.testing.assert_array_equal(x.member_indices,y.member_indices)
                    for key in ('key','gain','scheduled','priority','power','success','rate_pass','sic_pass','interference','sinr'):
                        self.assertEqual(getattr(x,key),getattr(y,key),key)
            for key in ['environment_trace_hash','channel_trace_hash','action_histogram','bandit_observations']:
                self.assertEqual(a['summary'][key],b['summary'][key])
            with (left/'slot_metrics.csv').open(encoding='utf-8',newline='') as f:off=list(csv.DictReader(f))
            with (right/'slot_metrics.csv').open(encoding='utf-8',newline='') as f:on=list(csv.DictReader(f))
            for x,y in zip(off,on):
                for key,value in x.items():self.assertEqual(value,y[key],key)
            self.assertEqual((left/'convergence_slot_metrics.csv').read_bytes(),(right/'convergence_slot_metrics.csv').read_bytes())


if __name__=='__main__':unittest.main()
