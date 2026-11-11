"""Exactly four MIRA angles plus Random/Fixed only at phi=15, with one allocator."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import numpy as np

from .config import (StudyConfig, P1_PHI_THETA_DEG, load_config, validate_fixed_study_config,
                     validate_p1_ablation_config)
from .protected_power import POWER_MODE, POOL_RULE_VERSION
from .reward_pilot import run_reward_bounds_pilot, load_reward_bounds_file
from .simulation import CHECKPOINT_SCHEMA_VERSION, run_simulation
from .storage import atomic_write_json, write_csv
from .geo_edge import analyze_geo_edge

DEFAULT_OUTPUTS = Path("outputs/p1_ablation_13_15_20_25")
ENVIRONMENT_KEYS = ("layout_hash", "subscription_hash", "trajectory_hash", "channel_trace_hash", "environment_trace_hash", "slots_completed")
RUN_MATRIX = (
    ("mira_phi13", "mira_sw_ucb", 13.), ("mira_phi15", "mira_sw_ucb", 15.),
    ("mira_phi20", "mira_sw_ucb", 20.), ("mira_phi25", "mira_sw_ucb", 25.),
    ("random_phi15", "random_grouping", 15.), ("fixed_phi15", "fixed_grouping", 15.),
)
ARRAY_FIELDS = ("time_s", "weighted_aoi_slots", "successful_groups", "P1_slot_value",
                "transmission_failure_power", "protected_scheduled_count", "ordinary_scheduled_count",
                "promoted_count", "protected_power_actual", "ordinary_power_actual")


def experiment_matrix():
    return [dict(run_id=name, policy=policy, phi_theta_deg=phi, fixed_action=4 if policy=="fixed_grouping" else None,
                 power_mode=POWER_MODE, power_pool_rule_version=POOL_RULE_VERSION, seed=0)
            for name,policy,phi in RUN_MATRIX]


def check_common_environment(summaries):
    if len(summaries)!=6 or {r["run_id"] for r in summaries}!={r[0] for r in RUN_MATRIX}:
        raise ValueError("P1 comparison requires exactly the six declared run IDs")
    check={}
    for key in ENVIRONMENT_KEYS:
        values=[r.get(key) for r in summaries]
        if any(v is None or v=="" for v in values) or len(set(values))!=1:
            raise ValueError(f"six-run environment mismatch: {key}")
        check[key]=dict(equal=True,value=values[0])
    return dict(status="PASS",run_count=6,checks=check)


def read_and_validate_run(directory, summary, config, slots):
    """Bounded CSV reading plus compact numeric arrays, never a full list of CSV dicts."""
    if summary.get("status") not in {"COMPLETED","COMPLETED_WITH_POWER_OUTAGES","REWARD_RANGE_MISMATCH"}:
        raise ValueError("P1 requires a complete, constraint-valid run")
    if summary.get("power_allocator")!=POWER_MODE or summary.get("power_pool_rule_version")!=POOL_RULE_VERSION:
        raise ValueError("cached run uses a different allocator/rule")
    if summary.get("schema_version")!=CHECKPOINT_SCHEMA_VERSION or summary.get("slots_completed")!=slots:
        raise ValueError("cached run has a different schema or horizon")
    if summary.get("phi_theta_deg")!=config.get("algorithm.phi_theta_deg") or summary.get("config_id")!=config.config_id:
        raise ValueError("cached config/phi identity mismatch")
    data=np.empty((slots,len(ARRAY_FIELDS)),dtype=float)
    held={}; count=0; projections=0; warm_c=0; warm_delta=0.; warm_count=0
    with (Path(directory)/"slot_metrics.csv").open(encoding="utf-8",newline="") as handle:
        for row in csv.DictReader(handle):
            if count>=slots or int(row["slot"])!=count:raise ValueError("missing or duplicate physical slot")
            values=np.array([float(row[key]) for key in ARRAY_FIELDS])
            if not np.all(np.isfinite(values)):raise ValueError("nonfinite slot metrics")
            data[count]=values
            if not math.isclose(values[0],count*.01,abs_tol=1e-10):raise ValueError("physical time mismatch")
            if not math.isclose(float(row["P1_slot_value"]),int(row["successful_groups"])-float(row["weighted_aoi_slots"]),rel_tol=1e-12,abs_tol=1e-8):
                raise ValueError("P1 must be C minus post-update weighted AoI in slots")
            expected_reward=float(row["P1_slot_value"])-.05*3*40*int(row["edge_groups"])
            if not math.isclose(float(row["raw_reward"]),expected_reward,rel_tol=1e-12,abs_tol=1e-8):raise ValueError("bandit penalty/P1 decomposition mismatch")
            if row["power_allocator"]!=POWER_MODE or int(row["post_schedule_power_drop_count"]) or int(row["transmitted_sic_violation_count"]):
                raise ValueError("allocator/schedule/global SIC mismatch")
            action=int(row["action"])
            if action not in config.get("algorithm.action_set") or (summary["policy"]=="fixed_grouping" and action!=4):
                raise ValueError("illegal grouping action")
            epoch=count//100;boundary=count%100==0
            if int(row["regrouping_epoch"])!=epoch or int(row["is_regrouping_slot"])!=int(boundary):raise ValueError("epoch cadence mismatch")
            identity=(action,row["membership_hash"],row["action_owner_satellite_id"])
            if epoch in held and held[epoch]!=identity:raise ValueError("membership/action owner changed within epoch")
            held[epoch]=identity
            if not boundary and (int(row["regrouping_action_change"]) or int(row["device_regrouping_events"])):raise ValueError("off-boundary regrouping")
            if int(row["csi_slot_index"])!=count or int(row["power_csi_slot_index"])!=count:raise ValueError("stale CSI")
            scheduled=int(row["scheduled_groups"]);protected=int(row["protected_scheduled_count"]);ordinary=int(row["ordinary_scheduled_count"])
            if protected+ordinary!=scheduled or int(row["geo_edge_scheduled_count"])+int(row["promoted_count"])!=protected:
                raise ValueError("power-pool membership does not partition the schedule")
            cap=int(config.get("derived.algorithm2_U"));edges=int(row["candidate_edge_groups"])
            if (int(row["scheduled_edge_groups"])!=min(edges,cap)
                    or int(row["scheduled_nonedge_groups"])!=(0 if edges>cap else min(int(row["candidate_nonedge_positive_priority"]),cap-edges))):
                raise ValueError("Algorithm-2 admission changed")
            actual=float(row["transmitted_power_normalized"])
            if actual<0 or actual>1+1e-10:raise ValueError("serving EIRP exceeded budget")
            outage=bool(int(row["transmission_failure_power"]))
            if outage:
                if (actual!=0 or int(row["successful_groups"]) or int(row["transmitted_groups"])
                        or int(row["power_plan_valid"]) or int(row["transmission_attempted"])):
                    raise ValueError("power outage must mean zero actual transmission and success")
                if row["power_failure_reason"] not in {"FLOOR_EXCEEDS_POOL","POOL_SIC_INCOMPATIBLE","LP_NUMERIC_FAILURE","INVALID_TARGET_CONSTRAINTS"}:
                    raise ValueError("unknown power failure reason")
            else:
                expected_p=.7 if protected and ordinary else 1. if protected else 0.
                expected_o=.3 if protected and ordinary else 1. if ordinary else 0.
                if not int(row["power_plan_valid"]) or abs(float(row["protected_power_actual"])-expected_p)>1e-10 or abs(float(row["ordinary_power_actual"])-expected_o)>1e-10:
                    raise ValueError("strict pool equalities not satisfied")
            if int(row["candidate_groups"])>30 or int(row["successful_groups"])>int(row["transmitted_groups"]) or int(row["transmitted_groups"])>scheduled:
                raise ValueError("group count mismatch")
            projections+=int(row["sic_projection_used"])
            if values[0]>=float(config.get("experiment.warmup_report_s")):
                warm_c+=int(row["successful_groups"]);warm_delta+=float(row["weighted_aoi_slots"]);warm_count+=1
            count+=1
    if count!=slots:raise ValueError("incomplete slot trace")
    delta=data[:,1];success=data[:,2];p1=data[:,3]
    for key,value in (("sum_C",success.sum()),("sum_weighted_delta_slots",delta.sum()),("sum_P1",p1.sum()),("mean_P1_per_slot",p1.mean())):
        if not math.isclose(float(summary[key]),float(value),rel_tol=1e-10,abs_tol=1e-6):raise ValueError(f"summary/trace mismatch: {key}")
    if int(summary["post_warmup_slots"])!=warm_count:raise ValueError("warmup range mismatch")
    if warm_count and not math.isclose(summary["post_warmup_mean_P1"],(warm_c-warm_delta)/warm_count,rel_tol=1e-10,abs_tol=1e-6):raise ValueError("post-warmup P1 mismatch")
    if int(summary["power_outage_slots"])!=int(data[:,4].sum()) or projections!=summary["sic_projection_count"]:raise ValueError("outage/projection counter mismatch")
    if summary["transmitted_sic_violation_count"]!=0 or summary["sum_power_excess"]!=0:raise ValueError("transmitted constraints violated")
    if sum(summary["action_decision_count"].values())!=math.ceil(slots/100):raise ValueError("action decisions not once per epoch")
    expected_observations=math.ceil(slots/100) if summary["policy"]=="mira_sw_ucb" else 0
    if summary["bandit_observation_count"]!=expected_observations:raise ValueError("bandit observation counter mismatch")
    # Reconcile independently written convergence data as well.
    count=0
    with (Path(directory)/"convergence_slot_metrics.csv").open(encoding="utf-8",newline="") as handle:
        for row in csv.DictReader(handle):
            if count>=slots or int(row["slot"])!=count:raise ValueError("invalid convergence slots")
            for key,value in (("weighted_sum_aoi_total",delta[count]),("P1_slot_value",p1[count]),("successful_groups",success[count])):
                if not math.isclose(float(row[key]),float(value),rel_tol=1e-10,abs_tol=1e-6):raise ValueError("convergence metrics mismatch")
            count+=1
    if count!=slots:raise ValueError("incomplete convergence trace")
    return data


def _csv_summary(summary):
    fields=("run_id","phi_theta_deg","policy","channel_seed","slots_completed","status","config_id",
            "sum_C","mean_C","sum_weighted_delta_slots","mean_weighted_delta_slots","sum_P1","mean_P1_per_slot","post_warmup_mean_P1",
            "post_warmup_sum_C","post_warmup_mean_C","post_warmup_sum_weighted_delta_slots","post_warmup_mean_weighted_delta_slots","post_warmup_sum_P1","post_warmup_slots",
            "weighted_mean_aoi_after_warmup_ms","weighted_mean_aoi_per_weight_unit_after_warmup_ms",
            "successful_group_count","scheduled_group_count","transmitted_group_count","conditional_group_success_ratio","capacity_goodput_bps",
            "geo_edge_candidate_count","geo_edge_scheduled_count","promoted_weak_nonedge_count","protected_count","ordinary_count","protected_pool_fraction_realized",
            "power_infeasible_serving_slots","power_outage_rate","floor_infeasible_slots","cross_pool_sic_infeasible_slots","lp_numeric_failure_slots","sic_projection_count","transmitted_sic_violation_count","sum_power_excess",
            "rate_threshold_failure_count","sic_failure_count","propagated_sic_failure_count","regrouping_events_total","action_decision_count","action_histogram","handover_count","route_unreachable_slots",
            "power_allocator","power_pool_rule_version","reward_bounds_identity","reward_clipping_rate_max")
    row={key:summary[key] for key in fields}
    row["seed"]=row["channel_seed"]
    row["full_horizon_weighted_sum_aoi_mean_ms"]=summary["mean_weighted_delta_slots"]*10.
    for key in ("action_decision_count","action_histogram"):row[key]=json.dumps(row[key],sort_keys=True)
    return row


def _comparison_artifacts(root, summaries, arrays):
    comparison=root/"comparison";figures=root/"figures"
    comparison.mkdir(exist_ok=True);figures.mkdir(exist_ok=True)
    rows=[_csv_summary(s) for s in summaries]
    mira=[r for r in rows if r["policy"]=="mira_sw_ucb"]
    baseline=[r for r in rows if r["phi_theta_deg"]==15.]
    write_csv(comparison/"all_six_runs.csv",rows)
    write_csv(comparison/"ablation_mira_13_15_20_25.csv",mira)
    write_csv(comparison/"baselines_phi15.csv",baseline)
    write_csv(comparison/"p1_objective_decomposition.csv",[{key:r[key] for key in ("run_id","sum_C","sum_weighted_delta_slots","sum_P1","mean_P1_per_slot","post_warmup_mean_P1")} for r in rows])
    write_csv(comparison/"pool_and_feasibility_summary.csv",[{key:r[key] for key in ("run_id","slots_completed","geo_edge_candidate_count","geo_edge_scheduled_count","promoted_weak_nonedge_count","protected_count","ordinary_count","protected_pool_fraction_realized","power_outage_rate","floor_infeasible_slots","cross_pool_sic_infeasible_slots","lp_numeric_failure_slots","sic_projection_count","transmitted_sic_violation_count","sum_power_excess")} for r in rows])
    running_fields=["run_id","slot","time_s","C","weighted_delta_slots","P1_slot_value","running_mean_aoi_slots","running_mean_P1","running_mean_aoi_after_warmup_slots","running_mean_P1_after_warmup"]
    with (comparison/"p1_running_objective.csv").open("w",encoding="utf-8",newline="") as handle:
        writer=csv.writer(handle);writer.writerow(running_fields)
        for run_id,data in arrays.items():
            mean_a=np.cumsum(data[:,1])/np.arange(1,len(data)+1)
            mean_p=np.cumsum(data[:,3])/np.arange(1,len(data)+1)
            warm_indices=np.flatnonzero(data[:,0]>=10.)
            warm_start=int(warm_indices[0]) if len(warm_indices) else len(data)
            warm_a=np.cumsum(data[warm_start:,1])/np.arange(1,len(data)-warm_start+1)
            warm_p=np.cumsum(data[warm_start:,3])/np.arange(1,len(data)-warm_start+1)
            for i in range(len(data)):
                writer.writerow((run_id,i,data[i,0],data[i,2],data[i,1],data[i,3],mean_a[i],mean_p[i],
                                 warm_a[i-warm_start] if i>=warm_start else None,
                                 warm_p[i-warm_start] if i>=warm_start else None))
    paired=[];main=next(r for r in baseline if r["policy"]=="mira_sw_ucb")
    for row in baseline:
        if row is not main:
            paired.append(dict(comparison=f"mira_phi15 - {row['run_id']}",mean_P1_absolute_difference=main["mean_P1_per_slot"]-row["mean_P1_per_slot"],mean_weighted_aoi_slots_difference=main["mean_weighted_delta_slots"]-row["mean_weighted_delta_slots"]))
    write_csv(comparison/"paired_absolute_differences_phi15.csv",paired)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure_sources={}
    def save(fig,name,source):
        fig.tight_layout();fig.savefig(figures/name,dpi=150);plt.close(fig);figure_sources[name]=source
    for prefix,subset,source in (("ablation_phi",mira,"ablation_mira_13_15_20_25.csv"),("baseline_phi15",baseline,"baselines_phi15.csv")):
        labels=[str(r["phi_theta_deg"]) if prefix=="ablation_phi" else r["policy"].replace("_grouping","") for r in subset]
        for suffix,field,label in (("weighted_aoi","full_horizon_weighted_sum_aoi_mean_ms","Weighted-sum AoI (ms), full horizon; lower is better"),
                                   ("p1_objective","mean_P1_per_slot","Achieved mean P1 (C - weighted AoI in slots); higher is better"),
                                   ("group_success","mean_C","Successful multicast groups / slot"),
                                   ("power_outage","power_outage_rate","No-transmission power-outage fraction"),
                                   ("capacity_goodput","capacity_goodput_bps","Capacity goodput (bit/s); no deadline model")):
            fig,ax=plt.subplots(figsize=(9,5));ax.bar(labels,[r[field] for r in subset]);ax.set(xlabel="Geographic edge threshold (degrees)" if prefix=="ablation_phi" else "Grouping policy at phi=15",ylabel=label);ax.grid(axis="y",alpha=.2)
            name=f"{prefix}_vs_{suffix}.png" if prefix=="ablation_phi" else f"{prefix}_{suffix}.png"
            save(fig,name,source)
        for metric,column,ylabel in (("aoi",1,"Running mean weighted-sum AoI (slots)"),("p1",3,"Running mean achieved P1")):
            fig,ax=plt.subplots(figsize=(11,5))
            for row in subset:
                data=arrays[row["run_id"]];values=np.cumsum(data[:,column])/np.arange(1,len(data)+1)
                ax.plot(data[:,0],values,label=row["run_id"],lw=1)
            ax.set(xlabel="Time (s)",ylabel=ylabel);ax.legend();ax.grid(alpha=.2)
            name=(f"ablation_phi_convergence_{metric}.png" if prefix=="ablation_phi" else f"baseline_phi15_convergence_{metric}.png")
            save(fig,name,"p1_running_objective.csv")
    fig,ax=plt.subplots(figsize=(13,5));positions=np.arange(6)
    for offset,key,label in ((-.25,"protected_count","Protected"),(0.,"promoted_weak_nonedge_count","Promoted (subset of protected)"),(.25,"ordinary_count","Ordinary")):
        ax.bar(positions+offset,[r[key]/r["slots_completed"] for r in rows],width=.24,label=label)
    ax.set(xticks=positions,xticklabels=[r["run_id"] for r in rows],ylabel="Mean scheduled groups per slot");ax.legend();save(fig,"power_pool_group_counts.png","pool_and_feasibility_summary.csv")
    fig,axes=plt.subplots(2,1,figsize=(12,8))
    axes[0].bar([r["run_id"] for r in rows],[r["protected_pool_fraction_realized"] if r["protected_pool_fraction_realized"] is not None else np.nan for r in rows]);axes[0].set_ylabel("Protected / total actual TX power\n(includes full-pool slots)")
    axes[1].bar([r["run_id"] for r in rows],[r["transmitted_sic_violation_count"] for r in rows]);axes[1].set_ylabel("Transmitted global-SIC violations")
    save(fig,"power_pool_allocation_and_sic.png","pool_and_feasibility_summary.csv")
    return rows,figure_sources


def _write_provenance(root,config):
    source_file=Path(__file__).with_name("configs")/"p1_source_provenance.yaml"
    original=json.loads(source_file.read_text(encoding="utf-8")) if source_file.exists() else {"original_source":"not bundled"}
    files=[*Path(__file__).parent.glob("*.py"),Path(__file__).with_name("configs")/"base.yaml"]
    original.update(current_source_sha256={str(p.relative_to(Path(__file__).parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                    runtime_python=platform.python_version(),runtime_numpy=np.__version__,resolved_config_id=config.config_id,
                    power_mode=POWER_MODE,power_pool_rule_version=POOL_RULE_VERSION,
                    seeds={key:config.get(key) for key in ("devices.layout_seed","devices.subscription_seed","randomness.fixed_channel_seed","randomness.policy_seed")})
    atomic_write_json(root/"provenance.json",original)


def run_p1_ablation(config: StudyConfig, output_directory=DEFAULT_OUTPUTS, *, debug_slots=None, seed=0,
                    resume=False, save_group_trace=False):
    validate_fixed_study_config(config)
    if seed!=0:raise ValueError("P1 ablation uses fixed channel seed 0")
    slots=config.slot_count if debug_slots is None else int(debug_slots)
    if not 1<=slots<=config.slot_count or (debug_slots is not None and slots==config.slot_count):
        raise ValueError("debug slots must be 1..47999; omit for the formal 48000-slot runs")
    root=Path(output_directory)
    if debug_slots is not None:root=root/"debug"/f"slots_{slots}_seed_0"
    root.mkdir(parents=True,exist_ok=True)
    _write_provenance(root,config)
    entries=experiment_matrix()
    for entry in entries:entry.update(status="PENDING",directory=str(root/"runs"/entry["run_id"]/"seed_0"))
    manifest=dict(status="RUNNING",formal_horizon=debug_slots is None,slots_per_run=slots,runs=entries,figures={})
    atomic_write_json(root/"manifest.json",manifest)
    summaries,arrays=[],{}
    try:
        bounds_by_phi={}
        for phi in P1_PHI_THETA_DEG:
            variant=config.with_overrides({"algorithm.phi_theta_deg":phi})
            if debug_slots is not None:
                window=max(1.,min(30.,float(math.ceil(slots/100))))
                variant=variant.with_overrides({"formal_results.reward_pilot_windows_s":[[0.,window]]})
            validate_p1_ablation_config(variant)
            calibration_root=root/"calibration"/f"phi{phi:g}"
            cache=calibration_root/"reward_bounds"/"reward_bounds.json"
            if cache.exists():variant=load_reward_bounds_file(variant,cache,0)
            else:variant=run_reward_bounds_pilot(variant,calibration_root,seed=0)["config"]
            bounds_by_phi[phi]=variant
        for entry in entries:
            variant=bounds_by_phi[entry["phi_theta_deg"]]
            directory=Path(entry["directory"])
            cached=directory/"summary.json"
            cached_summary=json.loads(cached.read_text(encoding="utf-8")) if resume and cached.exists() else None
            if cached_summary is not None:
                if cached_summary.get("config_id")!=variant.config_id or cached_summary.get("policy")!=entry["policy"]:
                    raise ValueError("cached result identity mismatch; choose a new output directory")
            if (cached_summary is not None and not (directory/"checkpoint.pkl").exists()
                    and cached_summary.get("status") in {"COMPLETED","COMPLETED_WITH_POWER_OUTAGES","REWARD_RANGE_MISMATCH"}):
                summary=cached_summary
            else:
                entry["status"]="RUNNING";atomic_write_json(root/"manifest.json",manifest)
                result=run_simulation(variant,channel_seed=0,policy=entry["policy"],fixed_action=4,
                                      max_slots=slots,run_directory=directory,resume=resume,
                                      save_group_trace=save_group_trace,save_geo_edge_trace=entry["run_id"]=="mira_phi15")
                summary=result["summary"]
            summary=dict(summary,run_id=entry["run_id"])
            arrays[entry["run_id"]]=read_and_validate_run(directory,summary,variant,slots)
            summaries.append(summary);entry["status"]=summary["status"];entry["config_id"]=variant.config_id
            atomic_write_json(root/"manifest.json",manifest)
        environment=check_common_environment(summaries)
        (root/"comparison").mkdir(exist_ok=True)
        atomic_write_json(root/"comparison/environment_identity_check.json",environment)
        rows,figure_sources=_comparison_artifacts(root,summaries,arrays)
        geo=analyze_geo_edge(bounds_by_phi[15.],root/"runs/mira_phi15/seed_0/geo_edge",root/"geo_edge_phi15",root/"figures",expected_slots=slots)
        write_csv(root/"comparison/edge_geo_summary_phi15.csv",[dict(run_id="mira_phi15",**{k:v for k,v in geo.items() if k!="spatial_metrics"})])
        for name,source in (("geo_edge_phi15_map.png","geo_edge_phi15/edge_member_positions.csv"),("geo_edge_phi15_spread_distribution.png","geo_edge_phi15/edge_group_snapshots.csv"),("geo_edge_phi15_time_occupancy.png","geo_edge_phi15/geo_edge_time_distribution.csv")):
            figure_sources[name]="../"+source
        for name,source in figure_sources.items():manifest["figures"][name]=dict(path=str(root/"figures"/name),source_csv=str(root/"comparison"/source))
        manifest["figures"]["geo_edge_phi15_map.png"]["additional_source_csv"]=str(root/"geo_edge_phi15/edge_group_snapshots.csv")
        manifest.update(status="COMPLETED",environment_identity_check=environment)
        atomic_write_json(root/"manifest.json",manifest)
        result=dict(status="COMPLETED",formal_horizon=debug_slots is None,slots_per_run=slots,run_count=6,
                    comparison=rows,environment=environment,geo_edge_phi15=geo,artifact_root=str(root))
        atomic_write_json(root/"comparison/ablation_summary.json",result)
        mira=[r for r in rows if r["policy"]=="mira_sw_ucb"]
        best_p1=max(mira,key=lambda r:r["mean_P1_per_slot"])
        best_aoi=min(mira,key=lambda r:r["mean_weighted_delta_slots"])
        report=["# MIRA-LEO P1 六組消融實驗", "",
                f"實際完成 {slots} slots/run；正式 48,000-slot 實驗={debug_slots is None}。所有比較採 post-update AoI，P1=C−weighted AoI（slots），不含 bandit penalty。",
                "唯一 allocator 為 scheduled geo-edge + 嚴格弱於最強 geo-edge 的 non-edge promotion；兩池共享 70/30、單池 100%，全域 SIC、floors 與 EIRP 無解時零傳輸且保留 schedule。",
                "Fixed(a=4) 固定每個 message 的分組數量，仍每秒依 CSI 重建成員；並非全實驗固定 membership。",
                "| Run | phi | Policy | Mean P1 ↑ | Weighted AoI (slots) ↓ | Mean C ↑ | Power outage fraction |",
                "|---|---:|---|---:|---:|---:|---:|"]
        report += [f"| {r['run_id']} | {r['phi_theta_deg']} | {r['policy']} | {r['mean_P1_per_slot']:.8g} | {r['mean_weighted_delta_slots']:.8g} | {r['mean_C']:.6g} | {r['power_outage_rate']:.6g} |" for r in rows]
        report += ["",f"本次有限時段中，MIRA achieved P1 最佳仰角={best_p1['phi_theta_deg']}°；weighted AoI 最佳仰角={best_aoi['phi_theta_deg']}°。這是啟發式達成值，並非 P1 global optimum。",
                   "warm-up 後欄位若為空，表示觀測未超過 10 秒，沒有以零替代未觀測結果。",
                   "weighted_mean_aoi_after_warmup_ms 是 aggregated weighted-sum AoI×slot_ms；per-weight-unit 另有明確欄位。capacity_goodput_bps 沒有 deadline 模型。",
                   "地理集中性：群內距離與同 snapshot 群組間 centroid 距離分開分析，見 geo_edge_phi15/report_geo_edge_phi15.md；低仰角不自動代表地面位置相近。",
                   "六組場景 hashes 已核對。圖與 CSV 的對照位於 manifest.json。REWARD_RANGE_MISMATCH（如有）保留原 clipping 診斷，不調 B 或掩蓋失敗。",
                   "單一 layout/channel/policy seed、有限視窗與固定滿載外部干擾限制了泛化推論。PPO/GA/AIM-static=NOT_IMPLEMENTED。"]
        (root/"report.md").write_text("\n\n".join(report)+"\n",encoding="utf-8")
        return result
    except Exception as exc:
        manifest.update(status="INCOMPLETE",error=str(exc))
        atomic_write_json(root/"manifest.json",manifest)
        (root/"report.md").write_text("# P1 實驗未完成\n\n"+str(exc)+"\n\n未生成正式比較結論；檢查 manifest.json 的各 run 狀態。\n",encoding="utf-8")
        raise
