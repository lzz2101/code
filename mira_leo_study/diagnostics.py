"""Post-run diagnostics using bounded CSV/Parquet batches and a disk-backed index.

No simulator import: missing traces never trigger a pilot or a simulation.
Quantiles use every matching observation, one numeric column at a time.
"""
from __future__ import annotations

import csv
import json
import math
import sqlite3
from pathlib import Path

import numpy as np

from .diagnostic_trace import ALLOWED_OUTCOMES, DIAGNOSTIC_SCHEMA_VERSION, TEXT_FIELDS
from .storage import atomic_write_json, write_csv

POLICIES = ("mira_sw_ucb", "random_grouping", "fixed_grouping")
REQUIRED_GROUP_FIELDS = {
    "diagnostic_schema_version", "policy", "slot", "time_s", "action",
    "message_id", "group_id", "is_edge", "scheduled", "outcome_category", "gain_db",
    "weighted_pre_group_aoi_slots", "weighted_post_group_aoi_slots", "aoi_rank_in_system",
    "elevation_bin", "worst_user_elevation_deg", "gain_change_from_boundary_db",
    "sic_first_failed_predecessor_group_id", "rate_margin_bps_hz", "milliseconds_since_regrouping",
    "satellite_id", "group_size", "message_weight", "max_elevation_deg", "power", "target_power", "power_floor",
    "transmission_attempted", "transmission_failure_power", "rate_pass", "sic_pass", "success", "rate_bps_hz",
    "interference", "sinr", "csi_rank_changed_from_boundary", "priority",
}
REQUIRED_SERVING_FIELDS = {"policy", "slot", "satellite_id", "action", "power_infeasible",
    "csi_order_class_switch_count", "serving_elevation_min_deg", "serving_elevation_mean_deg",
    "serving_elevation_max_deg", "group_order_inversion_count", "edge_fraction"}
REQUIRED_SERVING_FIELDS |= {"time_s", "is_regrouping_slot", "candidate_groups", "candidate_edge_groups",
    "candidate_nonedge_groups", "scheduled_edge_groups", "scheduled_nonedge_groups", "weighted_aoi_slots",
    "aggregated_weighted_sum_aoi_slots", "edge_budget", "nonedge_budget", "edge_floor_total", "nonedge_floor_total"}
INTEGER_FIELDS = {"slot", "regrouping_epoch", "is_regrouping_slot", "action",
    "message_id", "group_id", "group_size", "scheduled", "is_edge", "success", "rate_pass", "sic_pass",
    "transmission_attempted", "transmission_failure_power", "aoi_rank_in_system", "diagnostic_schema_version"}
EXTRA_TEXT = {"membership_hash", "action_owner_satellite_id", "power_status", "scheduler_algorithm", "power_allocator"}


class DiagnosticDataError(ValueError):
    pass


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else math.nan


def read_trace_batches(path, batch_size=5000):
    """Yield dictionaries in bounded batches. Prefer Parquet when available."""
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
            yield batch.to_pylist()
    else:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = []
            for row in csv.DictReader(handle):
                rows.append(row)
                if len(rows) == batch_size:
                    yield rows
                    rows = []
            if rows:
                yield rows


def trace_path(directory, name):
    for suffix in (".parquet", ".csv"):
        path = directory / (name + suffix)
        if path.exists(): return path
    raise DiagnosticDataError(f"Missing {directory / name}. Rerun paper-results --save-group-trace into a NEW output directory.")


def _header(path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).schema_arrow.names
    with path.open(encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle), [])


def _import_trace(db, table, path, required, policy):
    columns = _header(path)
    missing = required - set(columns)
    if missing:
        raise DiagnosticDataError(f"Old/incomplete trace {path}: missing {sorted(missing)}. "
                                  "Rerun paper-results --save-group-trace with this version into a NEW output directory; old fields cannot be reconstructed.")
    if any(not c.replace("_", "").isalnum() for c in columns):
        raise DiagnosticDataError("Unsupported trace column name")
    textual = TEXT_FIELDS | EXTRA_TEXT
    definition = ",".join(f'"{c}" {"TEXT" if c in textual else "INTEGER" if c in INTEGER_FIELDS else "REAL"}' for c in columns)
    db.execute(f"CREATE TABLE IF NOT EXISTS {table} ({definition})")
    if set(columns) != {r[1] for r in db.execute(f"PRAGMA table_info({table})")}:
        raise DiagnosticDataError("Policies have incompatible diagnostic schemas; rerun all three policies together.")
    names = ",".join(f'"{c}"' for c in columns)
    sql = f"INSERT INTO {table} ({names}) VALUES ({','.join('?' for _ in columns)})"
    count = 0
    for batch in read_trace_batches(path):
        converted = []
        for row in batch:
            if row.get("policy") != policy:
                raise DiagnosticDataError(f"Wrong policy in {path}")
            if table == "g" and int(float(row["diagnostic_schema_version"])) != DIAGNOSTIC_SCHEMA_VERSION:
                raise DiagnosticDataError("Unsupported group diagnostic schema; rerun with this code version.")
            values = []
            for c in columns:
                value = row.get(c)
                if c in textual:
                    values.append("" if value is None else str(value))
                elif value is None or value == "":
                    values.append(None)
                else:
                    number = float(value)
                    values.append((int(number) if c in INTEGER_FIELDS else number) if math.isfinite(number) else None)
            converted.append(values)
        db.executemany(sql, converted)
        count += len(converted)
        db.commit()
    if not count: raise DiagnosticDataError(f"Empty trace: {path}")
    return count


def export_query(db, path, sql, parameters=()):
    cursor = db.execute(sql, parameters)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([c[0] for c in cursor.description])
        while rows := cursor.fetchmany(5000): writer.writerows(rows)


def numeric_stats(db, table, column, where="1", parameters=()):
    values = np.fromiter((r[0] for r in db.execute(
        f'SELECT "{column}" FROM {table} WHERE {where} AND "{column}" IS NOT NULL', parameters)), dtype=float)
    if not len(values):
        return {name: math.nan for name in ("mean", "median", "p10", "p95", "p90", "sum")}
    q10, median, q90, q95 = np.quantile(values, [.1, .5, .9, .95])
    return dict(mean=float(values.mean()), median=float(median), p10=float(q10), p90=float(q90),
                p95=float(q95), sum=float(values.sum()))


def _keys(db, table, columns):
    return db.execute(f"SELECT DISTINCT {','.join(columns)} FROM {table} ORDER BY {','.join(columns)}").fetchall()


def build_group_summary(db, columns):
    """Exact full-data statistics. Failure denominators are explicit."""
    result = []
    for key in _keys(db, "g", columns):
        where = " AND ".join(f"{c}=?" for c in columns)
        counts = db.execute(f"SELECT COUNT(*),SUM(scheduled),SUM(transmission_attempted),SUM(success) FROM g WHERE {where}", key).fetchone()
        n, scheduled, transmitted, success = counts
        categories = dict(db.execute(f"SELECT outcome_category,COUNT(*) FROM g WHERE {where} GROUP BY outcome_category", key))
        row = dict(zip(columns, key))
        row.update(candidate_groups=n, scheduled_groups=scheduled, transmitted_groups=transmitted,
                   scheduling_rate=ratio(scheduled, n), success_count=success, success_rate=ratio(success, n),
                   power_no_tx_count=categories.get(ALLOWED_OUTCOMES[3], 0),
                   rate_fail_count=categories.get(ALLOWED_OUTCOMES[4], 0),
                   sic_fail_count=categories.get(ALLOWED_OUTCOMES[5], 0),
                   power_no_tx_rate=ratio(categories.get(ALLOWED_OUTCOMES[3], 0), n),
                   transmitted_rate_failure_rate=ratio(categories.get(ALLOWED_OUTCOMES[4], 0), transmitted),
                   sic_propagated_failure_rate=ratio(categories.get(ALLOWED_OUTCOMES[5], 0), transmitted),
                   unscheduled_nonpositive_rate=ratio(categories.get(ALLOWED_OUTCOMES[0], 0), n),
                   algorithm2_cap_reject_rate=ratio(sum(categories.get(c, 0) for c in ALLOWED_OUTCOMES[1:3]), n))
        for category in ALLOWED_OUTCOMES: row[category + "_count"] = categories.get(category, 0)
        for column, name in (("gain_db", "gain_db"), ("max_elevation_deg", "elevation_deg"),
                             ("weighted_post_group_aoi_slots", "weighted_group_aoi")):
            stats = numeric_stats(db, "g", column, where, key)
            for stat in ("mean", "median", "p10", "p90", "p95", "sum"): row[f"{stat}_{name}"] = stats[stat]
        row["p50_gain_db"] = row["median_gain_db"]
        row["mean_weighted_post_group_aoi"] = row["mean_weighted_group_aoi"]
        row["p95_weighted_post_group_aoi"] = row["p95_weighted_group_aoi"]
        row["rate_failure_rate"] = row["transmitted_rate_failure_rate"]
        row["sic_failure_rate"] = row["sic_propagated_failure_rate"]
        for column, name in (("power", "power"), ("rate_margin_bps_hz", "rate_margin"),
                             ("sinr", "sinr"), ("rate_bps_hz", "rate")):
            row["mean_" + name] = db.execute(f"SELECT AVG({column}) FROM g WHERE {where} AND transmission_attempted=1", key).fetchone()[0]
        row.update(count=n, fraction_of_all_candidates=ratio(n, db.execute("SELECT COUNT(*) FROM g WHERE policy=?", (row["policy"],)).fetchone()[0]),
                   fraction_of_scheduled=ratio(scheduled, db.execute("SELECT SUM(scheduled) FROM g WHERE policy=?", (row["policy"],)).fetchone()[0]))
        result.append(row)
    return result


def _serving_stats(db, policy, action=None):
    where = "policy=?" + (" AND action=?" if action is not None else "")
    args = (policy,) if action is None else (policy, action)
    n, slots, decisions, power = db.execute(f"SELECT COUNT(*),COUNT(DISTINCT slot),SUM(is_regrouping_slot),SUM(power_infeasible) FROM c WHERE {where}", args).fetchone()
    row = dict(slot_count=slots, serving_slot_count=n, decision_count=decisions,
               power_infeasible_serving_slot_rate=ratio(power, n))
    for src, name in (("candidate_edge_groups", "candidate_edge"), ("candidate_nonedge_groups", "candidate_nonedge"),
                      ("scheduled_edge_groups", "scheduled_edge"), ("scheduled_nonedge_groups", "scheduled_nonedge")):
        row["mean_" + name] = db.execute(f"SELECT SUM({src}) FROM c WHERE {where}", args).fetchone()[0] / slots
    values = np.fromiter((r[0] for r in db.execute(f"SELECT MAX(aggregated_weighted_sum_aoi_slots) FROM c WHERE {where} GROUP BY slot ORDER BY slot", args)), float)
    row.update(mean_aggregated_weighted_sum_aoi=float(values.mean()),
               median_aggregated_weighted_sum_aoi=float(np.median(values)),
               p95_aggregated_weighted_sum_aoi=float(np.quantile(values, .95)),
               last20pct_aggregated_weighted_sum_aoi=float(values[-max(1, math.ceil(len(values)*.2)):].mean()))
    return row


def _resolve_root(outputs, seed, slots):
    root = Path(outputs)
    if slots is not None: return root / "debug" / f"slots_{slots}_seed_{seed}"
    if (root / "runs").is_dir(): return root
    candidates = sorted(p for p in (root / "debug").glob(f"slots_*_seed_{seed}") if (p / "runs").is_dir())
    if len(candidates) == 1: return candidates[0]
    raise DiagnosticDataError("No unambiguous completed comparison. First run paper-results --save-group-trace; "
                              "for several debug runs, pass --slots N or the exact debug directory.")


def _load_database(root, db, seed):
    summaries = []
    for policy in POLICIES:
        directory = root / "runs" / policy
        path = directory / "summary.json"
        if not path.exists(): raise DiagnosticDataError(f"Missing {path}. First run paper-results --save-group-trace.")
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("channel_seed") != seed: raise DiagnosticDataError(f"Seed mismatch: {path}")
        if summary.get("status") not in {"COMPLETED", "COMPLETED_WITH_POWER_OUTAGES", "REWARD_RANGE_MISMATCH"}:
            raise DiagnosticDataError(f"Incomplete/invalid run: {path}")
        summaries.append(summary)
        _import_trace(db, "g", trace_path(directory, "group_trace"), REQUIRED_GROUP_FIELDS, policy)
        _import_trace(db, "c", trace_path(directory, "slot_metrics"), REQUIRED_SERVING_FIELDS, policy)
    for key in ("environment_trace_hash", "channel_trace_hash"):
        values = [s.get(key) for s in summaries]
        if not all(values) or len(set(values)) != 1:
            raise DiagnosticDataError(f"Policies do not share a verified {key}")
    if len({s["slots_completed"] for s in summaries}) != 1: raise DiagnosticDataError("Unequal policy horizons")
    db.executescript("""
        CREATE UNIQUE INDEX c_identity ON c(policy,slot);
        CREATE UNIQUE INDEX g_identity ON g(policy,slot,message_id,group_id);
        CREATE INDEX g_edge ON g(policy,is_edge,outcome_category);
        CREATE INDEX g_elevation ON g(policy,elevation_bin,is_edge);
        CREATE INDEX g_action ON g(policy,action);
    """)
    if db.execute("SELECT COUNT(*) FROM g WHERE outcome_category NOT IN (" + ",".join("?" for _ in ALLOWED_OUTCOMES) + ")", ALLOWED_OUTCOMES).fetchone()[0]:
        raise DiagnosticDataError("Unknown outcome category")
    invalid = db.execute("""SELECT COUNT(*) FROM
        (SELECT policy,slot,satellite_id,COUNT(*) n,SUM(weighted_post_group_aoi_slots) a FROM g GROUP BY policy,slot,satellite_id) x
        LEFT JOIN c USING(policy,slot,satellite_id)
        WHERE c.policy IS NULL OR x.n!=c.candidate_groups OR ABS(x.a-c.weighted_aoi_slots)>1e-7""").fetchone()[0]
    if invalid: raise DiagnosticDataError("Group AoI/counts do not reconcile with serving metrics")
    invalid_sic = db.execute("SELECT COUNT(*) FROM g WHERE outcome_category='SIC_PROPAGATED_FAILURE' AND sic_first_failed_predecessor_group_id IS NULL").fetchone()[0]
    if invalid_sic: raise DiagnosticDataError("SIC propagation is missing its first failed predecessor")
    invalid_flags=db.execute("""SELECT COUNT(*) FROM g WHERE
        (outcome_category='POWER_ALLOCATION_INFEASIBLE_NO_TX' AND (scheduled!=1 OR transmission_attempted!=0 OR power!=0)) OR
        (outcome_category='TRANSMITTED_RATE_BELOW_THRESHOLD' AND (transmission_attempted!=1 OR rate_pass!=0)) OR
        (outcome_category='SIC_PROPAGATED_FAILURE' AND (transmission_attempted!=1 OR rate_pass!=1 OR sic_pass!=0))""").fetchone()[0]
    if invalid_flags: raise DiagnosticDataError("Outcome categories contradict transmission / PHY flags")
    for s in summaries:
        p=s["policy"]; expected=s["slots_completed"]
        count,first,last=db.execute("SELECT COUNT(DISTINCT slot),MIN(slot),MAX(slot) FROM c WHERE policy=?",(p,)).fetchone()
        if (count,first,last)!=(expected,0,expected-1): raise DiagnosticDataError("Incomplete slot range")
        orphan=db.execute("SELECT COUNT(*) FROM c WHERE policy=? AND NOT EXISTS (SELECT 1 FROM g WHERE g.policy=c.policy AND g.slot=c.slot AND g.satellite_id=c.satellite_id)",(p,)).fetchone()[0]
        if orphan: raise DiagnosticDataError("Missing group opportunities for a serving-slot")
    return summaries


class _Quantile:
    def __init__(self): self.values=[]
    def step(self, value):
        if value is not None: self.values.append(value)
    def finalize(self): return float(np.quantile(self.values, self.q)) if self.values else None


def _additional_outputs(db, out):
    for name, q in (("q10", .1), ("q50", .5), ("q90", .9), ("q95", .95)):
        aggregate=type(name, (_Quantile,), {"q":q}); db.create_aggregate(name,1,aggregate)
    export_query(db,out/"group_time_summary.csv", """SELECT policy,slot,MIN(time_s) time_s,is_edge,COUNT(*) candidate_groups,
        SUM(scheduled) scheduled_groups,SUM(transmission_attempted) transmitted_groups,
        q10(gain_db) p10_gain_db,q50(gain_db) median_gain_db,q90(gain_db) p90_gain_db,
        SUM(weighted_post_group_aoi_slots) weighted_aoi FROM g GROUP BY policy,slot,is_edge ORDER BY policy,slot,is_edge""")
    export_query(db,out/"top_aoi_group_events.csv","SELECT * FROM g WHERE aoi_rank_in_system<=5 ORDER BY policy,slot,satellite_id,aoi_rank_in_system")
    export_query(db,out/"group_outcomes.csv","SELECT * FROM g ORDER BY policy,slot,satellite_id,message_id,group_id")
    export_query(db,out/"worst_rate_failure_groups.csv", "SELECT * FROM g WHERE outcome_category='TRANSMITTED_RATE_BELOW_THRESHOLD' ORDER BY rate_margin_bps_hz ASC LIMIT 100")
    episodes=[]
    for p in POLICIES:
        for slot,aoi in db.execute("SELECT slot,MAX(aggregated_weighted_sum_aoi_slots) a FROM c WHERE policy=? GROUP BY slot ORDER BY a DESC,slot LIMIT 50",(p,)):
            serving_rows=[dict(zip([x[0] for x in cur.description],r)) for cur in [db.execute("SELECT * FROM c WHERE policy=? AND slot=? ORDER BY satellite_id",(p,slot))] for r in cur]
            groups=[dict(zip([x[0] for x in cur.description],r)) for cur in [db.execute("SELECT * FROM g WHERE policy=? AND slot=? ORDER BY weighted_post_group_aoi_slots DESC LIMIT 5",(p,slot))] for r in cur]
            mechanism=db.execute("SELECT SUM(success),SUM(outcome_category='TRANSMITTED_RATE_BELOW_THRESHOLD'),SUM(outcome_category='SIC_PROPAGATED_FAILURE') FROM g WHERE policy=? AND slot=?",(p,slot)).fetchone()
            gains=db.execute("SELECT satellite_id,AVG(gain_db),MIN(gain_db),MAX(gain_db) FROM g WHERE policy=? AND slot=? GROUP BY satellite_id",(p,slot)).fetchall()
            episodes.append(dict(policy=p,slot=slot,time_s=serving_rows[0]["time_s"],aggregated_weighted_sum_aoi=aoi,
                action_by_serving_satellite=json.dumps({c['satellite_id']:c['action'] for c in serving_rows}),
                candidate_edge_groups=sum(c['candidate_edge_groups'] for c in serving_rows),
                candidate_nonedge_groups=sum(c['candidate_nonedge_groups'] for c in serving_rows),
                scheduled_edge_groups=sum(c['scheduled_edge_groups'] for c in serving_rows),
                scheduled_nonedge_groups=sum(c['scheduled_nonedge_groups'] for c in serving_rows),
                power_infeasible_serving=sum(c['power_infeasible'] for c in serving_rows),
                successful_groups=mechanism[0],rate_failures=mechanism[1],sic_failures=mechanism[2],
                serving_gain_statistics=json.dumps([dict(zip(['satellite_id','mean_gain_db','min_gain_db','max_gain_db'],g)) for g in gains]),
                serving_details=json.dumps(serving_rows),top5_contributing_groups=json.dumps(groups)))
    write_csv(out/"worst_aoi_episodes.csv",episodes)
    cur=db.execute("SELECT * FROM c WHERE power_infeasible=1 ORDER BY policy,slot,satellite_id")
    with (out/"worst_power_failure_episodes.csv").open("w",encoding="utf-8",newline="") as handle:
        names=[c[0] for c in cur.description]+["scheduled_group_details"]
        writer=csv.writer(handle);writer.writerow(names)
        for values in cur:
            row=dict(zip(names,values)); groups=db.execute("SELECT message_id,group_id,gain_db,is_edge,target_power,power_floor FROM g WHERE policy=? AND slot=? AND satellite_id=? AND scheduled=1 ORDER BY gain_db,message_id,group_id",(row["policy"],row["slot"],row["satellite_id"])).fetchall()
            writer.writerow(list(values)+[json.dumps([dict(zip(['message_id','group_id','gain_db','is_edge','target_power','power_floor'],g)) for g in groups])])
    message_rows=[]
    for p,message in _keys(db,"g",["policy","message_id"]):
        values=np.fromiter((r[0] for r in db.execute("SELECT SUM(weighted_post_group_aoi_slots) FROM g WHERE policy=? AND message_id=? GROUP BY slot ORDER BY slot",(p,message))),float)
        total=db.execute("SELECT SUM(weighted_post_group_aoi_slots) FROM g WHERE policy=?",(p,)).fetchone()[0]
        message_rows.append(dict(policy=p,message_id=message,mean=float(values.mean()),median=float(np.median(values)),
            p95=float(np.quantile(values,.95)),fraction_of_total_weighted_aoi=ratio(values.sum(),total)))
    write_csv(out/"weighted_aoi_by_message_summary.csv",message_rows)
    export_query(db,out/"power_composition_summary.csv","SELECT policy,scheduled_edge_groups,scheduled_nonedge_groups,COUNT(*) sample_count,AVG(power_infeasible) power_infeasible_rate FROM c GROUP BY policy,scheduled_edge_groups,scheduled_nonedge_groups")


def _table(rows, fields):
    def show(v): return f"{v:.6g}" if isinstance(v,float) else str(v)
    return ["| "+" | ".join(fields)+" |","| "+" | ".join("---" for _ in fields)+" |"] + ["| "+" | ".join(show(r.get(k,"")) for k in fields)+" |" for r in rows]


def write_diagnostic_report(db, out, summaries, policy_rows, action_rows, environment, seed, plots):
    other=db.execute("SELECT COUNT(*) FROM g WHERE outcome_category='OTHER_FAILURE'").fetchone()[0]
    lines=["# MIRA-LEO poor-result diagnostics", "",f"Seed / realization: {seed}. Observational associations only; no controlled ablation or universal causal claim.",
        "diagnostics do not change the simulation policy or RNG.",
        "Statistics and quantiles use ALL rows. Scatter plots alone use post-run sampling (seed=0, at most 100,000 points per policy).",
        "Elevation bins use group MAX elevation; [60,90] is inclusive. CSI quintiles share pooled full-data thresholds across all policies.",
        "CSI is worst-user instantaneous amplitude squared (power gain); dB=10 log10(gain). Edge means MAX member elevation <20 degrees, not a CSI threshold.",
        "Missing/zero denominators are NaN. PHY statistics use transmitted opportunities; no-TX rate margins are undefined.",
        "Action-conditioned comparisons are serving-slot observations at different states/times, not counterfactual action effects.",
        "Environment / channel hashes verified equal: "+json.dumps(environment),
        f"OTHER_FAILURE count: {other}. Status: {'WARNING_OTHER_FAILURE' if other else 'ATTRIBUTION_COMPLETE'}.",
        "", "## Overall AoI and scheduling", ""]
    lines+=_table(policy_rows,["policy","mean_aggregated_weighted_sum_aoi","p95_aggregated_weighted_sum_aoi","last20pct_aggregated_weighted_sum_aoi","mean_scheduled_edge_groups","mean_scheduled_nonedge_groups","edge_schedule_rate","nonedge_schedule_rate"])
    lines += ["", "## Power allocation, own-rate failure and SIC propagation", ""]
    lines+=_table(policy_rows,["policy","power_infeasible_serving_slot_rate","power_infeasible_group_opportunity_rate","transmitted_rate_failure_rate","sic_propagated_failure_rate","unscheduled_nonpositive_rate","algorithm2_cap_reject_rate"])
    lines += ["", "Power composition and interleaving (full serving-slot counts):", ""]
    cur=db.execute("SELECT policy,csi_order_class_switch_count,COUNT(*) sample_count,AVG(power_infeasible) failure_rate FROM c GROUP BY policy,csi_order_class_switch_count ORDER BY policy,csi_order_class_switch_count")
    lines+=_table([dict(zip([d[0] for d in cur.description],r)) for r in cur],["policy","csi_order_class_switch_count","sample_count","failure_rate"])
    lines += ["", "See worst_power_failure_episodes.csv for each plan's budgets, floors, class order and per-group target power.",
              "", "## CSI and elevation", ""]
    lines+=_table(policy_rows,["policy","mean_group_gain_db_edge","mean_group_gain_db_nonedge","p10_group_gain_db_edge","p10_group_gain_db_nonedge","p50_group_gain_db_edge","p50_group_gain_db_nonedge","mean_max_elevation_edge","mean_max_elevation_nonedge"])
    with (out/"policy_comparison_conditioned_on_elevation.csv").open(encoding="utf-8",newline="") as f:
        lines+=_table(list(csv.DictReader(f)),["policy","elevation_bin","mean_gain_db","power_no_tx_rate","transmitted_rate_failure_rate","sic_propagated_failure_rate","success_rate","mean_weighted_group_aoi"])
    lines += ["", "## AoI contribution by message and outcome", ""]
    cur=db.execute("SELECT policy,outcome_category,COUNT(*) group_opportunities,SUM(weighted_post_group_aoi_slots) weighted_aoi_total FROM g GROUP BY policy,outcome_category ORDER BY policy,weighted_aoi_total DESC")
    attribution=[dict(zip([d[0] for d in cur.description],r)) for r in cur]
    lines+=_table(attribution,["policy","outcome_category","group_opportunities","weighted_aoi_total"])
    with (out/"weighted_aoi_by_message_summary.csv").open(encoding="utf-8",newline="") as f:
        lines+=_table(list(csv.DictReader(f)),["policy","message_id","mean","median","p95","fraction_of_total_weighted_aoi"])
    lines += ["", "## Top 20 group-slot AoI contributors", ""]
    cols="policy,slot,satellite_id,satellite_id,message_id,message_weight,action,is_edge,max_elevation_deg,gain_db,scheduled,power,outcome_category,weighted_post_group_aoi_slots"
    cur=db.execute(f"SELECT {cols} FROM g ORDER BY weighted_post_group_aoi_slots DESC,policy,slot LIMIT 20")
    lines+=_table([dict(zip(cols.split(','),r)) for r in cur],cols.split(','))
    lines += ["", "## Action occupancy and conditional outcomes", ""]
    lines+=_table(action_rows,["policy","action","decision_count","slot_count","mean_aggregated_weighted_sum_aoi","power_infeasible_serving_slot_rate","transmitted_rate_failure_rate","sic_propagated_failure_rate","success_rate"])
    lines += ["", "## Observed differences from baselines", ""]
    mira=next(r for r in policy_rows if r["policy"]==POLICIES[0])
    for baseline in policy_rows[1:]:
        lines.append(f"- In this realization MIRA minus {baseline['policy']}: mean AoI {mira['mean_aggregated_weighted_sum_aoi']-baseline['mean_aggregated_weighted_sum_aoi']:.6g}; "
                     f"power-no-TX opportunity rate {mira['power_infeasible_group_opportunity_rate']-baseline['power_infeasible_group_opportunity_rate']:.6g}; "
                     f"own-rate failure rate {mira['transmitted_rate_failure_rate']-baseline['transmitted_rate_failure_rate']:.6g}; "
                     f"SIC propagation rate {mira['sic_propagated_failure_rate']-baseline['sic_propagated_failure_rate']:.6g}.")
    failures=[r for r in attribution if r['policy']==POLICIES[0] and r['outcome_category']!='SUCCESS']
    if failures:
        leading=max(failures,key=lambda r:r['weighted_aoi_total'])
        lines.append(f"Largest observed MIRA AoI mass among failure categories: {leading['outcome_category']} ({leading['weighted_aoi_total']:.6g}). This local attribution does not prove a policy-level causal effect.")
    else: lines.append("No failed MIRA group opportunities were observed in this trace.")
    lines += ["", "Run statuses / reward clipping (unchanged): "+json.dumps({s['policy']:{'status':s['status'],'reward_clipping_rate_max':s.get('reward_clipping_rate_max')} for s in summaries}),
              "", "## Generated plots", ""]+[f"- [{p}]({p})" for p in plots]
    (out/"diagnostic_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def generate_diagnostics(config, output_directory, *, seed=0, slots=None):
    root=_resolve_root(output_directory,seed,slots)
    # Preflight before writing anything: old published results remain untouched.
    for p in POLICIES:
        path=trace_path(root/'runs'/p,'group_trace')
        if REQUIRED_GROUP_FIELDS-set(_header(path)):
            raise DiagnosticDataError(f"{path} lacks new AoI/CSI diagnostics. Rerun paper-results --save-group-trace into a NEW output directory.")
    out=root/'formal_results'/'diagnostics';out.mkdir(parents=True,exist_ok=True)
    database=out/'diagnostic_data.sqlite'
    if database.exists(): database.unlink()
    db=sqlite3.connect(database)
    db.execute("PRAGMA cache_size=-65536");db.execute("PRAGMA temp_store=FILE")
    try:
        summaries=_load_database(root,db,seed)
        values=np.fromiter((r[0] for r in db.execute("SELECT gain_db FROM g WHERE gain_db IS NOT NULL")),float)
        boundaries=np.quantile(values,[.2,.4,.6,.8]).tolist();del values
        db.execute("ALTER TABLE g ADD COLUMN gain_db_bin TEXT")
        db.execute("UPDATE g SET gain_db_bin=CASE WHEN gain_db<? THEN 'Q0-Q20' WHEN gain_db<? THEN 'Q20-Q40' WHEN gain_db<? THEN 'Q40-Q60' WHEN gain_db<? THEN 'Q60-Q80' ELSE 'Q80-Q100' END",boundaries)
        db.execute("CREATE INDEX g_csi ON g(policy,gain_db_bin)");db.commit()
        failure=build_group_summary(db,["policy","is_edge","outcome_category"])
        elevation=build_group_summary(db,["policy","elevation_bin","is_edge"])
        action=build_group_summary(db,["policy","action"])
        overall=build_group_summary(db,["policy"])
        by_edge=build_group_summary(db,["policy","is_edge"])
        for row in action: row.update(_serving_stats(db,row['policy'],row['action']))
        for row in overall:
            row.update(_serving_stats(db,row['policy']))
            row['power_infeasible_group_opportunity_rate']=row['power_no_tx_rate']
            for edge,label in ((1,'edge'),(0,'nonedge')):
                sub=next((s for s in by_edge if s['policy']==row['policy'] and s['is_edge']==edge),{})
                row[f'{label}_schedule_rate']=sub.get('scheduling_rate',math.nan)
                for src,dst in [('mean_gain_db','mean_group_gain_db'),('p10_gain_db','p10_group_gain_db'),('p50_gain_db','p50_group_gain_db'),('mean_elevation_deg','mean_max_elevation'),('mean_power','mean_allocated_power'),('mean_rate_margin','mean_rate_margin'),('mean_weighted_group_aoi','mean_weighted_group_aoi')]:
                    row[dst+'_'+label]=sub.get(src,math.nan)
                for src in ['candidate','scheduled']: row[f'mean_{src}_{label}_groups']=row[f'mean_{src}_{label}']
        overall.sort(key=lambda r:POLICIES.index(r['policy']))
        for name,rows in [('failure_attribution_summary',failure),('elevation_bin_summary',elevation),('action_diagnostic_summary',action),('policy_comparison_conditioned_on_action',action),('policy_diagnostic_summary',overall),('edge_nonedge_scheduling_summary',by_edge),('policy_comparison_conditioned_on_elevation',build_group_summary(db,['policy','elevation_bin'])),('policy_comparison_conditioned_on_csi',build_group_summary(db,['policy','gain_db_bin']))]:
            write_csv(out/(name+'.csv'),rows)
        _additional_outputs(db,out)
        from .diagnostic_plots import generate_plots
        plots=generate_plots(db,out,config)
        environment={key:summaries[0][key] for key in ('environment_trace_hash','channel_trace_hash')}
        write_diagnostic_report(db,out,summaries,overall,action,environment,seed,plots)
        other=db.execute("SELECT COUNT(*) FROM g WHERE outcome_category='OTHER_FAILURE'").fetchone()[0]
        result=dict(status='WARNING_OTHER_FAILURE' if other else 'DIAGNOSTICS_COMPLETED',seed=seed,
            source_root=str(root),diagnostics_directory=str(out),diagnostic_schema_version=DIAGNOSTIC_SCHEMA_VERSION,
            environment_identity_equal=True,environment=environment,pooled_gain_db_quintile_boundaries=boundaries,
            gain_db_quantile_note='pooled full-data thresholds; ties go to the upper bin',
            scatter_sampling_seed=0,scatter_max_points_per_policy=100000,other_failure_count=other,plots=plots)
        atomic_write_json(out/'diagnostic_summary.json',result)
        return result
    finally:
        db.close()
