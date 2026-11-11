"""Read-only instrumentation. No random draws and no policy decisions."""
from __future__ import annotations

import hashlib
import math

import numpy as np

DIAGNOSTIC_SCHEMA_VERSION = 3
ALLOWED_OUTCOMES = (
    "UNSCHEDULED_NONPOSITIVE_NONEDGE", "UNSCHEDULED_ALGORITHM2_CAP_EDGE",
    "UNSCHEDULED_ALGORITHM2_CAP_NONEDGE", "POWER_ALLOCATION_INFEASIBLE_NO_TX",
    "TRANSMITTED_RATE_BELOW_THRESHOLD", "SIC_PROPAGATED_FAILURE",
    "ROUTE_UNREACHABLE", "NOT_VISIBLE_OR_COVERAGE_FAILURE", "HANDOVER_BLACKOUT",
    "SUCCESS", "OTHER_FAILURE",
)
ELEVATION_EDGES = (10., 15., 20., 25., 30., 40., 50., 60., 90.)


def gain_db(gain):
    return 10. * math.log10(max(float(gain), np.finfo(float).tiny))


def elevation_bin(value):
    if value < 10.: return "below_10"
    for index, (left, right) in enumerate(zip(ELEVATION_EDGES, ELEVATION_EDGES[1:])):
        if value < right or (index == 7 and value <= right):
            return f"{left:g}-{right:g}"
    return "above_90"


def csi_ranks(groups):
    return {g.key: rank for rank, g in enumerate(
        sorted(groups, key=lambda g: (g.gain, g.message_id, g.group_id)), 1)}


def classify_group_outcome(group, diagnostic, *, edge_count=0, route_reachable=True,
                           visible=True, handover_blackout=False):
    """Exclusive precedence: scheduling, no-TX, own rate, SIC, delivery gates."""
    if not group.scheduled:
        if not group.is_edge and group.priority <= 0:
            return ALLOWED_OUTCOMES[0]
        if group.is_edge:
            return (ALLOWED_OUTCOMES[1] if edge_count > group.metadata.get("algorithm2_U", math.inf)
                    else "OTHER_FAILURE")
        return ALLOWED_OUTCOMES[2]
    if diagnostic.get("transmission_failure", False): return ALLOWED_OUTCOMES[3]
    if diagnostic.get("transmission_attempted", False):
        if not group.rate_pass: return ALLOWED_OUTCOMES[4]
        if not group.sic_pass: return ALLOWED_OUTCOMES[5]
    if not route_reachable: return "ROUTE_UNREACHABLE"
    if not visible: return "NOT_VISIBLE_OR_COVERAGE_FAILURE"
    if handover_blackout: return "HANDOVER_BLACKOUT"
    if group.success: return "SUCCESS"
    return "OTHER_FAILURE"


def serving_power_diagnostics(groups, diagnostic):
    scheduled = sorted((g for g in groups if g.scheduled),
                       key=lambda g: (g.gain, g.message_id, g.group_id))
    sequence = ["E" if g.is_edge else "N" for g in scheduled]
    row = {
        "power_infeasible": int(diagnostic.get("transmission_failure", False)),
        "power_failure_reason": diagnostic.get("power_infeasible_reason",diagnostic["status"]) if diagnostic.get("transmission_failure") else "",
        "lp_solver_status": diagnostic.get("projection_solver_status") if diagnostic.get("projection_solver_status") is not None else -1,
        "lp_solver_message": diagnostic.get("projection_solver_message") or "",
        "weakest_scheduled_group_class": sequence[0] if sequence else "",
        "strongest_scheduled_group_class": sequence[-1] if sequence else "",
        "csi_order_class_sequence": " ".join(sequence),
        "csi_order_class_switch_count": sum(a != b for a, b in zip(sequence, sequence[1:])),
    }
    for label, edge in (("edge", True), ("nonedge", False)):
        values = [gain_db(g.gain) for g in scheduled if g.is_edge == edge]
        row[f"weakest_{label}_gain_db"] = min(values, default=math.nan)
        row[f"strongest_{label}_gain_db"] = max(values, default=math.nan)
    return row


def enrich_group_diagnostic_rows(config, groups, diagnostic, *, ages_after, amplitudes,
                                elevations, visible, route_reachable, handover_blackout,
                                system_weighted_aoi, policy, slot, satellite_id, action):
    """Read current arrays and scheduler metadata after the existing AoI update."""
    ranks = csi_ranks(groups)
    ordered = sorted(groups, key=lambda g: ranks[g.key])
    boundary_order = [g.metadata.get("boundary_group_csi_rank", ranks[g.key]) for g in ordered]
    inversions = sum(a > b for i, a in enumerate(boundary_order) for b in boundary_order[i + 1:])
    rows = []
    phi = float(config.get("algorithm.phi_theta_deg"))
    threshold = float(config.get("algorithm.R_theta"))
    weights = config.get("services.weights")
    interval = int(config.get("derived.regrouping_interval_slots"))
    dt = float(config.get("experiment.slot_dt_s"))
    serving_row = serving_power_diagnostics(groups, diagnostic)
    members = np.concatenate([g.member_indices for g in groups])
    sat = groups[0].satellite_index
    elev_all = elevations[sat, members]
    serving_row.update(serving_elevation_min_deg=float(np.min(elev_all)),
                      serving_elevation_mean_deg=float(np.mean(elev_all)),
                      serving_elevation_max_deg=float(np.max(elev_all)),
                      edge_fraction=sum(g.is_edge for g in groups) / len(groups),
                      group_order_inversion_count=inversions)
    for group in groups:
        members = group.member_indices
        amp = amplitudes[sat, members]
        elev = elevations[sat, members]
        worst = int(np.argmin(amp))
        post = ages_after[members]
        weight = float(weights[group.message_id])
        pre = group.metadata["pre_group_aoi_sum_slots"]
        total = float(np.sum(post))
        db = gain_db(group.gain)
        boundary_gain = group.metadata.get("boundary_group_gain", group.gain)
        boundary_rank = group.metadata.get("boundary_group_csi_rank", ranks[group.key])
        floor = diagnostic["floors"].get(id(group), 0.)
        attempted = bool(group.scheduled and diagnostic.get("transmission_attempted", False))
        class_budget = diagnostic.get(group.metadata.get("power_pool","")+"_budget", diagnostic["edge_budget" if group.is_edge else "nonedge_budget"])
        row = {
            "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
            "policy": policy, "slot": slot, "time_s": slot * dt,
            "regrouping_epoch": slot // interval, "is_regrouping_slot": int(slot % interval == 0),
            "satellite_id": satellite_id, "action": action,
            "message_id": group.message_id, "message_weight": weight, "group_id": group.group_id,
            "member_hash": hashlib.sha256(members.astype("<i4").tobytes()).hexdigest()[:16],
            "group_size": len(members), "is_edge": int(group.is_edge),
            "geo_edge": int(group.is_edge),
            "power_pool": group.metadata.get("power_pool", "unscheduled"),
            "power_pool_promoted": int(group.metadata.get("power_pool_promoted",False)),
            "power_pool_csi_reference_gain": group.metadata.get("power_pool_csi_reference_gain"),
            "global_csi_rank": group.metadata.get("global_csi_rank"),
            "pool_target_power": group.metadata.get("pool_target_power"),
            "pool_allocated_power": group.metadata.get("pool_allocated_power"),
            "gain": group.gain, "gain_db": db, "group_gain_db": db,
            "worst_user_channel_amplitude": float(amp[worst]),
            "worst_user_channel_power_gain": group.gain,
            "member_channel_amplitude_mean": float(np.mean(amp)),
            "member_channel_amplitude_max": float(np.max(amp)),
            "member_channel_amplitude_std": float(np.std(amp)),
            "worst_user_device_index": int(members[worst]), "worst_user_elevation_deg": float(elev[worst]),
            "elevation_min_deg": float(np.min(elev)), "elevation_mean_deg": float(np.mean(elev)),
            "elevation_max_deg": group.max_elevation_deg, "max_elevation_deg": group.max_elevation_deg,
            "elevation_std_deg": float(np.std(elev)), "fraction_members_below_phi": float(np.mean(elev < phi)),
            "elevation_bin": elevation_bin(group.max_elevation_deg),
            "boundary_group_gain": boundary_gain, "boundary_group_gain_db": gain_db(boundary_gain),
            "boundary_group_csi_rank": boundary_rank, "current_group_gain_db": db,
            "gain_change_from_boundary_db": db - gain_db(boundary_gain),
            "current_group_csi_rank": ranks[group.key],
            "csi_rank_changed_from_boundary": int(ranks[group.key] != boundary_rank),
            "milliseconds_since_regrouping": (slot % interval) * dt * 1000.,
            "scheduled": int(group.scheduled), "priority": group.priority,
            "scheduler_candidate": int(group.metadata["scheduler_candidate"]),
            "scheduler_rejection_reason": group.metadata["scheduler_rejection_reason"],
            "power": group.power, "target_power": diagnostic["target_powers"].get(id(group), 0.),
            "power_floor": floor, "power_margin_above_floor": group.power - floor,
            "power_over_floor_ratio": group.power / floor if floor > 0 else math.nan,
            "power_fraction_of_satellite": group.power / float(config.get("channel.normalized_satellite_power")),
            "power_fraction_of_class_budget": group.power / class_budget if class_budget else math.nan,
            "power_fraction_of_pool_budget": group.power / class_budget if class_budget else math.nan,
            "transmission_attempted": int(attempted),
            "transmission_failure_power": int(group.scheduled and diagnostic.get("transmission_failure", False)),
            "interference": group.interference,
            "external_interference_power": group.interference,
            "interferer_count_at_worst_ue": group.metadata.get("interferer_count_at_worst_ue"),
            "worst_interference_device_index": group.metadata.get("worst_interference_device_index"), "sinr": group.sinr, "rate_bps_hz": group.rate_bps_hz,
            "rate_bps": float(config.get("services.bandwidth_hz")) * group.rate_bps_hz,
            "R_theta_bps_hz": threshold,
            "rate_margin_bps_hz": group.rate_bps_hz - threshold if attempted else math.nan,
            "rate_pass": int(group.rate_pass), "sic_pass": int(group.sic_pass), "success": int(group.success),
            "route_reachable": int(route_reachable), "members_visible": int(np.all(visible[sat, members])),
            "handover_blackout": int(handover_blackout), "csi_slot_index": slot, "power_csi_slot_index": slot,
            "outcome_category": classify_group_outcome(group, diagnostic,
                edge_count=sum(g.is_edge for g in groups), route_reachable=route_reachable,
                visible=bool(np.all(visible[sat, members])), handover_blackout=handover_blackout),
            "post_group_aoi_sum_slots": total, "post_group_aoi_mean_slots": float(np.mean(post)),
            "post_group_aoi_max_slots": float(np.max(post)), "post_group_aoi_p95_slots": float(np.quantile(post, .95)),
            "weighted_post_group_aoi_slots": weight * total, "aoi_change_sum_slots": total - pre,
            "weighted_aoi_change_slots": weight * (total - pre),
            "system_weighted_aoi_post_slots": system_weighted_aoi,
            "group_weighted_aoi_fraction_of_system": weight * total / system_weighted_aoi if system_weighted_aoi else 0.,
        }
        for key in ("pre_group_aoi_sum_slots", "predicted_unsuccessful_next_aoi_sum_slots",
                    "predicted_successful_next_aoi_sum_slots", "gamma_slots", "weighted_pre_group_aoi_slots",
                    "weighted_predicted_unsuccessful_aoi_slots", "weighted_predicted_successful_aoi_slots"):
            row[key] = group.metadata[key]
        for suffix in ("message_id", "group_id", "gain_db", "power", "rate"):
            row["sic_first_failed_predecessor_" + suffix] = group.metadata.get("sic_first_failed_predecessor_" + suffix)
        rows.append(row)
    for rank, row in enumerate(sorted(rows, key=lambda r: (-r["weighted_post_group_aoi_slots"], r["message_id"], r["group_id"])), 1):
        row.update(aoi_rank_in_system=rank, is_top1_aoi_group=int(rank == 1),
                   is_top3_aoi_group=int(rank <= 3), is_top5_aoi_group=int(rank <= 5))
    return rows, serving_row


TEXT_FIELDS = {"power_pool", "pool_budget_mode", "power_pool_rule_version", "power_infeasible_reason","policy", "satellite_id", "member_hash", "scheduler_rejection_reason", "members",
               "external_interferer_power_by_satellite",
               "outcome_category", "elevation_bin", "power_failure_reason", "lp_solver_message",
               "weakest_scheduled_group_class", "strongest_scheduled_group_class", "csi_order_class_sequence",
               "membership_hash", "action_owner_satellite_id", "power_status", "scheduler_algorithm", "power_allocator"}


def parquet_diagnostic_types(columns):
    """Explicit nullable types prevent first-batch null inference failures."""
    integer_fields = {"slot", "message_id", "group_id", "group_size", "action", "geo_edge", "power_pool_promoted", "global_csi_rank",
        "regrouping_epoch", "is_regrouping_slot", "diagnostic_schema_version", "csi_slot_index", "power_csi_slot_index",
        "scheduled", "is_edge", "rate_pass", "sic_pass", "success", "transmission_attempted", "transmission_failure_power",
        "interferer_count_at_worst_ue", "worst_interference_device_index", "worst_user_device_index", "aoi_rank_in_system", "boundary_group_csi_rank", "current_group_csi_rank",
        "csi_rank_changed_from_boundary", "sic_first_failed_predecessor_message_id", "sic_first_failed_predecessor_group_id"}
    return {name: "string" if name in TEXT_FIELDS else "int64" if name in integer_fields else "float64" for name in columns}
