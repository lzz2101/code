from __future__ import annotations

import hashlib
import json
import math
import time
import tracemalloc
from array import array
from collections import Counter
from pathlib import Path

import numpy as np

from .algorithms import (
    RewardNormalizer,
    SlidingWindowUCB,
    allocate_power,
    decode_groups,
    finite_horizon_reward_bounds,
    raw_reward,
    schedule_groups,
    update_user_aoi,
)
from .association import AssociationTracker
from .channel import (
    external_interferer_powers, expected_cross_link_interference_by_ue,
    path_gain, serving_channel_amplitude,
)
from .config import FIXED_PHI_THETA_DEG, FIXED_B_PENALTY, StudyConfig, save_config
from .errors import BaselineUnavailableError, ModelConsistencyError
from .geometry import link_geometry
from .metrics import AssignmentTracker, advance_trace_hash, regrouping_overhead
from .isl import build_isl_graph, shortest_route, route_delay_s
from .scenario import StudyScenario
from .protected_power import POWER_MODE, POOL_RULE_VERSION
from .p1_metrics import p1_slot_value, objective_summary
from .geo_edge import snapshot_group_rows, geo_occupancy_row
from .regrouping import groups_for_slot, initial_epoch_state, observe_slot_reward
from .storage import AppendCsvWriter, atomic_pickle, atomic_write_json, csv_to_parquet, load_pickle, write_csv


POLICY_ALIASES = {
    "sw-ucb": "mira_sw_ucb",
    "fixed": "fixed_grouping",
}
AVAILABLE_POLICIES = {"mira_sw_ucb", "random_grouping", "fixed_grouping"}
BLOCKED_POLICIES = {
    "aim_static": "BLOCKED_REFERENCE_IMPLEMENTATION_MISSING",
    "ppo_multicast": "BLOCKED_REFERENCE_IMPLEMENTATION_MISSING",
    "ga_grouping": "BLOCKED_REFERENCE_IMPLEMENTATION_MISSING",
}
CHECKPOINT_SCHEMA_VERSION = 7


def canonical_policy_name(policy: str) -> str:
    name = POLICY_ALIASES.get(str(policy), str(policy))
    if name in BLOCKED_POLICIES:
        raise BaselineUnavailableError(f"{name}: {BLOCKED_POLICIES[name]}")
    if name not in AVAILABLE_POLICIES:
        allowed = sorted(AVAILABLE_POLICIES | set(BLOCKED_POLICIES))
        raise ValueError(f"unknown policy {policy!r}; expected one of {allowed}")
    return name


def _random_action(actions, policy_seed: int, slot: int, satellite: int) -> int:
    choices = tuple(int(value) for value in actions)
    sequence = np.random.SeedSequence([int(policy_seed), int(slot), int(satellite), 0x4D495241])
    rng = np.random.default_rng(sequence)
    return choices[int(rng.integers(0, len(choices)))]


def select_grouping_action(policy, *, actions, slot, satellite, policy_seed, fixed_action=4, agent=None):
    policy = canonical_policy_name(policy)
    if policy == "random_grouping":
        return _random_action(actions, policy_seed, slot, satellite)
    if policy == "fixed_grouping":
        if fixed_action not in actions:
            raise ValueError("fixed action must be legal")
        return int(fixed_action)
    if agent is None:
        raise ValueError("MIRA grouping requires its satellite SW-UCB agent")
    return agent.select(slot)


def serialize_power_diagnostic(groups, diagnostic):
    result = dict(diagnostic)
    labels = {id(g): ":".join(str(k) for k in g.key) for g in groups}
    for name in ("powers", "target_powers", "final_powers", "floors"):
        result[name] = {labels[key]: value for key, value in diagnostic.get(name, {}).items()}
    result["groups_by_current_csi"] = [
        {"key": g.key, "gain": g.gain, "is_edge": g.is_edge, "scheduled": g.scheduled}
        for g in sorted(groups, key=lambda item: (item.gain, item.message_id, item.group_id))
    ]
    return result


def build_parameter_snapshot(config: StudyConfig, policy: str, channel_seed: int) -> dict:
    snapshot = {
        "policy": policy,

        "total_ue": 120,
        "message_count": 3,
        "subscribers_per_message": int(config.get("services.subscribers_per_message")),
        "serving_satellite_count": 1,
        "interference_mode": "fixed_full_load",
        "interference_model": str(config.get("channel.external_interference")),
        "interferer_power_normalized": float(config.get("channel.normalized_satellite_power")),
        "aoi_scope": "evaluated_120_ue_system",
        "per_satellite_aoi_semantics": "120-UE contribution only at the current serving satellite",
        "fixed_channel_seed": int(channel_seed),
        "slot_dt_s": float(config.get("experiment.slot_dt_s")),
        "duration_s": float(config.get("experiment.duration_s")),
        "warmup_report_s": float(config.get("experiment.warmup_report_s")),
        "carrier_hz": float(config.get("channel.carrier_hz")),
        "scs_hz": float(config.get("radio.scs_hz")),
        "bandwidth_hz": float(config.get("services.bandwidth_hz")),
        "reference_bandwidth_hz": float(config.get("channel.reference_bandwidth_hz")),
        "R_theta_bps_hz": float(config.get("algorithm.R_theta")),
        "phi_min_deg": float(config.get("geometry.phi_min_deg")),
        "phi_theta_deg": float(config.get("algorithm.phi_theta_deg")),
        "B_penalty": float(config.get("algorithm.B_penalty")),
        "fixed_B": config.get("algorithm.B_penalty") == FIXED_B_PENALTY,
        "power_pool_rule_version": config.get("algorithm.power_pool_rule_version"),
        "infeasible_power_handling": str(config.get("algorithm.infeasible_power_handling", "error")),
        "reward_bounds": config.get("algorithm.reward_bounds"),
        "V_c": float(config.get("algorithm.V_c")),
        "V_nc": float(config.get("algorithm.V_nc")),
        "V_phi": float(config.get("algorithm.V_phi")),
        "normalized_satellite_power": float(config.get("channel.normalized_satellite_power")),
        "schedule_capacity_continuous": float(config.get("derived.schedule_capacity_continuous")),
        "schedule_capacity_integer": int(config.get("derived.schedule_capacity_integer")),
        "paper_approximation": int(config.get("derived.schedule_capacity_paper_approximation")),
        "messages_per_active_satellite": int(config.get("services.messages_per_active_satellite")),
        "weights": list(config.get("services.weights")),
        "action_set": list(config.get("algorithm.action_set")),
        "action_set_name": str(config.get("algorithm.action_set_name")),
        "regrouping_interval_s": float(config.get("algorithm.regrouping_interval_s")),
        "regrouping_interval_slots": int(config.get("derived.regrouping_interval_slots")),
        "sw_ucb_window_epochs": int(config.get("derived.sw_ucb_window_epochs")),
        "bandit_time_index": "global_regrouping_epoch",
        "reward_aggregation": "mean_per_slot_normalized_reward_within_epoch",
        "scheduler_algorithm": "Algorithm2_strict",
        "power_allocator": str(config.get("algorithm.power_mode")),
        "power_csi_timing": "current_slot_error_free_feedback_no_extra_delay",
        "capacity_semantics": str(config.get("derived.schedule_capacity_semantics")),
        "schedule_capacity_single_class_floor_only": config.get("derived.schedule_capacity_single_class_floor_only"),
        "schedule_capacity_edge_when_mixed_floor_only": config.get("derived.schedule_capacity_edge_when_mixed_floor_only"),
        "schedule_capacity_nonedge_when_mixed_floor_only": config.get("derived.schedule_capacity_nonedge_when_mixed_floor_only"),
        "sw_ucb_xi": float(config.get("algorithm.sw_ucb_xi")),
        "sw_ucb_window_s": float(config.get("algorithm.sw_ucb_window_s")),
        "policy_seed": int(config.get("randomness.policy_seed")),
        "handover_delay_slots": float(config.get("derived.handover_delay_slots")),
        "fixed_isl_per_hop_slots": float(config.get("derived.fixed_isl_per_hop_slots")),
        "shadowed_rician": config.get("channel.shadowed_rician"),
        "noise_assumption": "reference_snr_at_configured_reference_bandwidth",
        "goodput_under_deadlines_status": "NOT_EVALUATED_DEADLINE_RANGE_UNSPECIFIED",
    }
    if any("payload" in key.lower() or "message_size" in key.lower() for key in snapshot):
        raise AssertionError("formal parameter snapshot must be payload independent")
    return snapshot


def _membership_hash(groups, message: int) -> str:
    selected = [group for group in groups if group.message_id == message]
    digest = hashlib.sha256()
    for group in sorted(selected, key=lambda item: item.group_id):
        digest.update(np.sort(group.member_indices).astype("<i4").tobytes())
        digest.update(b"|")
    return digest.hexdigest()[:16]


def _new_normalizers(config: StudyConfig, satellite_count: int, channel_seed: int | None = None) -> list[RewardNormalizer]:
    mode = str(config.get("algorithm.reward_normalization"))
    if mode == "finite_horizon_bound":
        lower, upper = finite_horizon_reward_bounds(config)
        identity = f"finite:{config.config_id}"
    elif mode == "pilot_fixed_clip":
        bounds = config.get("algorithm.reward_bounds")
        if not isinstance(bounds, dict) or bounds.get("lower") is None or bounds.get("upper") is None:
            raise ValueError("pilot_fixed_clip requires frozen algorithm.reward_bounds")
        from .reward_pilot import reward_context
        seed = int(config.get("randomness.fixed_channel_seed")) if channel_seed is None else channel_seed
        if bounds.get("context") != reward_context(config, seed):
            raise ValueError("reward bounds require a compatible 120-UE full-load model context; recalibrate on the new model")
        lower = float(bounds["lower"])
        upper = float(bounds["upper"])
        identity = str(bounds.get("identity", "pilot-unspecified"))
    else:
        raise ValueError(f"unknown reward normalization mode: {mode}")
    return [RewardNormalizer(lower, upper, identity) for _ in range(satellite_count)]


def _all_routes(config: StudyConfig, graph, ingress: int):
    return {
        target: shortest_route(
            graph,
            ingress,
            target,
            float(config.get("geometry.light_speed_km_s")),
            float(config.get("delay.fixed_isl_per_hop_s")),
        )
        for target in graph
    }


def _routes_are_valid(routes, graph) -> bool:
    edges = {(left, right) for left, values in graph.items() for right, _ in values}
    for route in routes.values():
        if not route.reachable:
            continue
        if any((left, right) not in edges for left, right in zip(route.path, route.path[1:])):
            return False
    return True


def _initial_state(config, scenario, policy: str, slot_limit: int, channel_seed: int):
    device_count = len(scenario.devices.ids)
    satellite_count = len(scenario.constellation.elements)
    ages = np.full(device_count, np.nan, dtype=float)
    ages[scenario.subscriptions.active_device_ids] = float(config.get("services.initial_age_slots"))
    agents = None
    normalizers = None
    if policy == "mira_sw_ucb":
        agents = [
            SlidingWindowUCB(
                config.get("algorithm.action_set"),
                int(config.get("derived.sw_ucb_window_epochs")),
                float(config.get("algorithm.sw_ucb_xi")),
                satellite,
            )
            for satellite in range(satellite_count)
        ]
        normalizers = _new_normalizers(config, satellite_count, channel_seed)
    return {
        **initial_epoch_state(),
        "next_slot": 0,
        "ages": ages,
        "aoi_samples": np.empty(
            (slot_limit, len(scenario.subscriptions.active_device_ids)), dtype=np.float32
        ),
        "weighted_sum_aoi_samples": np.empty(slot_limit, dtype=np.float64),
        "association": AssociationTracker(config, scenario.subscriptions),
        "agents": agents,
        "normalizers": normalizers,
        "routes": None,
        "blackout_remaining": 0,
        "previous_membership_hash": {},
        "assignment_tracker": AssignmentTracker(),
        "previous_action": None,
        "sic_projection_count": 0,
        "power_infeasible_serving_slots": 0,
        "power_outage_slots": 0,
        "power_failure_scheduled_groups": 0,
        "power_failure_device_opportunities": 0,
        "transmitted_group_count": 0,
        "scheduler_cap_rejected_group_count": 0,
        "bandit_observation_rows": [],
        "action_change_events": 0,
        "device_regrouping_events": 0,
        "trajectory_hash": "",
        "channel_trace_hash": "",
        "association_trace_hash": "",
        "max_transmitted_power": 0.0,
        "min_slot_scheduled_groups": math.inf,
        "max_slot_scheduled_groups": 0,
        "power_rejected_group_count": 0,
        "nonpositive_priority_rejected_group_count": 0,
        "action_histogram": Counter(),
        "grouping_membership_changes": 0,
        "weighted_age_accumulator": 0.0,
        "weighted_age_warm_accumulator": 0.0,
        "warm_slots": 0,
        "warm_success_groups": 0,
        "power_failure_reasons": Counter(),
        "pool_count_sums": Counter(),
        "protected_power_sum": 0.0,
        "ordinary_power_sum": 0.0,
        "transmitted_sic_violation_count": 0,
        "sum_power_excess": 0.0,
        "propagated_sic_failure_count": 0,
        "geo_pending": {name: [] for name in ("edge_group_snapshots", "edge_member_positions", "geo_edge_time_distribution")},
        "successful_user_updates": 0,
        "scheduled_user_opportunities": 0,
        "successful_capacity_sum_bits_per_s": 0.0,
        "success_group_count": 0,
        "scheduled_group_count": 0,
        "edge_group_count": 0,
        "rate_threshold_failure_count": 0,
        "sic_failure_count": 0,
        "handover_count": 0,
        "coverage_outage_receiver_slots": 0,
        "route_unreachable_slots": 0,
        "power_sum": 0.0,
        "power_samples": 0,
        "interference_samples": array("f"),
        "scheduled_rate_samples": array("f"),
        "violation_counts": Counter(),
        "invalid": False,
        "slot_rows_pending": [],
        "group_rows_pending": [],
        "convergence_rows_pending": [],
        "scheduler_runtime_s": 0.0,
        "regrouping_runtime_s": 0.0,
    }


def validate_checkpoint_identity(checkpoint, *, config_id, channel_seed, slot_limit,
                                 policy, fixed_action, save_group_trace, save_geo_edge_trace=False):
    """Reject older population/state schemas before they can be resumed."""
    expected = dict(config_id=config_id, channel_seed=channel_seed, slot_limit=slot_limit,
                    policy=policy, fixed_action=fixed_action, save_group_trace=save_group_trace,
                    schema_version=CHECKPOINT_SCHEMA_VERSION)
    if (checkpoint.get("save_geo_edge_trace", False) != save_geo_edge_trace
            or any(checkpoint.get(key) != value for key, value in expected.items())):
        raise ValueError("checkpoint identity/schema does not match this 120-UE single-serving run")


def _flush_geo_traces(state, writers):
    for name, writer in writers.items():
        writer.append(state["geo_pending"][name])
        state["geo_pending"][name] = []


def run_simulation(
    config: StudyConfig,
    *,
    channel_seed: int,
    policy: str = "sw-ucb",
    fixed_action: int = 4,
    max_slots: int | None = None,
    output_directory: str | Path | None = None,
    resume: bool = False,
    save_group_trace: bool = False,
    collect_slot_metrics: bool = False,
    save_geo_edge_trace: bool = False,
    stop_after_slots: int | None = None,
    run_directory: str | Path | None = None,
    policy_seed: int | None = None,
) -> dict:
    """Run one seed. The caller controls whether any artifacts are written."""
    policy = canonical_policy_name(policy)
    if config.get("algorithm.power_mode") != POWER_MODE:
        raise ValueError("simulation requires the unique csi_expanded_protected_pool allocator; legacy helpers are fixtures only")
    if policy_seed is not None:
        config = config.with_overrides({"randomness.policy_seed": int(policy_seed)})
    if config.get("algorithm.B_penalty") is None:
        raise ValueError("algorithm.B_penalty must be calibrated or explicitly set before simulation")
    if policy == "fixed_grouping" and fixed_action not in config.get("algorithm.action_set"):
        raise ValueError("fixed_action must belong to algorithm.action_set")
    slot_limit = config.slot_count if max_slots is None else min(int(max_slots), config.slot_count)
    if slot_limit <= 0:
        raise ValueError("simulation requires at least one slot")
    if output_directory is not None and run_directory is not None:
        raise ValueError("use output_directory or run_directory, not both")
    if stop_after_slots is not None and (
        stop_after_slots <= 0 or (output_directory is None and run_directory is None)
    ):
        raise ValueError("stop_after_slots requires a positive value and an artifact directory")

    scenario = StudyScenario(config)
    run_dir = None
    checkpoint_path = None
    slot_writer = None
    group_writer = None
    convergence_writer = None
    geo_writers = {}
    if output_directory is not None or run_directory is not None:
        run_dir = (
            Path(run_directory)
            if run_directory is not None
            else Path(output_directory) / "runs" / config.config_id / f"seed_{channel_seed}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = run_dir / "checkpoint.pkl"
        continue_checkpoint = resume and checkpoint_path.exists()
        if save_geo_edge_trace:
            geo_writers = {name: AppendCsvWriter(run_dir / f"_{name}.partial.csv", resume=continue_checkpoint)
                           for name in ("edge_group_snapshots", "edge_member_positions", "geo_edge_time_distribution")}
        slot_writer = AppendCsvWriter(run_dir / "_slot_metrics.partial.csv", resume=continue_checkpoint)
        convergence_writer = AppendCsvWriter(run_dir / "_convergence.partial.csv", resume=continue_checkpoint)
        if save_group_trace:
            group_writer = AppendCsvWriter(run_dir / "_group_trace.partial.csv", resume=continue_checkpoint)

    if resume and checkpoint_path is not None and checkpoint_path.exists():
        state = load_pickle(checkpoint_path)
        validate_checkpoint_identity(state, config_id=config.config_id, channel_seed=channel_seed,
                                     slot_limit=slot_limit, policy=policy, fixed_action=fixed_action,
                                     save_group_trace=save_group_trace, save_geo_edge_trace=save_geo_edge_trace)
        state = state["state"]
    else:
        state = _initial_state(config, scenario, policy, slot_limit, channel_seed)

    if save_geo_edge_trace and (run_dir is None or float(config.get("algorithm.phi_theta_deg")) != 15.0):
        raise ValueError("geo trace requires a run directory and phi=15 degrees")
    if resume and run_dir is not None:
        from .storage import recover_trace_file
        final_paths = {
            slot_writer.path: run_dir / "slot_metrics.csv",
            convergence_writer.path: run_dir / "convergence_slot_metrics.csv",
        }
        if group_writer is not None:
            final_paths[group_writer.path] = run_dir / "group_trace.csv"
        final_paths.update({writer.path: run_dir / "geo_edge" / f"{name}.csv"
                            for name, writer in geo_writers.items()})
        for writer in [slot_writer, convergence_writer, group_writer, *geo_writers.values()]:
            if writer is not None:
                recover_trace_file(writer.path, final_paths[writer.path], int(state["next_slot"]))
    if run_dir is not None:
        save_config(config, run_dir / "config.json")
        atomic_write_json(run_dir / "parameter_snapshot.json", build_parameter_snapshot(config, policy, channel_seed))

    dt_s = float(config.get("experiment.slot_dt_s"))
    c_km_s = float(config.get("geometry.light_speed_km_s"))
    phi_min = float(config.get("geometry.phi_min_deg"))
    messages = int(config.get("services.messages_per_active_satellite"))
    subscribers = int(config.get("services.subscribers_per_message"))
    weights = np.asarray(config.get("services.weights"), dtype=float)
    bandwidth_hz = float(config.get("services.bandwidth_hz"))
    frozen_b_penalty = float(config.get("algorithm.B_penalty"))
    reward_bounds_identity = str(config.get("algorithm.reward_bounds.identity", ""))
    policy_seed = int(config.get("randomness.policy_seed", channel_seed))
    active = scenario.subscriptions.active_device_ids
    device_weights = np.zeros(len(scenario.devices.ids), dtype=float)
    for message, members in scenario.subscriptions.message_members.items():
        device_weights[members] = weights[message]
    active_weight_sum = float(np.sum(device_weights[active]))
    route_update_slots = max(1, int(round(float(config.get("delay.isl_route_update_s")) / dt_s)))
    ingress = int(str(config.get("delay.ingress_satellite")).split("_")[-1]) - 1
    chunk_size = int(config.get("execution.slot_chunk_size"))
    warmup_s = float(config.get("experiment.warmup_report_s"))
    kh_slots = float(config.get("derived.handover_delay_slots"))
    handover_mode = str(config.get("delay.handover_mode"))
    blackout_length = int(math.ceil(kh_slots))
    satellite_count = len(scenario.constellation.elements)
    interval_slots = int(config.get("derived.regrouping_interval_slots"))
    formal_power = config.get("algorithm.power_mode") == POWER_MODE
    collected_rows: list[dict] = []
    convergence_rows: list[dict] = []

    tracemalloc.start()
    start_time = time.perf_counter()
    call_start_slot = int(state["next_slot"])
    call_end_slot = slot_limit
    if stop_after_slots is not None:
        call_end_slot = min(slot_limit, call_start_slot + int(stop_after_slots))
    for slot in range(call_start_slot, call_end_slot):
        time_s = slot * dt_s
        regrouping_epoch = slot // interval_slots
        is_regrouping_slot = slot % interval_slots == 0
        _, satellite_ecef = scenario.constellation.state(time_s)
        distance, elevation = link_geometry(satellite_ecef, scenario.devices.ecef_km)
        visible = elevation >= phi_min
        association = state["association"].update(slot, elevation)
        state["handover_count"] += int(association.handover)
        state["coverage_outage_receiver_slots"] += association.outage_receivers

        graph = build_isl_graph(
            satellite_ecef,
            float(config.get("geometry.earth_radius_km")),
            float(config.get("delay.isl_max_link_distance_km")),
        )
        if state["routes"] is None or slot % route_update_slots == 0 or not _routes_are_valid(state["routes"], graph):
            state["routes"] = _all_routes(config, graph, ingress)

        channel_amplitude = serving_channel_amplitude(config, channel_seed, slot, distance)
        state["trajectory_hash"] = advance_trace_hash(state["trajectory_hash"], satellite_ecef)
        state["channel_trace_hash"] = advance_trace_hash(state["channel_trace_hash"], channel_amplitude)
        state["association_trace_hash"] = advance_trace_hash(
            state["association_trace_hash"], np.asarray([association.serving_satellite], dtype="<i4"),
            np.asarray(association.handover, dtype=np.uint8), visible,
        )
        path_gains = path_gain(
            distance,
            float(config.get("channel.carrier_hz")),
            c_km_s,
        )

        regroup_start = time.perf_counter()
        satellite = association.serving_satellite
        action, groups, action_changed = groups_for_slot(
            config, state, satellite=satellite, slot=slot,
            policy=policy, policy_seed=policy_seed, fixed_action=fixed_action,
            message_members=scenario.subscriptions.message_members,
            channel_amplitude=channel_amplitude, elevation_deg=elevation,
            select_action=select_grouping_action,
            diagnostics_enabled=save_group_trace,
        )
        if formal_power and len(groups) > 30:
            raise ModelConsistencyError("formal compact setting permits at most 30 groups per serving satellite")
        for message in range(messages):
            membership = _membership_hash(groups, message)
            key = message
            previous = state["previous_membership_hash"].get(key)
            if previous is not None and previous != membership:
                state["grouping_membership_changes"] += 1
            state["previous_membership_hash"][key] = membership
        events = state["assignment_tracker"].update(groups, visible)
        events["action_change_events"] = action_changed
        if not is_regrouping_slot and events["regrouping_events"]:
            raise ModelConsistencyError("off-boundary membership change violates the 1-second cadence")
        state["device_regrouping_events"] += events["regrouping_events"]
        state["regrouping_runtime_s"] += time.perf_counter() - regroup_start

        schedule_start = time.perf_counter()
        satellite = association.serving_satellite
        route = state["routes"][satellite]
        isl_s = route_delay_s(config, route)
        if not route.reachable:
            state["route_unreachable_slots"] += 1
        isl_slots = isl_s / dt_s if math.isfinite(isl_s) else math.inf
        schedule_groups(
            config,
            groups,
            state["ages"],
            distance,
            association.handover,
            isl_slots,
        )
        scheduled_before = tuple(g.key for g in groups if g.scheduled)
        diagnostics = allocate_power(config, groups)
        state["sic_projection_count"] += int(diagnostics["sic_projection_used"])
        schedule_changed = scheduled_before != tuple(g.key for g in groups if g.scheduled)
        power_failure = diagnostics.get("transmission_failure", False)
        if power_failure:
            state["power_infeasible_serving_slots"] += 1
            state["power_failure_reasons"][diagnostics["power_infeasible_reason"]] += 1
            state["power_failure_scheduled_groups"] += len(scheduled_before)
            state["power_failure_device_opportunities"] += sum(len(g.member_indices) for g in groups if g.scheduled)
        if diagnostics.get("removed_groups"):
            state["invalid"] = True
            state["violation_counts"]["post_schedule_power_drop"] += len(diagnostics["removed_groups"])
        if not diagnostics["valid"] and not power_failure:
            state["invalid"] = True
            state["violation_counts"]["invalid_power_or_sic"] += 1
            state["violation_counts"]["power_floor"] += len(diagnostics["floor_violations"])
            state["violation_counts"]["sic_power_order"] += len(diagnostics["sic_order_violations"])
        if schedule_changed or diagnostics.get("removed_groups") or (formal_power and not diagnostics["valid"] and not power_failure):
            failure = {
                "status": "MODEL_CONSISTENCY_ERROR", "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "config_id": config.config_id, "policy": policy, "channel_seed": channel_seed,
                "slot": slot, "scheduled_keys": scheduled_before,
                "schedule_changed": schedule_changed,
                "diagnostic": serialize_power_diagnostic(groups, diagnostics),
            }
            if run_dir is not None:
                atomic_write_json(run_dir / "model_consistency_error.json", failure)
                atomic_write_json(run_dir / "summary.json", failure)
            tracemalloc.stop()
            raise ModelConsistencyError(
                f"MODEL_CONSISTENCY_ERROR at slot={slot}, "
                f"{diagnostics['status']}; the Algorithm-2 scheduled set is unchanged. "
                "Check fixed class budgets, floors and global SIC; no group was dropped."
            )
        state["scheduler_runtime_s"] += time.perf_counter() - schedule_start
        slot_power_failure = int(diagnostics.get("transmission_failure", False))
        state["power_outage_slots"] += int(slot_power_failure > 0)

        serving_actual_tx_power = sum(group.power for group in groups if group.scheduled)
        external_interferer_power_by_satellite = external_interferer_powers(
            config, satellite_count, association.serving_satellite)
        state["max_transmitted_power"] = max(state["max_transmitted_power"], serving_actual_tx_power)
        # Environment powers are recorded for every candidate group, including outages.
        for group in groups:
            ue_interference = expected_cross_link_interference_by_ue(
                config, satellite, group.member_indices, path_gains, visible,
                external_interferer_power_by_satellite)
            worst = int(np.argmax(ue_interference))
            worst_device = int(group.member_indices[worst])
            group.interference = float(ue_interference[worst])
            contributing = visible[:, worst_device] & (external_interferer_power_by_satellite > 0)
            group.metadata["interferer_count_at_worst_ue"] = int(np.count_nonzero(contributing))
            group.metadata["worst_interference_device_index"] = worst_device

        if handover_mode == "physical_blackout":
            event = association.handover
            if event:
                state["blackout_remaining"] = blackout_length

        satellite = association.serving_satellite
        route = state["routes"][satellite]
        for group in groups:
            if group.scheduled and group.power > 0.0:
                state["interference_samples"].append(group.interference)
        if diagnostics["valid"]:
            decode_groups(config, groups)
        state["propagated_sic_failure_count"] += sum(g.scheduled and g.rate_pass and not g.sic_pass for g in groups)
        for group in groups:
            group.metadata["diagnostic_handover_blackout"] = bool(
                handover_mode == "physical_blackout" and state["blackout_remaining"] > 0)
            if not group.scheduled:
                continue
            if not route.reachable or not np.all(visible[satellite, group.member_indices]):
                group.success = False
            if handover_mode == "physical_blackout" and state["blackout_remaining"] > 0:
                group.success = False

        successful_devices = np.zeros(len(scenario.devices.ids), dtype=bool)
        for group in groups:
            if group.success:
                successful_devices[group.member_indices] = True
        members = active
        satellite = association.serving_satellite
        route = state["routes"][satellite]
        isl_s = route_delay_s(config, route)
        isl_slots = isl_s / dt_s if math.isfinite(isl_s) else math.inf
        for device in members:
            if successful_devices[device]:
                propagation_slots = distance[satellite, device] / (c_km_s * dt_s)
                state["ages"][device] = update_user_aoi(
                    state["ages"][device],
                    successful=True,
                    delta_c_slots=float(config.get("services.delta_c_slots")),
                    propagation_delay_slots=propagation_slots,
                    isl_delay_slots=isl_slots,
                    handover_event=association.handover,
                    handover_delay_slots=kh_slots,
                    handover_mode=handover_mode,
                )
            else:
                state["ages"][device] = update_user_aoi(
                    state["ages"][device],
                    successful=False,
                    delta_c_slots=float(config.get("services.delta_c_slots")),
                    propagation_delay_slots=0.0,
                    isl_delay_slots=0.0,
                    handover_event=association.handover,
                    handover_delay_slots=kh_slots,
                    handover_mode=handover_mode,
                )
        if handover_mode == "physical_blackout":
            state["blackout_remaining"] = max(0, state["blackout_remaining"] - 1)

        state["aoi_samples"][slot] = state["ages"][active]
        weighted_age = float(np.sum(state["ages"][active] * device_weights[active]))
        state["weighted_sum_aoi_samples"][slot] = weighted_age
        state["weighted_age_accumulator"] += weighted_age
        if time_s >= warmup_s:
            state["weighted_age_warm_accumulator"] += weighted_age
            state["warm_slots"] += 1

        slot_success_users = int(np.count_nonzero(successful_devices[active]))
        state["successful_user_updates"] += slot_success_users
        slot_scheduled_groups = sum(group.scheduled for group in groups)
        slot_success_groups = sum(group.success for group in groups if group.scheduled)
        if time_s >= warmup_s:
            state["warm_success_groups"] += slot_success_groups
        p1_value = p1_slot_value(slot_success_groups, weighted_age)
        pool_metrics = dict(
            geo_edge_candidate_count=sum(g.is_edge for g in groups),
            geo_edge_scheduled_count=diagnostics["geo_edge_count_scheduled"],
            promoted_count=diagnostics["promoted_weak_nonedge_count"],
            protected_scheduled_count=diagnostics["protected_count"], ordinary_scheduled_count=diagnostics["ordinary_count"],
            protected_power_actual=diagnostics["protected_power_actual"], ordinary_power_actual=diagnostics["ordinary_power_actual"],
            protected_budget=diagnostics["protected_budget"], ordinary_budget=diagnostics["ordinary_budget"],
            pool_budget_mode=diagnostics["pool_budget_mode"], power_failure_reason=diagnostics["power_infeasible_reason"],
            P1_slot_value=p1_value)
        for key in ("geo_edge_candidate_count", "geo_edge_scheduled_count", "promoted_count", "protected_scheduled_count", "ordinary_scheduled_count"):
            state["pool_count_sums"][key] += pool_metrics[key]
        state["protected_power_sum"] += diagnostics["protected_power_actual"]
        state["ordinary_power_sum"] += diagnostics["ordinary_power_actual"]
        state["transmitted_sic_violation_count"] += len(diagnostics["sic_order_violations_after"]) if diagnostics["transmission_attempted"] else 0
        state["sum_power_excess"] += max(0., serving_actual_tx_power-float(config.get("channel.normalized_satellite_power"))-1e-10)
        if save_geo_edge_trace:
            state["geo_pending"]["geo_edge_time_distribution"].append(geo_occupancy_row(
                groups, slot=slot, dt_s=dt_s, satellite_id=scenario.constellation.elements[satellite].satellite_id,
                handover=association.handover))
            if is_regrouping_slot:
                snapshot_rows, member_rows = snapshot_group_rows(config, scenario, groups, slot=slot,
                    elevation=elevation, amplitudes=channel_amplitude, satellite_ecef=satellite_ecef)
                state["geo_pending"]["edge_group_snapshots"].extend(snapshot_rows)
                state["geo_pending"]["edge_member_positions"].extend(member_rows)
        state["min_slot_scheduled_groups"] = min(state["min_slot_scheduled_groups"], slot_scheduled_groups)
        state["max_slot_scheduled_groups"] = max(state["max_slot_scheduled_groups"], slot_scheduled_groups)
        per_satellite_aoi = np.zeros(satellite_count)
        members = active
        per_satellite_aoi[association.serving_satellite] += float(np.sum(state["ages"][members] * device_weights[members]))
        convergence_row = {
            "slot": slot, "time_s": time_s, "policy": policy, "aoi_unit": "slots",
            "regrouping_epoch": regrouping_epoch,
            "is_regrouping_slot": int(is_regrouping_slot),
            "weighted_sum_aoi_total": weighted_age,
            **pool_metrics,
            "running_mean_P1": (state["success_group_count"]+slot_success_groups-state["weighted_age_accumulator"])/(slot+1),
            "running_mean_P1_after_warmup": ((state["warm_success_groups"]-state["weighted_age_warm_accumulator"])/state["warm_slots"]
                                               if state["warm_slots"] else None),
            "evaluated_system_weighted_sum_aoi": weighted_age,
            "weighted_sum_aoi_per_satellite_max": float(np.max(per_satellite_aoi)),
            "running_mean_weighted_sum_aoi_total": state["weighted_age_accumulator"] / (slot + 1),
            "scheduled_groups": slot_scheduled_groups,
            "successful_groups": slot_success_groups,
            "power_infeasible_serving": slot_power_failure,
            "transmitted_groups": sum(g.scheduled for g in groups) if diagnostics.get("transmission_attempted", False) else 0,
            "action_summary": json.dumps({scenario.constellation.elements[satellite].satellite_id: action}, sort_keys=True),
        }
        state["convergence_rows_pending"].append(convergence_row)
        if collect_slot_metrics:
            convergence_rows.append(dict(convergence_row))
        serving_edge_groups = sum(group.is_edge for group in groups)
        state["scheduled_group_count"] += slot_scheduled_groups
        state["success_group_count"] += slot_success_groups
        state["edge_group_count"] += serving_edge_groups
        for group in groups:
            if not group.scheduled or group.metadata.get("power_failure_no_transmission", False):
                continue
            state["scheduled_rate_samples"].append(group.rate_bps_hz)
            state["rate_threshold_failure_count"] += int(not group.rate_pass)
            state["sic_failure_count"] += int(not group.sic_pass)
            if group.success:
                state["successful_capacity_sum_bits_per_s"] += bandwidth_hz * group.rate_bps_hz
        satellite = association.serving_satellite
        scheduled = [group for group in groups if group.scheduled]
        success_count = sum(group.success for group in scheduled)
        actual_transmitted_groups = len(scheduled) if diagnostics.get("transmission_attempted", False) else 0
        state["transmitted_group_count"] += actual_transmitted_groups
        edge_count = sum(group.is_edge for group in groups)
        scheduled_users = sum(len(group.member_indices) for group in scheduled)
        state["scheduled_user_opportunities"] += scheduled_users
        state["power_sum"] += serving_actual_tx_power
        state["power_samples"] += 1
        members = active
        system_weighted_age = float(np.sum(state["ages"][members] * device_weights[members]))
        reward = raw_reward(
            success_count,
            system_weighted_age,
            frozen_b_penalty,
            messages,
            subscribers,
            edge_count,
        )
        normalized_reward = None
        epoch_reward = None
        if policy == "mira_sw_ucb":
            normalized_reward, epoch_reward = observe_slot_reward(
                config, state, slot=slot, slot_limit=slot_limit, raw_reward=reward,
            )
            if epoch_reward is not None:
                state["bandit_observation_rows"].append({
                    "epoch": regrouping_epoch, 
                    "owner_satellite": state["epoch_action_owner_satellite"],
                    "action": action,
                    "sample_count": state["epoch_reward_count"],
                    "epoch_reward": epoch_reward,
                })
        row = {
            "policy": policy,
            **pool_metrics,
            "slot": slot,
            "time_s": time_s,

            "satellite_id": scenario.constellation.elements[satellite].satellite_id,
            "action": action,
            "regrouping_epoch": regrouping_epoch,
            "is_regrouping_slot": int(is_regrouping_slot),
            "regrouping_interval_s": float(config.get("algorithm.regrouping_interval_s")),
            "regrouping_interval_slots": interval_slots,
            "sw_ucb_window_epochs": int(config.get("derived.sw_ucb_window_epochs")),
            "action_owner_satellite_id": scenario.constellation.elements[state["epoch_action_owner_satellite"]].satellite_id,
            "membership_hash": state["held_partition_hash"],
            "handover": int(association.handover),
            "outage_receivers": association.outage_receivers,
            "route_reachable": int(state["routes"][satellite].reachable),
            "route_hops": state["routes"][satellite].hop_count,
            "scheduled_groups": len(scheduled),
            "successful_groups": success_count,
            "transmitted_groups": actual_transmitted_groups,
            "power_plan_valid": int(diagnostics["valid"]),
            "transmission_attempted": int(diagnostics.get("transmission_attempted", False)),
            "transmission_failure_power": int(diagnostics.get("transmission_failure", False)),
            "power_solver_status": -1 if diagnostics.get("projection_solver_status") is None else int(diagnostics["projection_solver_status"]),
            "edge_groups": edge_count,
            "serving_edge_groups": serving_edge_groups,
            "serving_scheduled_groups": slot_scheduled_groups,
            "serving_successful_groups": slot_success_groups,
            "cumulative_successful_groups": int(state["success_group_count"]),
            "transmitted_power_normalized": serving_actual_tx_power,
            "serving_actual_tx_power": serving_actual_tx_power,
            "interferer_power_normalized": float(config.get("channel.normalized_satellite_power")),
            "external_interferer_count": satellite_count - 1,
            "external_interferer_power_by_satellite": json.dumps(external_interferer_power_by_satellite.tolist()),
            "weighted_aoi_slots": system_weighted_age,
            "aggregated_weighted_sum_aoi_slots": weighted_age,
            "aggregated_weighted_sum_aoi_ms": weighted_age * dt_s * 1000.0,
            "raw_reward": reward,
            "normalized_reward": normalized_reward,
            "epoch_reward": epoch_reward,
            "power_status": diagnostics["status"],
            "candidate_groups": len(groups),
            "candidate_groups_positive_priority": sum(g.priority > 0.0 for g in groups),
            "rejected_by_power_groups": 0,
            "rejected_by_algorithm2_cap": sum(g.metadata["scheduler_rejection_reason"] == "ALGORITHM2_CAP" for g in groups),
            "scheduled_edge_groups": sum(g.is_edge for g in scheduled),
            "scheduled_nonedge_groups": sum(not g.is_edge for g in scheduled),
            "rejected_edge_by_power": 0,
            "rejected_nonedge_by_power": 0,
            "unscheduled_nonpositive_priority_groups": sum(not g.is_edge and not (g.priority > 0.0) and not g.scheduled for g in groups),
            "edge_budget": diagnostics["edge_budget"],
            "nonedge_budget": diagnostics["nonedge_budget"],
            "edge_floor_total": diagnostics["edge_floor_total"],
            "nonedge_floor_total": diagnostics["nonedge_floor_total"],
            "edge_residual": diagnostics["edge_residual"],
            "nonedge_residual": diagnostics["nonedge_residual"],
            "current_csi_feedback_available": 1,
            "csi_slot_index": slot,
            "power_csi_slot_index": slot,
            "scheduler_algorithm": "Algorithm2_strict",
            "power_allocator": str(config.get("algorithm.power_mode")),
            "algorithm2_U": int(config.get("derived.algorithm2_U")),
            "candidate_edge_groups": sum(g.is_edge for g in groups),
            "candidate_nonedge_groups": sum(not g.is_edge for g in groups),
            "candidate_nonedge_positive_priority": sum(not g.is_edge and g.priority > 0 for g in groups),
            "sic_projection_used": int(diagnostics["sic_projection_used"]),
            "sic_violation_before_count": len(diagnostics["sic_order_violations_before"]),
            "sic_violation_after_count": len(diagnostics["sic_order_violations_after"]),
            "planned_total_power": diagnostics["total_power"],
            "total_transmitted_power": diagnostics["actual_total_power"],
            "transmitted_sic_violation_count": 0 if diagnostics.get("transmission_failure", False) else len(diagnostics["sic_order_violations_after"]),
            "post_schedule_power_drop_count": len(diagnostics["removed_groups"]),
            "regrouping_action_change": events["action_change_events"],
            "device_regrouping_events": events["regrouping_events"],
            "initial_assignments": events["initial_assignments"],
            "handover_assignments": events["handover_assignments"],
        }
        state["power_rejected_group_count"] += row["rejected_by_power_groups"]
        state["scheduler_cap_rejected_group_count"] += row["rejected_by_algorithm2_cap"]
        state["nonpositive_priority_rejected_group_count"] += row["unscheduled_nonpositive_priority_groups"]
        state["slot_rows_pending"].append(row)
        if collect_slot_metrics:
            collected_rows.append(dict(row))
        if save_group_trace:
            from .diagnostic_trace import enrich_group_diagnostic_rows
            group_rows, serving_diagnostics = enrich_group_diagnostic_rows(
                config, groups, diagnostics, ages_after=state["ages"],
                amplitudes=channel_amplitude, elevations=elevation, visible=visible,
                route_reachable=state["routes"][satellite].reachable,
                handover_blackout=any(g.metadata.get("diagnostic_handover_blackout", False) for g in groups),
                system_weighted_aoi=system_weighted_age, policy=policy, slot=slot,
                satellite_id=scenario.constellation.elements[satellite].satellite_id,
                action=action)
            # The CSV row object was queued above; extend it before chunk flush.
            row.update(serving_diagnostics)
            if collect_slot_metrics:
                collected_rows[-1].update(serving_diagnostics)
            for group_row, group in zip(group_rows, groups):
                group_row["members"] = json.dumps([int(d) for d in group.member_indices])
            state["group_rows_pending"].extend(group_rows)

        state["next_slot"] = slot + 1
        if run_dir is not None and (
            len(state["slot_rows_pending"]) >= chunk_size
            or state["next_slot"] == slot_limit
        ):
            slot_writer.append(state["slot_rows_pending"])
            state["slot_rows_pending"] = []
            convergence_writer.append(state["convergence_rows_pending"])
            state["convergence_rows_pending"] = []
            if group_writer is not None:
                group_writer.append(state["group_rows_pending"])
                state["group_rows_pending"] = []
            _flush_geo_traces(state, geo_writers)
            atomic_pickle(
                checkpoint_path,
                {
                    "config_id": config.config_id,
                    "channel_seed": channel_seed,
                    "slot_limit": slot_limit,
                    "policy": policy,
                    "fixed_action": fixed_action,
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "save_group_trace": save_group_trace,
                    "save_geo_edge_trace": save_geo_edge_trace,
                    "state": state,
                },
            )

    if int(state["next_slot"]) < slot_limit:
        if state["slot_rows_pending"]:
            slot_writer.append(state["slot_rows_pending"])
            state["slot_rows_pending"] = []
        convergence_writer.append(state["convergence_rows_pending"])
        state["convergence_rows_pending"] = []
        if group_writer is not None and state["group_rows_pending"]:
            group_writer.append(state["group_rows_pending"])
            state["group_rows_pending"] = []
        _flush_geo_traces(state, geo_writers)
        atomic_pickle(
            checkpoint_path,
            {
                "config_id": config.config_id,
                "channel_seed": channel_seed,
                "slot_limit": slot_limit,
                "policy": policy,
                "fixed_action": fixed_action,
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "save_group_trace": save_group_trace,
                "save_geo_edge_trace": save_geo_edge_trace,
                "state": state,
            },
        )
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "summary": {
                "status": "CHECKPOINTED",
                "config_id": config.config_id,
                "channel_seed": int(channel_seed),
                "slots_completed": int(state["next_slot"]),
                "slot_limit": slot_limit,
                "peak_traced_memory_mib": peak_memory / (1024.0 * 1024.0),
            },
            "slot_metrics": collected_rows,
        }

    elapsed = time.perf_counter() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    aoi_values = np.asarray(state["aoi_samples"][:slot_limit], dtype=float)
    weighted_sum_values = np.asarray(
        state["weighted_sum_aoi_samples"][:slot_limit], dtype=float
    )
    interference_values = np.frombuffer(state["interference_samples"], dtype=np.float32)
    rate_values = np.frombuffer(state["scheduled_rate_samples"], dtype=np.float32)
    opportunities = slot_limit * len(active)
    conditional_opportunities = int(state["scheduled_user_opportunities"])
    clipping_rates = (
        [normalizer.clipping_rate for normalizer in state["normalizers"]]
        if state["normalizers"] is not None
        else []
    )
    summary = {
        **objective_summary(sum_c=state["success_group_count"], sum_delta=state["weighted_age_accumulator"],
                            slots=slot_limit, warm_c=state["warm_success_groups"],
                            warm_delta=state["weighted_age_warm_accumulator"], warm_slots=state["warm_slots"],
                            warmup_s=warmup_s, slot_dt_s=dt_s),
        "power_pool_rule_version": POOL_RULE_VERSION,
        **dict(state["pool_count_sums"]),
        "promoted_weak_nonedge_count": state["pool_count_sums"]["promoted_count"],
        "protected_count": state["pool_count_sums"]["protected_scheduled_count"],
        "ordinary_count": state["pool_count_sums"]["ordinary_scheduled_count"],
        "protected_pool_fraction_realized": state["protected_power_sum"]/(state["protected_power_sum"]+state["ordinary_power_sum"]) if state["protected_power_sum"]+state["ordinary_power_sum"] > 0 else None,
        "floor_infeasible_slots": state["power_failure_reasons"]["FLOOR_EXCEEDS_POOL"],
        "cross_pool_sic_infeasible_slots": state["power_failure_reasons"]["POOL_SIC_INCOMPATIBLE"],
        "lp_numeric_failure_slots": state["power_failure_reasons"]["LP_NUMERIC_FAILURE"],
        "power_failure_reasons": dict(state["power_failure_reasons"]),
        "power_outage_rate": state["power_outage_slots"]/slot_limit,
        "transmitted_sic_violation_count": state["transmitted_sic_violation_count"],
        "sum_power_excess": state["sum_power_excess"],
        "propagated_sic_failure_count": state["propagated_sic_failure_count"],

        "status": "INVALID_POWER_OR_SIC" if state["invalid"] else (
            "COMPLETED_WITH_POWER_OUTAGES" if state["power_infeasible_serving_slots"] else "COMPLETED"),
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "total_ue": 120,
        "message_count": 3,
        "subscribers_per_message": int(config.get("services.subscribers_per_message")),
        "serving_satellite_count": 1,
        "interference_mode": "fixed_full_load",
        "interference_model": str(config.get("channel.external_interference")),
        "interferer_power_normalized": float(config.get("channel.normalized_satellite_power")),
        "aoi_scope": "evaluated_120_ue_system",
        "per_satellite_aoi_semantics": "120-UE contribution only at the current serving satellite",

        "config_id": config.config_id,
        "policy": policy,
        "channel_seed": int(channel_seed),
        "slots_completed": slot_limit,
        "duration_simulated_s": slot_limit * dt_s,
        "spectral_efficiency_threshold_bps_hz": float(config.get("algorithm.R_theta")),
        "bandwidth_hz": bandwidth_hz,
        "scs_hz": float(config.get("radio.scs_hz")),
        "phi_theta_deg": float(config.get("algorithm.phi_theta_deg")),
        "B_penalty": frozen_b_penalty,
        "V_c": float(config.get("algorithm.V_c")),
        "V_nc": float(config.get("algorithm.V_nc")),
        "V_phi": float(config.get("algorithm.V_phi")),
        "normalized_satellite_power": float(config.get("channel.normalized_satellite_power")),
        "fixed_action": fixed_action if policy == "fixed_grouping" else None,
        "policy_seed": policy_seed,
        "tau_s": float(config.get("algorithm.sw_ucb_window_s")),
        "action_set_name": str(config.get("algorithm.action_set_name")),
        "action_set": list(config.get("algorithm.action_set")),
        "regrouping_interval_s": float(config.get("algorithm.regrouping_interval_s")),
        "regrouping_interval_slots": interval_slots,
        "sw_ucb_window_epochs": int(config.get("derived.sw_ucb_window_epochs")),
        "scheduler_algorithm": "Algorithm2_strict",
        "power_allocator": str(config.get("algorithm.power_mode")),
        "infeasible_power_handling": str(config.get("algorithm.infeasible_power_handling", "error")),
        "power_infeasible_serving_slots": state["power_infeasible_serving_slots"],
        "power_outage_slots": state["power_outage_slots"],
        "power_failure_scheduled_groups": state["power_failure_scheduled_groups"],
        "power_failure_device_opportunities": state["power_failure_device_opportunities"],
        "transmitted_group_count": state["transmitted_group_count"],
        "sic_projection_count": state["sic_projection_count"],
        "scheduler_cap_rejected_group_count": state["scheduler_cap_rejected_group_count"],
        "bandit_observation_count": state["bandit_observation_count"],
        "bandit_observations": state["bandit_observation_rows"],
        "reward_bounds_identity": reward_bounds_identity,
        "schedule_capacity_continuous": float(config.get("derived.schedule_capacity_continuous")),
        "schedule_capacity_integer": int(config.get("derived.schedule_capacity_integer")),
        "schedule_capacity_paper_approximation": int(
            config.get("derived.schedule_capacity_paper_approximation")
        ),
        "layout_hash": scenario.devices.layout_hash,
        "subscription_hash": scenario.subscriptions.subscription_hash,
        "device_layout_hash": scenario.devices.layout_hash,
        "trajectory_hash": state["trajectory_hash"],
        "channel_trace_hash": state["channel_trace_hash"],
        "environment_trace_hash": hashlib.sha256(json.dumps([
            scenario.devices.layout_hash, scenario.subscriptions.subscription_hash,
            state["trajectory_hash"], state["channel_trace_hash"], state["association_trace_hash"],
            config.get("channel.external_interference"), config.get("channel.interferer_power_mode"),
            config.get("channel.normalized_satellite_power"),
        ]).encode("ascii")).hexdigest(),
        "weighted_mean_aoi_ms": dt_s * 1000.0 * state["weighted_age_accumulator"] / (slot_limit * active_weight_sum),
        "aggregated_weighted_sum_aoi_mean_ms": (
            dt_s * 1000.0 * state["weighted_age_accumulator"] / slot_limit
        ),
        "weighted_mean_aoi_after_warmup_ms": (
            dt_s * 1000.0 * state["weighted_age_warm_accumulator"] / state["warm_slots"]
            if state["warm_slots"]
            else None
        ),
        "weighted_mean_aoi_per_weight_unit_after_warmup_ms": (
            dt_s
            * 1000.0
            * state["weighted_age_warm_accumulator"]
            / (state["warm_slots"] * active_weight_sum)
            if state["warm_slots"]
            else None
        ),
        "aoi_p50_ms": float(np.quantile(aoi_values, 0.50) * dt_s * 1000.0),
        "aoi_p95_ms": float(np.quantile(aoi_values, 0.95) * dt_s * 1000.0),
        "aoi_p99_ms": float(np.quantile(aoi_values, 0.99) * dt_s * 1000.0),
        "weighted_sum_aoi_p50_ms": float(np.quantile(weighted_sum_values, 0.50) * dt_s * 1000.0),
        "weighted_sum_aoi_p95_ms": float(np.quantile(weighted_sum_values, 0.95) * dt_s * 1000.0),
        "weighted_sum_aoi_p99_ms": float(np.quantile(weighted_sum_values, 0.99) * dt_s * 1000.0),
        "packet_success_ratio": state["successful_user_updates"] / opportunities,
        "conditional_packet_success_ratio": (
            state["successful_user_updates"] / conditional_opportunities
            if conditional_opportunities
            else None
        ),
        "capacity_goodput_bps": state["successful_capacity_sum_bits_per_s"] / slot_limit,
        "goodput_under_deadlines_status": "NOT_EVALUATED_DEADLINE_RANGE_UNSPECIFIED",
        "successful_group_count": int(state["success_group_count"]),
        "scheduled_group_count": int(state["scheduled_group_count"]),
        "mean_scheduled_groups": state["scheduled_group_count"] / slot_limit,
        "min_scheduled_groups": state["min_slot_scheduled_groups"],
        "max_scheduled_groups": state["max_slot_scheduled_groups"],
        "power_rejected_group_count": state["power_rejected_group_count"],
        "nonpositive_priority_rejected_group_count": state["nonpositive_priority_rejected_group_count"],
        "invalid_power_or_sic_count": state["violation_counts"]["invalid_power_or_sic"],
        "post_schedule_power_drop_count": state["violation_counts"]["post_schedule_power_drop"],
        "max_transmitted_power_normalized": state["max_transmitted_power"],
        "conditional_group_success_ratio": (
            state["success_group_count"] / state["scheduled_group_count"]
            if state["scheduled_group_count"]
            else None
        ),
        "edge_group_count": int(state["edge_group_count"]),
        "mean_scheduled_R_bps_hz": float(np.mean(rate_values)) if rate_values.size else None,
        "p05_scheduled_R_bps_hz": float(np.quantile(rate_values, 0.05)) if rate_values.size else None,
        "p50_scheduled_R_bps_hz": float(np.quantile(rate_values, 0.50)) if rate_values.size else None,
        "p95_scheduled_R_bps_hz": float(np.quantile(rate_values, 0.95)) if rate_values.size else None,
        "rate_threshold_failure_count": int(state["rate_threshold_failure_count"]),
        "sic_failure_count": int(state["sic_failure_count"]),
        "mean_transmitted_power_normalized": (
            float(state["power_sum"] / state["power_samples"])
            if state["power_samples"]
            else 0.0
        ),
        "mean_interference_power": float(np.mean(interference_values)) if interference_values.size else None,
        "p95_interference_power": float(np.quantile(interference_values, 0.95)) if interference_values.size else None,
        "coverage_outage_receiver_slots": int(state["coverage_outage_receiver_slots"]),
        "handover_count": int(state["handover_count"]),
        "route_unreachable_slots": int(state["route_unreachable_slots"]),
        "grouping_membership_changes": int(state["grouping_membership_changes"]),
        "regrouping_events_total": int(state["device_regrouping_events"]),
        "action_change_events": int(state["action_change_events"]),
        **regrouping_overhead(state["device_regrouping_events"], sum(len(m) for m in scenario.subscriptions.message_members.values()), slot_limit * dt_s),
        "action_histogram": {
            f"sat_{satellite + 1:02d}:a={action}": count
            for (satellite, action), count in sorted(state["action_histogram"].items())
        },
        "action_occupancy_slots": {f"sat_{s + 1:02d}:a={a}": n for (s, a), n in sorted(state["action_histogram"].items())},
        "action_decision_count": {f"sat_{s + 1:02d}:a={a}": n for (s, a), n in sorted(state["action_decision_histogram"].items())},
        "reward_clipping_rate_max": max(clipping_rates, default=0.0),
        "violation_counts": dict(state["violation_counts"]),
        "runtime_s": elapsed,
        "scheduler_runtime_s": float(state["scheduler_runtime_s"]),
        "regrouping_runtime_s": float(state["regrouping_runtime_s"]),
        "peak_traced_memory_mib": peak_memory / (1024.0 * 1024.0),
    }
    if not state["invalid"] and summary["reward_clipping_rate_max"] > 0.01:
        summary["status"] = "REWARD_RANGE_MISMATCH"
    if not math.isclose(
        frozen_b_penalty,
        float(config.get("algorithm.B_penalty")),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise RuntimeError("B_penalty changed during a formal run")
    if reward_bounds_identity != str(config.get("algorithm.reward_bounds.identity", "")):
        raise RuntimeError("reward bounds changed during a formal run")

    if run_dir is not None:
        from .diagnostic_trace import DIAGNOSTIC_SCHEMA_VERSION, parquet_diagnostic_types
        summary["diagnostic_schema_version"] = DIAGNOSTIC_SCHEMA_VERSION if save_group_trace else 0
        if save_geo_edge_trace:
            geo_directory = run_dir / "geo_edge"
            geo_directory.mkdir(exist_ok=True)
            for name, writer in geo_writers.items():
                writer.path.replace(geo_directory / f"{name}.csv")
        atomic_write_json(run_dir / "summary.json", summary)
        write_csv(run_dir / "bandit_observations.csv", state["bandit_observation_rows"],
                  fieldnames=["epoch", "owner_satellite", "action", "sample_count", "epoch_reward"])
        with (run_dir / "_slot_metrics.partial.csv").open(encoding="utf-8", newline="") as handle:
            import csv
            columns = next(csv.reader(handle))
        # Nullable class CSI and solver messages need stable Arrow types as well.
        types = parquet_diagnostic_types(columns)
        csv_to_parquet(run_dir / "_slot_metrics.partial.csv", run_dir / "slot_metrics.parquet", column_types=types)
        if save_group_trace:
            with (run_dir / "_group_trace.partial.csv").open(encoding="utf-8", newline="") as handle:
                columns = next(csv.reader(handle))
            csv_to_parquet(run_dir / "_group_trace.partial.csv", run_dir / "group_trace.parquet",
                           column_types=parquet_diagnostic_types(columns))
        Path(run_dir / "_slot_metrics.partial.csv").replace(run_dir / "slot_metrics.csv")
        if save_group_trace:
            Path(run_dir / "_group_trace.partial.csv").replace(run_dir / "group_trace.csv")
        Path(run_dir / "_convergence.partial.csv").replace(run_dir / "convergence_slot_metrics.csv")
        checkpoint_path.unlink(missing_ok=True)
    return {"summary": summary, "slot_metrics": collected_rows, "convergence_metrics": convergence_rows}
