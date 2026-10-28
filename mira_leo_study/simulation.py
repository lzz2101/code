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
from .channel import expected_cross_link_interference, path_gain, serving_channel_amplitude
from .config import FIXED_PHI_THETA_DEG, FIXED_B_PENALTY, StudyConfig, save_config
from .errors import BaselineUnavailableError, ModelConsistencyError
from .geometry import link_geometry
from .metrics import AssignmentTracker, advance_trace_hash, regrouping_overhead
from .isl import build_isl_graph, shortest_route, route_delay_s
from .scenario import StudyScenario
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
CHECKPOINT_SCHEMA_VERSION = 4


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
        for g in sorted(groups, key=lambda item: (item.gain, item.cohort_id, item.message_id, item.group_id))
    ]
    return result


def build_parameter_snapshot(config: StudyConfig, policy: str, channel_seed: int) -> dict:
    snapshot = {
        "policy": policy,
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
        "fixed_phi_and_B": config.get("algorithm.phi_theta_deg") == FIXED_PHI_THETA_DEG and config.get("algorithm.B_penalty") == FIXED_B_PENALTY,
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
        "subscribers_per_message": int(config.get("services.subscribers_per_message")),
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


def _membership_hash(groups, cohort: int, message: int) -> str:
    selected = [group for group in groups if group.cohort_id == cohort and group.message_id == message]
    digest = hashlib.sha256()
    for group in sorted(selected, key=lambda item: item.group_id):
        digest.update(np.sort(group.member_indices).astype("<i4").tobytes())
        digest.update(b"|")
    return digest.hexdigest()[:16]


def _new_normalizers(config: StudyConfig, satellite_count: int) -> list[RewardNormalizer]:
    mode = str(config.get("algorithm.reward_normalization"))
    if mode == "finite_horizon_bound":
        lower, upper = finite_horizon_reward_bounds(config)
        identity = f"finite:{config.config_id}"
    elif mode == "pilot_fixed_clip":
        bounds = config.get("algorithm.reward_bounds")
        if not isinstance(bounds, dict) or bounds.get("lower") is None or bounds.get("upper") is None:
            raise ValueError("pilot_fixed_clip requires frozen algorithm.reward_bounds")
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


def _initial_state(config, scenario, policy: str, slot_limit: int):
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
        normalizers = _new_normalizers(config, satellite_count)
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
        "blackout_remaining": np.zeros(int(config.get("services.cohort_count")), dtype=np.int32),
        "previous_membership_hash": {},
        "assignment_trackers": [AssignmentTracker() for _ in scenario.subscriptions.cohort_members],
        "previous_actions": {},
        "sic_projection_count": 0,
        "power_infeasible_cohort_slots": 0,
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
    stop_after_slots: int | None = None,
    run_directory: str | Path | None = None,
    policy_seed: int | None = None,
) -> dict:
    """Run one seed. The caller controls whether any artifacts are written."""
    policy = canonical_policy_name(policy)
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
    if output_directory is not None or run_directory is not None:
        run_dir = (
            Path(run_directory)
            if run_directory is not None
            else Path(output_directory) / "runs" / config.config_id / f"seed_{channel_seed}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = run_dir / "checkpoint.pkl"
        continue_checkpoint = resume and checkpoint_path.exists()
        slot_writer = AppendCsvWriter(run_dir / "_slot_metrics.partial.csv", resume=continue_checkpoint)
        convergence_writer = AppendCsvWriter(run_dir / "_convergence.partial.csv", resume=continue_checkpoint)
        if save_group_trace:
            group_writer = AppendCsvWriter(run_dir / "_group_trace.partial.csv", resume=continue_checkpoint)

    if resume and checkpoint_path is not None and checkpoint_path.exists():
        state = load_pickle(checkpoint_path)
        if (
            state["config_id"] != config.config_id
            or state["channel_seed"] != channel_seed
            or state["slot_limit"] != slot_limit
            or state.get("policy") != policy
            or state.get("fixed_action") != fixed_action
            or state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or state.get("save_group_trace") != save_group_trace
        ):
            raise ValueError("checkpoint identity does not match this run")
        state = state["state"]
    else:
        state = _initial_state(config, scenario, policy, slot_limit)

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
    for (cohort, message), members in scenario.subscriptions.message_members.items():
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
    formal_power = config.get("algorithm.power_mode") == "twc_aim_staircase"
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
        state["handover_count"] += sum(association.handover)
        state["coverage_outage_receiver_slots"] += sum(association.outage_receivers)
        if len(set(association.serving_satellites)) != len(association.serving_satellites):
            raise ModelConsistencyError("distinct serving satellites are required for the cohort power pools")

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
            state["association_trace_hash"], np.asarray(association.serving_satellites, dtype="<i4"),
            np.asarray(association.handover, dtype=np.uint8), visible,
        )
        path_gains = path_gain(
            distance,
            float(config.get("channel.carrier_hz")),
            c_km_s,
        )

        groups_by_cohort: dict[int, list] = {}
        actions_by_cohort: dict[int, int] = {}
        power_diagnostics: dict[int, dict] = {}
        regrouping_diagnostics: dict[int, dict] = {}
        all_groups = []
        regroup_start = time.perf_counter()
        for cohort, satellite in enumerate(association.serving_satellites):
            action, groups, action_changed = groups_for_slot(
                config, state, cohort=cohort, satellite=satellite, slot=slot,
                policy=policy, policy_seed=policy_seed, fixed_action=fixed_action,
                message_members=scenario.subscriptions.message_members,
                channel_amplitude=channel_amplitude, elevation_deg=elevation,
                select_action=select_grouping_action,
            )
            actions_by_cohort[cohort] = action
            if formal_power and len(groups) > 30:
                raise ModelConsistencyError("formal compact setting permits at most 30 groups per serving satellite")
            for message in range(messages):
                membership = _membership_hash(groups, cohort, message)
                key = (cohort, message)
                previous = state["previous_membership_hash"].get(key)
                if previous is not None and previous != membership:
                    state["grouping_membership_changes"] += 1
                state["previous_membership_hash"][key] = membership
            groups_by_cohort[cohort] = groups
            events = state["assignment_trackers"][cohort].update(groups, visible)
            events["action_change_events"] = action_changed
            regrouping_diagnostics[cohort] = events
            if not is_regrouping_slot and events["regrouping_events"]:
                raise ModelConsistencyError("off-boundary membership change violates the 1-second cadence")
            state["device_regrouping_events"] += events["regrouping_events"]
            all_groups.extend(groups)
        state["regrouping_runtime_s"] += time.perf_counter() - regroup_start

        schedule_start = time.perf_counter()
        for cohort, groups in groups_by_cohort.items():
            satellite = association.serving_satellites[cohort]
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
                association.handover[cohort],
                isl_slots,
            )
            scheduled_before = tuple(g.key for g in groups if g.scheduled)
            diagnostics = allocate_power(config, groups)
            power_diagnostics[cohort] = diagnostics
            state["sic_projection_count"] += int(diagnostics["sic_projection_used"])
            schedule_changed = scheduled_before != tuple(g.key for g in groups if g.scheduled)
            power_failure = diagnostics.get("transmission_failure", False)
            if power_failure:
                state["power_infeasible_cohort_slots"] += 1
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
                    "slot": slot, "cohort_id": cohort, "scheduled_keys": scheduled_before,
                    "schedule_changed": schedule_changed,
                    "diagnostic": serialize_power_diagnostic(groups, diagnostics),
                }
                if run_dir is not None:
                    atomic_write_json(run_dir / "model_consistency_error.json", failure)
                    atomic_write_json(run_dir / "summary.json", failure)
                tracemalloc.stop()
                raise ModelConsistencyError(
                    f"MODEL_CONSISTENCY_ERROR at slot={slot}, cohort={cohort}: "
                    f"{diagnostics['status']}; the Algorithm-2 scheduled set is unchanged. "
                    "Check fixed class budgets, floors and global SIC; no group was dropped."
                )
        state["scheduler_runtime_s"] += time.perf_counter() - schedule_start
        slot_power_failure_cohorts = sum(d.get("transmission_failure", False) for d in power_diagnostics.values())
        state["power_outage_slots"] += int(slot_power_failure_cohorts > 0)

        transmitted_power = np.zeros(satellite_count, dtype=float)
        for cohort, groups in groups_by_cohort.items():
            satellite = association.serving_satellites[cohort]
            transmitted_power[satellite] = sum(group.power for group in groups if group.scheduled)
        state["max_transmitted_power"] = max(state["max_transmitted_power"], float(np.max(transmitted_power)))

        if handover_mode == "physical_blackout":
            for cohort, event in enumerate(association.handover):
                if event:
                    state["blackout_remaining"][cohort] = blackout_length

        for cohort, groups in groups_by_cohort.items():
            satellite = association.serving_satellites[cohort]
            route = state["routes"][satellite]
            diagnostics = power_diagnostics[cohort]
            for group in groups:
                if group.scheduled and group.power > 0.0:
                    group.interference = expected_cross_link_interference(
                        config,
                        satellite,
                        group.member_indices,
                        path_gains,
                        visible,
                        transmitted_power,
                    )
                    state["interference_samples"].append(group.interference)
            if diagnostics["valid"]:
                decode_groups(config, groups)
            for group in groups:
                if not group.scheduled:
                    continue
                if not route.reachable or not np.all(visible[satellite, group.member_indices]):
                    group.success = False
                if handover_mode == "physical_blackout" and state["blackout_remaining"][cohort] > 0:
                    group.success = False

        successful_devices = np.zeros(len(scenario.devices.ids), dtype=bool)
        for groups in groups_by_cohort.values():
            for group in groups:
                if group.success:
                    successful_devices[group.member_indices] = True
        for cohort, members in enumerate(scenario.subscriptions.cohort_members):
            satellite = association.serving_satellites[cohort]
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
                        handover_event=association.handover[cohort],
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
                        handover_event=association.handover[cohort],
                        handover_delay_slots=kh_slots,
                        handover_mode=handover_mode,
                    )
        if handover_mode == "physical_blackout":
            state["blackout_remaining"] = np.maximum(0, state["blackout_remaining"] - 1)

        state["aoi_samples"][slot] = state["ages"][active]
        weighted_age = float(np.sum(state["ages"][active] * device_weights[active]))
        state["weighted_sum_aoi_samples"][slot] = weighted_age
        state["weighted_age_accumulator"] += weighted_age
        if time_s >= warmup_s:
            state["weighted_age_warm_accumulator"] += weighted_age
            state["warm_slots"] += 1

        slot_success_users = int(np.count_nonzero(successful_devices[active]))
        state["successful_user_updates"] += slot_success_users
        slot_scheduled_groups = sum(
            sum(group.scheduled for group in groups) for groups in groups_by_cohort.values()
        )
        slot_success_groups = sum(
            sum(group.success for group in groups if group.scheduled)
            for groups in groups_by_cohort.values()
        )
        state["min_slot_scheduled_groups"] = min(state["min_slot_scheduled_groups"], slot_scheduled_groups)
        state["max_slot_scheduled_groups"] = max(state["max_slot_scheduled_groups"], slot_scheduled_groups)
        per_satellite_aoi = np.zeros(satellite_count)
        for cohort, members in enumerate(scenario.subscriptions.cohort_members):
            per_satellite_aoi[association.serving_satellites[cohort]] += float(np.sum(state["ages"][members] * device_weights[members]))
        convergence_row = {
            "slot": slot, "time_s": time_s, "policy": policy, "aoi_unit": "slots",
            "regrouping_epoch": regrouping_epoch,
            "is_regrouping_slot": int(is_regrouping_slot),
            "weighted_sum_aoi_total": weighted_age,
            "weighted_sum_aoi_per_satellite_mean": float(np.mean(per_satellite_aoi)),
            "weighted_sum_aoi_per_satellite_max": float(np.max(per_satellite_aoi)),
            "running_mean_weighted_sum_aoi_total": state["weighted_age_accumulator"] / (slot + 1),
            "scheduled_groups": slot_scheduled_groups,
            "successful_groups": slot_success_groups,
            "power_infeasible_cohorts": slot_power_failure_cohorts,
            "transmitted_groups": sum(sum(g.scheduled for g in groups_by_cohort[c])
                                      for c, d in power_diagnostics.items() if d.get("transmission_attempted", False)),
            "action_summary": json.dumps({scenario.constellation.elements[association.serving_satellites[c]].satellite_id: a for c, a in actions_by_cohort.items()}, sort_keys=True),
        }
        state["convergence_rows_pending"].append(convergence_row)
        if collect_slot_metrics:
            convergence_rows.append(dict(convergence_row))
        constellation_edge_groups = sum(group.is_edge for group in all_groups)
        state["scheduled_group_count"] += slot_scheduled_groups
        state["success_group_count"] += slot_success_groups
        state["edge_group_count"] += constellation_edge_groups
        for group in all_groups:
            if not group.scheduled or group.metadata.get("power_failure_no_transmission", False):
                continue
            state["scheduled_rate_samples"].append(group.rate_bps_hz)
            state["rate_threshold_failure_count"] += int(not group.rate_pass)
            state["sic_failure_count"] += int(not group.sic_pass)
            if group.success:
                state["successful_capacity_sum_bits_per_s"] += bandwidth_hz * group.rate_bps_hz
        for cohort, groups in groups_by_cohort.items():
            satellite = association.serving_satellites[cohort]
            scheduled = [group for group in groups if group.scheduled]
            diagnostics = power_diagnostics[cohort]
            success_count = sum(group.success for group in scheduled)
            actual_transmitted_groups = len(scheduled) if diagnostics.get("transmission_attempted", False) else 0
            state["transmitted_group_count"] += actual_transmitted_groups
            edge_count = sum(group.is_edge for group in groups)
            scheduled_users = sum(len(group.member_indices) for group in scheduled)
            state["scheduled_user_opportunities"] += scheduled_users
            state["power_sum"] += transmitted_power[satellite]
            state["power_samples"] += 1
            members = scenario.subscriptions.cohort_members[cohort]
            cohort_weighted_age = float(np.sum(state["ages"][members] * device_weights[members]))
            reward = raw_reward(
                success_count,
                cohort_weighted_age,
                frozen_b_penalty,
                messages,
                subscribers,
                edge_count,
            )
            normalized_reward = None
            epoch_reward = None
            if policy == "mira_sw_ucb":
                normalized_reward, epoch_reward = observe_slot_reward(
                    config, state, cohort=cohort, slot=slot, slot_limit=slot_limit, raw_reward=reward,
                )
                if epoch_reward is not None:
                    state["bandit_observation_rows"].append({
                        "epoch": regrouping_epoch, "cohort_id": cohort,
                        "owner_satellite": state["epoch_action_owner_satellite"][cohort],
                        "action": actions_by_cohort[cohort],
                        "sample_count": state["epoch_reward_count"][cohort],
                        "epoch_reward": epoch_reward,
                    })
            row = {
                "policy": policy,
                "slot": slot,
                "time_s": time_s,
                "cohort_id": cohort,
                "satellite_id": scenario.constellation.elements[satellite].satellite_id,
                "action": actions_by_cohort[cohort],
                "regrouping_epoch": regrouping_epoch,
                "is_regrouping_slot": int(is_regrouping_slot),
                "regrouping_interval_s": float(config.get("algorithm.regrouping_interval_s")),
                "regrouping_interval_slots": interval_slots,
                "sw_ucb_window_epochs": int(config.get("derived.sw_ucb_window_epochs")),
                "action_owner_satellite_id": scenario.constellation.elements[state["epoch_action_owner_satellite"][cohort]].satellite_id,
                "membership_hash": state["held_partition_hash"][cohort],
                "handover": int(association.handover[cohort]),
                "outage_receivers": association.outage_receivers[cohort],
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
                "constellation_edge_groups": constellation_edge_groups,
                "constellation_scheduled_groups": slot_scheduled_groups,
                "constellation_successful_groups": slot_success_groups,
                "cumulative_successful_groups": int(state["success_group_count"]),
                "transmitted_power_normalized": transmitted_power[satellite],
                "weighted_aoi_slots": cohort_weighted_age,
                "aggregated_weighted_sum_aoi_slots": weighted_age,
                "aggregated_weighted_sum_aoi_ms": weighted_age * dt_s * 1000.0,
                "raw_reward": reward,
                "normalized_reward": normalized_reward,
                "epoch_reward": epoch_reward,
                "power_status": power_diagnostics[cohort]["status"],
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
                "regrouping_action_change": regrouping_diagnostics[cohort]["action_change_events"],
                "device_regrouping_events": regrouping_diagnostics[cohort]["regrouping_events"],
                "initial_assignments": regrouping_diagnostics[cohort]["initial_assignments"],
                "handover_assignments": regrouping_diagnostics[cohort]["handover_assignments"],
            }
            state["power_rejected_group_count"] += row["rejected_by_power_groups"]
            state["scheduler_cap_rejected_group_count"] += row["rejected_by_algorithm2_cap"]
            state["nonpositive_priority_rejected_group_count"] += row["unscheduled_nonpositive_priority_groups"]
            state["slot_rows_pending"].append(row)
            if collect_slot_metrics:
                collected_rows.append(dict(row))
            if save_group_trace:
                for group in groups:
                    state["group_rows_pending"].append(
                        {
                            "slot": slot,
                            "regrouping_epoch": regrouping_epoch,
                            "is_regrouping_slot": int(is_regrouping_slot),
                            "cohort_id": cohort,
                            "satellite_id": scenario.constellation.elements[satellite].satellite_id,
                            "message_id": group.message_id,
                            "group_id": group.group_id,
                            "member_hash": hashlib.sha256(group.member_indices.astype("<i4").tobytes()).hexdigest()[:16],
                            "group_size": len(group.member_indices),
                            "gain": group.gain,
                            "max_elevation_deg": group.max_elevation_deg,
                            "is_edge": int(group.is_edge),
                            "scheduled": int(group.scheduled),
                            "priority": group.priority,
                            "scheduler_candidate": int(group.metadata["scheduler_candidate"]),
                            "scheduler_rejection_reason": group.metadata["scheduler_rejection_reason"],
                            "members": json.dumps([int(d) for d in group.member_indices]),
                            "power": group.power,
                            "transmission_failure_power": int(diagnostics.get("transmission_failure", False)),
                            "transmission_attempted": int(group.scheduled and diagnostics.get("transmission_attempted", False)),
                            "target_power": diagnostics["target_powers"].get(id(group), 0.0),
                            "power_floor": diagnostics["floors"].get(id(group), 0.0),
                            "csi_slot_index": slot,
                            "power_csi_slot_index": slot,
                            "interference": group.interference,
                            "sinr": group.sinr,
                            "rate_bps_hz": group.rate_bps_hz,
                            "rate_bps": bandwidth_hz * group.rate_bps_hz,
                            "R_theta_bps_hz": float(config.get("algorithm.R_theta")),
                            "rate_pass": int(group.rate_pass),
                            "sic_pass": int(group.sic_pass),
                            "success": int(group.success),
                        }
                    )

        state["next_slot"] = slot + 1
        if run_dir is not None and (
            len(state["slot_rows_pending"]) >= chunk_size * int(config.get("services.cohort_count"))
            or state["next_slot"] == slot_limit
        ):
            slot_writer.append(state["slot_rows_pending"])
            state["slot_rows_pending"] = []
            convergence_writer.append(state["convergence_rows_pending"])
            state["convergence_rows_pending"] = []
            if group_writer is not None:
                group_writer.append(state["group_rows_pending"])
                state["group_rows_pending"] = []
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
        "status": "INVALID_POWER_OR_SIC" if state["invalid"] else (
            "COMPLETED_WITH_POWER_OUTAGES" if state["power_infeasible_cohort_slots"] else "COMPLETED"),
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
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
        "power_infeasible_cohort_slots": state["power_infeasible_cohort_slots"],
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
        atomic_write_json(run_dir / "summary.json", summary)
        write_csv(run_dir / "bandit_observations.csv", state["bandit_observation_rows"],
                  fieldnames=["epoch", "cohort_id", "owner_satellite", "action", "sample_count", "epoch_reward"])
        csv_to_parquet(run_dir / "_slot_metrics.partial.csv", run_dir / "slot_metrics.parquet")
        if save_group_trace:
            csv_to_parquet(run_dir / "_group_trace.partial.csv", run_dir / "group_trace.parquet")
        Path(run_dir / "_slot_metrics.partial.csv").replace(run_dir / "slot_metrics.csv")
        if save_group_trace:
            Path(run_dir / "_group_trace.partial.csv").replace(run_dir / "group_trace.csv")
        Path(run_dir / "_convergence.partial.csv").replace(run_dir / "convergence_slot_metrics.csv")
        checkpoint_path.unlink(missing_ok=True)
    return {"summary": summary, "slot_metrics": collected_rows, "convergence_metrics": convergence_rows}
