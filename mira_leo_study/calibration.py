"""Shared counterfactual evaluator and archived phi/B calibration APIs.

The fixed formal workflow uses only evaluate_counterfactual_slot and route
helpers here; reward_pilot.py never calls the archived B estimators.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .algorithms import allocate_power, decode_groups, make_groups, materialize_groups_from_partition, pilot_fixed_bounds, schedule_groups
from .association import AssociationTracker
from .channel import expected_cross_link_interference, external_interferer_powers, path_gain, serving_channel_amplitude
from .config import PHI_THETA_CANDIDATES_DEG, StudyConfig, save_config, validate_phi_study_config
from .geometry import link_geometry
from .errors import ModelConsistencyError
from .isl import build_isl_graph, shortest_route, route_delay_s
from .scenario import StudyScenario
from .storage import atomic_write_json, write_csv, write_parquet


PILOT_WINDOWS_S = ((0.0, 30.0), (225.0, 255.0), (450.0, 480.0))


def _routes(config, graph):
    ingress = int(str(config.get("delay.ingress_satellite")).split("_")[-1]) - 1
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


def _routes_valid(routes, graph) -> bool:
    if routes is None:
        return False
    edges = {(left, right) for left, neighbors in graph.items() for right, _ in neighbors}
    return all(
        not route.reachable
        or all((left, right) in edges for left, right in zip(route.path, route.path[1:]))
        for route in routes.values()
    )


def evaluate_counterfactual_slot(
    config: StudyConfig,
    scenario: StudyScenario,
    *,
    slot: int,
    seed: int,
    ages_before: np.ndarray,
    association,
    distance: np.ndarray,
    elevation: np.ndarray,
    routes,
    action: int,
    partition: tuple | None = None,
    channel_amplitude: np.ndarray | None = None,
) -> dict:
    """Evaluate one common-age/common-channel action without mutating caller state."""
    dt_s = float(config.get("experiment.slot_dt_s"))
    c_km_s = float(config.get("geometry.light_speed_km_s"))
    channel = serving_channel_amplitude(config, seed, slot, distance) if channel_amplitude is None else channel_amplitude
    gains = path_gain(
        distance,
        float(config.get("channel.carrier_hz")),
        c_km_s,
    )
    visible = elevation >= float(config.get("geometry.phi_min_deg"))
    satellite = association.serving_satellite
    if partition is None:
        groups = make_groups(config, satellite, action,
                             scenario.subscriptions.message_members, channel, elevation)
    else:
        groups = materialize_groups_from_partition(config, satellite,
                                                   partition, channel, elevation)
    route = routes[satellite]
    delay_s = route_delay_s(config, route)
    delay_slots = delay_s / dt_s if math.isfinite(delay_s) else math.inf
    schedule_groups(
        config,
        groups,
        ages_before,
        distance,
        association.handover,
        delay_slots,
    )
    diagnostics = allocate_power(config, groups)
    if (config.get("algorithm.power_mode") == "csi_expanded_protected_pool" and not diagnostics["valid"]
            and not diagnostics.get("transmission_failure", False)):
        raise ModelConsistencyError(
            f"reward pilot power/SIC inconsistency at slot={slot}, action={action}; "
            "scheduled groups are not dropped and invalid samples are not calibrated"
        )

    external_interferer_power_by_satellite = external_interferer_powers(
        config, len(scenario.constellation.elements), association.serving_satellite)
    satellite = association.serving_satellite
    route = routes[satellite]
    # External transmissions persist even if the evaluated serving plan fails.
    for group in groups:
        group.interference = expected_cross_link_interference(
            config, satellite, group.member_indices, gains, visible, external_interferer_power_by_satellite)
    if diagnostics["valid"]:
        decode_groups(config, groups)
    for group in groups:
        if group.scheduled and (
            not route.reachable
            or not np.all(visible[satellite, group.member_indices])
        ):
            group.success = False

    successful = np.zeros(len(scenario.devices.ids), dtype=bool)
    for group in groups:
        if group.success:
            successful[group.member_indices] = True
    ages_after = ages_before.copy()
    members = scenario.subscriptions.active_device_ids
    satellite = association.serving_satellite
    route = routes[satellite]
    delay_s = route_delay_s(config, route)
    delay_slots = delay_s / dt_s if math.isfinite(delay_s) else math.inf
    for device in members:
        if successful[device]:
            ages_after[device] = (
                float(config.get("services.delta_c_slots"))
                + distance[satellite, device] / (c_km_s * dt_s)
                + delay_slots
            )
        else:
            ages_after[device] += 1.0 + float(
                config.get("derived.handover_delay_slots")
            ) * float(association.handover)

    weights = np.asarray(config.get("services.weights"), dtype=float)
    rows = []
    weighted_aoi = 0.0
    for message in range(int(config.get("services.messages_per_active_satellite"))):
        members = scenario.subscriptions.message_members[message]
        weighted_aoi += weights[message] * float(np.sum(ages_after[members]))
    success_count = sum(group.success for group in groups if group.scheduled)
    edge_count = sum(group.is_edge for group in groups)
    rows.append(
        {
            
            "satellite_id": scenario.constellation.elements[
                association.serving_satellite
            ].satellite_id,
            "action": action,
            "q": float(success_count - weighted_aoi),
            "edge_group_count": edge_count,
            "success_group_count": success_count,
            "weighted_post_aoi_slots": weighted_aoi,
            "feasible": bool(diagnostics["valid"]),
            "power_status": diagnostics["status"],
            "transmission_failure_power": bool(diagnostics.get("transmission_failure", False)),
            "transmission_attempted": bool(diagnostics.get("transmission_attempted", False)),
            "actual_transmitted_power": diagnostics.get("actual_total_power", 0.0),
            "power_solver_status": diagnostics.get("projection_solver_status"),
        }
    )
    return {"ages_after": ages_after, "rows": rows}


def _derive_calibration(config: StudyConfig, samples: list[dict]) -> dict:
    actions = list(config.get("algorithm.action_set"))
    grouped = defaultdict(dict)
    for row in samples:
        key = (row["seed"], row["window_id"], row["snapshot_slot"])
        grouped[key][row["action"]] = row
    ratios = []
    for key, by_action in grouped.items():
        for left, right in zip(actions, actions[1:]):
            if left not in by_action or right not in by_action:
                continue
            a = by_action[left]
            b = by_action[right]
            delta_q = float(b["q"] - a["q"])
            delta_g = int(b["edge_group_count"] - a["edge_group_count"])
            if a["feasible"] and b["feasible"] and delta_q != 0.0 and delta_g != 0:
                ratios.append(
                    {
                        "phi_theta_deg": float(config.get("algorithm.phi_theta_deg")),
                        "seed": key[0],
                        "window_id": key[1],
                        "snapshot_slot": key[2],
                        "left_action": left,
                        "right_action": right,
                        "delta_q": delta_q,
                        "delta_edge_groups": delta_g,
                        "ratio": abs(delta_q)
                        / (
                            int(config.get("services.messages_per_active_satellite"))
                            * int(config.get("services.subscribers_per_message"))
                            * abs(delta_g)
                        ),
                    }
                )
    identifiable = bool(ratios)
    if not identifiable:
        return {
            "phi_theta_deg": float(config.get("algorithm.phi_theta_deg")),
            "B0": None,
            "B_identifiable": False,
            "B_status": "B_NOT_IDENTIFIABLE_FROM_PILOT",
            "B_ratio_sample_count": 0,
            "ratio_sample_count": 0,
            "B_penalty": None,
            "B_penalty_star": None,
            "selection_quantile": float(config.get("algorithm.B0_calibration_quantile")),
            "beta_B": float(config.get("algorithm.beta_B")),
            "ratio_q50": None,
            "ratio_q75": None,
            "ratio_q90": None,
            "ratio_q95": None,
            "ratio_max": None,
            "reward_lower": None,
            "reward_upper": None,
            "reward_identity": None,
            "pilot_reward_sample_count": 0,
            "fixed_channel_seed": int(config.get("randomness.fixed_channel_seed")),
            "formal_B_is_frozen_within_this_phi": True,
            "ratios": [],
        }
    ratio_values = np.asarray([row["ratio"] for row in ratios], dtype=float)
    quantile = float(config.get("algorithm.B0_calibration_quantile"))
    b0 = float(np.quantile(ratio_values, quantile))
    b_penalty = b0 * float(config.get("algorithm.beta_B"))
    multiplier = (
        int(config.get("services.messages_per_active_satellite"))
        * int(config.get("services.subscribers_per_message"))
    )
    rewards = np.asarray(
        [
            row["q"] - b_penalty * multiplier * row["edge_group_count"]
            for row in samples
            if row["feasible"]
        ],
        dtype=float,
    )
    lower, upper = pilot_fixed_bounds(rewards)
    identity_source = {
        "config_id_before_calibration": config.config_id,
        "fixed_channel_seed": int(config.get("randomness.fixed_channel_seed")),
        "phi_theta_deg": float(config.get("algorithm.phi_theta_deg")),
        "b0": b0,
        "lower": lower,
        "upper": upper,
    }
    identity = hashlib.sha256(
        json.dumps(identity_source, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "phi_theta_deg": float(config.get("algorithm.phi_theta_deg")),
        "B0": b0,
        "B_identifiable": identifiable,
        "B_status": "CALIBRATED" if identifiable else "B_NOT_IDENTIFIABLE_FROM_PILOT",
        "B_ratio_sample_count": len(ratios),
        "ratio_sample_count": len(ratios),
        "B_penalty": b_penalty,
        "B_penalty_star": b_penalty,
        "selection_quantile": quantile,
        "beta_B": float(config.get("algorithm.beta_B")),
        "ratio_q50": float(np.quantile(ratio_values, 0.50)),
        "ratio_q75": float(np.quantile(ratio_values, 0.75)),
        "ratio_q90": float(np.quantile(ratio_values, 0.90)),
        "ratio_q95": float(np.quantile(ratio_values, 0.95)),
        "ratio_max": float(np.max(ratio_values)),
        "reward_lower": lower,
        "reward_upper": upper,
        "reward_identity": identity,
        "pilot_reward_sample_count": int(rewards.size),
        "fixed_channel_seed": int(config.get("randomness.fixed_channel_seed")),
        "formal_B_is_frozen_within_this_phi": True,
        "ratios": ratios,
    }


def run_calibration(
    config: StudyConfig,
    output_directory: str | Path,
    *,
    phi_output_layout: bool = False,
) -> dict:
    """Execute independent pilot windows and freeze B0/reward normalization."""
    if config.get("delay.handover_mode") != "paper_additive":
        raise ValueError("main pilot calibration is defined for paper_additive handover mode")
    pilot_config = config.with_overrides(
        {
            "algorithm.B_penalty": 0.0,
            "algorithm.reward_normalization": "finite_horizon_bound",
        }
    )
    scenario = StudyScenario(pilot_config)
    tracker = AssociationTracker(pilot_config, scenario.subscriptions)
    dt_s = float(pilot_config.get("experiment.slot_dt_s"))
    total_slots = pilot_config.slot_count
    window_slots = [
        (int(round(start / dt_s)), int(round(end / dt_s)))
        for start, end in PILOT_WINDOWS_S
    ]
    fixed_seed = int(pilot_config.get("randomness.fixed_channel_seed", 0))
    configured_seed_lists = [
        pilot_config.get(name)
        for name in (
            "randomness.pilot_channel_seeds",
            "randomness.validation_channel_seeds",
            "randomness.test_channel_seeds",
        )
        if pilot_config.get(name) is not None
    ]
    if any(len(list(values)) != 1 or int(list(values)[0]) != fixed_seed for values in configured_seed_lists):
        raise ValueError("phi-theta calibration requires exactly fixed_channel_seed=0")
    pilot_seeds = [fixed_seed]
    actions = [int(value) for value in pilot_config.get("algorithm.action_set")]
    baseline_action = 4
    if baseline_action not in actions:
        raise ValueError("pilot calibration requires fixed action a=4")
    ages_by_seed: dict[int, np.ndarray] = {}
    samples: list[dict] = []
    route_cache = None
    route_update_slots = max(1, int(round(float(pilot_config.get("delay.isl_route_update_s")) / dt_s)))

    for slot in range(total_slots):
        _, satellite_ecef = scenario.constellation.state(slot * dt_s)
        distance, elevation = link_geometry(satellite_ecef, scenario.devices.ecef_km)
        association = tracker.update(slot, elevation)
        window_id = next(
            (index for index, (start, end) in enumerate(window_slots) if start <= slot < end),
            None,
        )
        if window_id is None:
            continue
        start_slot, _ = window_slots[window_id]
        if slot == start_slot:
            ages_by_seed = {}
            for seed in pilot_seeds:
                ages = np.full(len(scenario.devices.ids), np.nan, dtype=float)
                ages[scenario.subscriptions.active_device_ids] = float(
                    pilot_config.get("services.initial_age_slots")
                )
                ages_by_seed[seed] = ages
        graph = build_isl_graph(
            satellite_ecef,
            float(pilot_config.get("geometry.earth_radius_km")),
            float(pilot_config.get("delay.isl_max_link_distance_km")),
        )
        if route_cache is None or slot % route_update_slots == 0 or not _routes_valid(route_cache, graph):
            route_cache = _routes(pilot_config, graph)
        snapshot = (slot - start_slot) % route_update_slots == 0
        for seed in pilot_seeds:
            ages = ages_by_seed[seed]
            if snapshot:
                for action in actions:
                    result = evaluate_counterfactual_slot(
                        pilot_config,
                        scenario,
                        slot=slot,
                        seed=seed,
                        ages_before=ages,
                        association=association,
                        distance=distance,
                        elevation=elevation,
                        routes=route_cache,
                        action=action,
                    )
                    for row in result["rows"]:
                        samples.append(
                            {
                                "seed": seed,
                                "window_id": window_id,
                                "snapshot_slot": slot,
                                "snapshot_time_s": slot * dt_s,
                                "phi_theta_deg": float(pilot_config.get("algorithm.phi_theta_deg")),
                                **row,
                            }
                        )
            baseline = evaluate_counterfactual_slot(
                pilot_config,
                scenario,
                slot=slot,
                seed=seed,
                ages_before=ages,
                association=association,
                distance=distance,
                elevation=elevation,
                routes=route_cache,
                action=baseline_action,
            )
            ages_by_seed[seed] = baseline["ages_after"]

    calibration = _derive_calibration(config, samples)
    resolved = None
    if calibration["B_identifiable"]:
        resolved = config.with_overrides(
            {
                "algorithm.B_penalty": calibration["B_penalty_star"],
                "algorithm.reward_bounds": {
                    "lower": calibration["reward_lower"],
                    "upper": calibration["reward_upper"],
                    "identity": calibration["reward_identity"],
                },
            }
        )
    output = Path(output_directory) if phi_output_layout else Path(output_directory) / "calibration"
    output.mkdir(parents=True, exist_ok=True)
    ratios = calibration.pop("ratios")
    if phi_output_layout:
        write_parquet(output / "pilot_counterfactuals.parquet", samples)
        write_csv(
            output / "B_ratio_samples.csv",
            ratios,
            fieldnames=[
                "phi_theta_deg",
                "seed",
                "window_id",
                "snapshot_slot",
                "left_action",
                "right_action",
                "delta_q",
                "delta_edge_groups",
                "ratio",
            ],
        )
        atomic_write_json(output / "B_star.json", calibration)
        atomic_write_json(
            output / "resolved_reward_bounds.json",
            {
                "phi_theta_deg": calibration["phi_theta_deg"],
                "lower": calibration["reward_lower"],
                "upper": calibration["reward_upper"],
                "identity": calibration["reward_identity"],
                "status": calibration["B_status"],
            },
        )
    else:
        write_parquet(output / "b_penalty_samples.parquet", samples)
        write_parquet(output / "b_penalty_ratios.parquet", ratios)
        atomic_write_json(output / "calibration.json", calibration)
    if resolved is not None:
        save_config(resolved, output / "resolved_config.json")
    return {"calibration": calibration, "resolved_config": resolved}


def run_per_phi_B_pilots(
    config: StudyConfig,
    phi_candidates: list[int] | tuple[int, ...] | None,
    output_directory: str | Path,
) -> dict:
    """Calibrate one independent frozen B and reward range for each edge angle."""
    validate_phi_study_config(config)
    candidates = list(PHI_THETA_CANDIDATES_DEG if phi_candidates is None else phi_candidates)
    if candidates != PHI_THETA_CANDIDATES_DEG:
        raise ValueError(f"phi candidates must be exactly {PHI_THETA_CANDIDATES_DEG}")
    root = Path(output_directory) / "pilot_B"
    root.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "fixed_channel_seed": int(config.get("randomness.fixed_channel_seed")),
        "phi_theta_candidates_deg": candidates,
        "slot_dt_s": float(config.get("experiment.slot_dt_s")),
        "duration_s": float(config.get("experiment.duration_s")),
        "warmup_report_s": float(config.get("experiment.warmup_report_s")),
        "carrier_hz": float(config.get("channel.carrier_hz")),
        "bandwidth_hz": float(config.get("services.bandwidth_hz")),
        "reference_bandwidth_hz": float(config.get("channel.reference_bandwidth_hz")),
        "scs_hz": float(config.get("radio.scs_hz")),
        "R_theta_bps_hz": float(config.get("algorithm.R_theta")),
        "V_c": float(config.get("algorithm.V_c")),
        "V_nc": float(config.get("algorithm.V_nc")),
        "V_phi": float(config.get("algorithm.V_phi")),
        "normalized_satellite_power": float(
            config.get("channel.normalized_satellite_power")
        ),
        "messages_per_active_satellite": int(
            config.get("services.messages_per_active_satellite")
        ),
        "subscribers_per_message": int(config.get("services.subscribers_per_message")),
        "weights": list(config.get("services.weights")),
        "action_set": list(config.get("algorithm.action_set")),
        "sw_ucb_xi": float(config.get("algorithm.sw_ucb_xi")),
        "sw_ucb_window_s": float(config.get("algorithm.sw_ucb_window_s")),
        "phi_min_deg": float(config.get("geometry.phi_min_deg")),
        "handover_delay_slots": float(config.get("derived.handover_delay_slots")),
        "fixed_isl_per_hop_slots": float(config.get("derived.fixed_isl_per_hop_slots")),
        "shadowed_rician": config.get("channel.shadowed_rician"),
        "schedule_capacity_continuous": float(
            config.get("derived.schedule_capacity_continuous")
        ),
        "schedule_capacity_integer": int(config.get("derived.schedule_capacity_integer")),
        "schedule_capacity_paper_approximation": int(
            config.get("derived.schedule_capacity_paper_approximation")
        ),
        "selection_quantile": float(config.get("algorithm.B0_calibration_quantile")),
        "selection_quantile_note": "empirical implementation choice; not specified by the paper",
        "B_semantics": "edge-count penalty coefficient, distinct from radio bandwidth",
    }
    atomic_write_json(root / "pilot_parameter_snapshot.json", snapshot)
    rows = []
    entries = {}
    for phi in candidates:
        angle_config = config.with_overrides({"algorithm.phi_theta_deg": float(phi)})
        result = run_calibration(
            angle_config,
            root / f"phi_{phi}",
            phi_output_layout=True,
        )
        calibration = result["calibration"]
        row = {
            "phi_theta_deg": phi,
            "B_penalty_star": calibration["B_penalty_star"],
            "ratio_q50": calibration["ratio_q50"],
            "ratio_q75": calibration["ratio_q75"],
            "ratio_q90": calibration["ratio_q90"],
            "ratio_q95": calibration["ratio_q95"],
            "ratio_max": calibration["ratio_max"],
            "ratio_sample_count": calibration["B_ratio_sample_count"],
            "reward_lower": calibration["reward_lower"],
            "reward_upper": calibration["reward_upper"],
            "reward_bounds_identity": calibration["reward_identity"],
            "status": "OK" if calibration["B_identifiable"] else calibration["B_status"],
        }
        rows.append(row)
        entries[str(phi)] = row
    write_csv(
        root / "B_by_phi.csv",
        rows,
        fieldnames=list(rows[0].keys()),
    )
    atomic_write_json(
        root / "B_by_phi.json",
        {
            "fixed_channel_seed": int(config.get("randomness.fixed_channel_seed")),
            "selection_quantile": float(config.get("algorithm.B0_calibration_quantile")),
            "B_penalty_star_by_phi": entries,
        },
    )
    return {"B_penalty_star_by_phi": entries, "output_directory": str(root)}
