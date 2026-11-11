from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .algorithms import RewardNormalizer, build_group_partition, pilot_fixed_bounds, raw_reward
from .channel import serving_channel_amplitude
from .association import AssociationTracker
from .calibration import _routes, _routes_valid, evaluate_counterfactual_slot
from .config import FIXED_PHI_THETA_DEG, FIXED_B_PENALTY, StudyConfig, save_config, validate_p1_ablation_config
from .errors import ArtifactReadError
from .geometry import link_geometry
from .isl import build_isl_graph
from .scenario import StudyScenario
from .storage import atomic_write_json, write_csv


def reward_context(config: StudyConfig, seed: int) -> str:
    data = config.to_dict()
    data["algorithm"].pop("reward_bounds", None)
    # The pilot uses a held a=4 trajectory and does not depend on a bandit window.
    data["algorithm"].pop("sw_ucb_window_s", None)
    for key in ("derived", "execution", "study"):
        data.pop(key, None)
    # Overhead sensitivity points and fixed baseline choice do not affect this pilot.
    formal = data.get("formal_results", {})
    formal.pop("regrouping_overhead_tau_s", None)
    formal.pop("fixed_action", None)
    data["reward_pilot_seed"] = int(seed)
    data["implementation_protocol"] = "p1_csi_expanded_protected_pool_v7"
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


def apply_reward_bounds(config: StudyConfig, payload: dict, seed: int) -> StudyConfig:
    if payload.get("phi_theta_deg") != float(config.get("algorithm.phi_theta_deg")) or payload.get("B_penalty") != FIXED_B_PENALTY:
        raise ValueError(f"reward bounds must have matching phi={config.get('algorithm.phi_theta_deg'):g}, B={FIXED_B_PENALTY:g}")
    if payload.get("reward_context") != reward_context(config, seed):
        raise ValueError("reward bounds were produced for a different configuration/seed")
    identity = payload.get("reward_identity")
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("reward bounds require a nonempty frozen identity")
    lower, upper = float(payload["reward_lower"]), float(payload["reward_upper"])
    RewardNormalizer(lower, upper, identity)
    return config.with_overrides({"algorithm.reward_bounds": {
        "lower": lower, "upper": upper, "identity": identity, "context": payload["reward_context"],
    }})


def load_reward_bounds_file(config: StudyConfig, path: str | Path, seed: int) -> StudyConfig:
    """Load frozen bounds without replacing an unreadable cache or running a pilot."""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8-sig")
    except OSError as exc:
        cloud_hint = (
            " This path is inside OneDrive; its cloud provider may be unavailable or the file may be online-only."
            if "onedrive" in str(source.absolute()).lower() else ""
        )
        raise ArtifactReadError(
            f"Cannot read reward-bounds file: {source.absolute()}. OS error: {exc}."
            + cloud_hint +
            " Make the file available locally, or pass --reward-bounds with a readable compatible file"
            " and --outputs with a local directory outside OneDrive. No automatic recalibration was performed."
        ) from exc
    except UnicodeError as exc:
        raise ArtifactReadError(f"Reward-bounds file is not valid UTF-8: {source.absolute()}") from exc
    try:
        payload = json.loads(text)
    except (ValueError, UnicodeError) as exc:
        raise ArtifactReadError(
            f"Invalid reward-bounds JSON: {source.absolute()}. Restore a verified calibration file;"
            " no automatic recalibration was performed."
        ) from exc
    if not isinstance(payload, dict):
        raise ArtifactReadError(f"Reward-bounds JSON must be an object: {source.absolute()}")
    return apply_reward_bounds(config, payload, seed)


def run_reward_bounds_pilot(config: StudyConfig, output_directory, *, seed: int = 0) -> dict:
    """Estimate normalization bounds only. Never estimate or change B or phi.

    Replay the existing independent windows with a=4 age trajectories and
    evaluate every legal action on common-age/common-channel snapshots.
    """
    config = config.with_overrides({"randomness.fixed_channel_seed": int(seed)})
    validate_p1_ablation_config(config)
    scenario = StudyScenario(config)
    tracker = AssociationTracker(config, scenario.subscriptions)
    dt = float(config.get("experiment.slot_dt_s"))
    duration = float(config.get("experiment.duration_s"))
    windows = config.get("formal_results.reward_pilot_windows_s", [[0, 30], [225, 255], [450, 480]])
    ranges = []
    for start, end in windows:
        if not (0 <= float(start) < float(end) <= duration):
            raise ValueError("reward pilot windows must lie inside the experiment")
        start_slot, end_slot = int(round(float(start) / dt)), int(round(float(end) / dt))
        if start_slot >= end_slot or (ranges and start_slot < ranges[-1][1]):
            raise ValueError("reward pilot windows must be ordered, nonoverlapping and contain slots")
        ranges.append((start_slot, end_slot))
    interval = float(config.get("formal_results.reward_pilot_snapshot_interval_s", 1.0))
    if not math.isfinite(interval) or interval < dt:
        raise ValueError("reward pilot snapshot interval must be at least one slot")
    stride = max(1, int(round(interval / dt)))
    regroup_slots = int(config.get("derived.regrouping_interval_slots"))
    if abs(interval / dt - stride) > 1e-9 or stride % regroup_slots:
        raise ValueError("reward pilot snapshot interval must be an integer number of regrouping epochs")
    if any(start % regroup_slots or end % regroup_slots for start, end in ranges):
        raise ValueError("reward pilot windows must align with regrouping boundaries")
    route_stride = max(1, int(round(float(config.get("delay.isl_route_update_s")) / dt)))
    routes = None
    samples = []
    ages = None
    held_baseline_partition = None
    for slot in range(ranges[-1][1]):
        _, ecef = scenario.constellation.state(slot * dt)
        distance, elevation = link_geometry(ecef, scenario.devices.ecef_km)
        association = tracker.update(slot, elevation)
        window = next((i for i, (start, end) in enumerate(ranges) if start <= slot < end), None)
        if window is None:
            continue
        start, _ = ranges[window]
        if slot == start:
            ages = np.full(len(scenario.devices.ids), np.nan)
            ages[scenario.subscriptions.active_device_ids] = float(config.get("services.initial_age_slots"))
            held_baseline_partition = None
        current_channel = serving_channel_amplitude(config, seed, slot, distance)
        if slot % regroup_slots == 0 or held_baseline_partition is None:
            held_baseline_partition = build_group_partition(
                config, association.serving_satellite, 4,
                scenario.subscriptions.message_members, current_channel)

        graph = build_isl_graph(ecef, float(config.get("geometry.earth_radius_km")), float(config.get("delay.isl_max_link_distance_km")))
        if slot % route_stride == 0 or not _routes_valid(routes, graph):
            routes = _routes(config, graph)
        kwargs = dict(slot=slot, seed=seed, ages_before=ages, association=association,
                      distance=distance, elevation=elevation, routes=routes, channel_amplitude=current_channel)
        if (slot - start) % stride == 0:
            for action in config.get("algorithm.action_set"):
                result = evaluate_counterfactual_slot(config, scenario, action=action, **kwargs)
                for row in result["rows"]:
                    if not row["feasible"] and not row.get("transmission_failure_power", False):
                        raise RuntimeError("strict Algorithm-2 reward pilot produced an invalid allocation")
                    reward = raw_reward(row["success_group_count"], row["weighted_post_aoi_slots"],
                                        0.05, int(config.get("services.messages_per_active_satellite")),
                                        int(config.get("services.subscribers_per_message")), row["edge_group_count"])
                    samples.append({"slot": slot, "seed": seed, "window": window, **row, "raw_reward": reward})
        ages = evaluate_counterfactual_slot(config, scenario, action=4,
                                           partition=held_baseline_partition, **kwargs)["ages_after"]
    lower, upper = pilot_fixed_bounds(np.asarray([r["raw_reward"] for r in samples]))
    context = reward_context(config, seed)
    identity = hashlib.sha256(json.dumps([context, lower, upper]).encode("ascii")).hexdigest()
    payload = {
        "total_ue": 120, "message_count": 3, "subscribers_per_message": 40,
        "serving_satellite_count": 1, "interference_mode": "fixed_full_load",
        "interference_model": config.get("channel.external_interference"),
        "interferer_power_normalized": float(config.get("channel.normalized_satellite_power")),
        "phi_theta_deg": float(config.get("algorithm.phi_theta_deg")), "B_penalty": FIXED_B_PENALTY,
        "power_mode": config.get("algorithm.power_mode"), "power_pool_rule_version": config.get("algorithm.power_pool_rule_version"),
        "reward_lower": lower, "reward_upper": upper, "reward_identity": identity,
        "reward_context": context, "seed": int(seed), "sample_count": len(samples),
        "windows_s": windows, "snapshot_interval_s": interval,
        "action_set": list(config.get("algorithm.action_set")),
        "regrouping_interval_s": float(config.get("algorithm.regrouping_interval_s")),
        "scheduler_algorithm": "Algorithm2_strict", "power_allocator": config.get("algorithm.power_mode"),
        "infeasible_power_handling": str(config.get("algorithm.infeasible_power_handling")),
        "power_infeasible_samples": sum(r.get("transmission_failure_power", False) for r in samples),
    }
    resolved = apply_reward_bounds(config, payload, seed)
    root = Path(output_directory) / "reward_bounds"
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "reward_bounds.json", payload)
    write_csv(root / "pilot_reward_samples.csv", samples)
    save_config(resolved, root / "resolved_config.json")
    return {"config": resolved, "calibration": payload, "path": str(root / "reward_bounds.json")}
