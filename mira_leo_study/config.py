from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigurationError, OptionalDependencyError


ACTION_SETS = {
    "full": [1, 2, 4, 5, 8, 10, 20, 40],
    "compact": [1, 2, 4, 5, 8, 10],
    "coarse": [1, 2, 4, 8],
}
FIXED_PHI_THETA_DEG = 15.0
P1_PHI_THETA_DEG = [13.0, 15.0, 20.0, 25.0]
from .protected_power import POWER_MODE, POOL_RULE_VERSION
FIXED_B_PENALTY = 0.05

# Archived APIs/tests only. The formal pipeline uses validate_fixed_study_config.
PHI_THETA_CANDIDATES_DEG = [20, 25, 30, 35, 40, 45, 50]


def _deep_merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise OptionalDependencyError(
                "Non-JSON YAML requires PyYAML. Install requirements.txt or use the "
                "bundled JSON-compatible configs/base.yaml."
            ) from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ConfigurationError("configuration root must be a mapping")
    return value


@dataclass(frozen=True)
class StudyConfig:
    data: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        value = self.data.get(name)
        if not isinstance(value, dict):
            raise ConfigurationError(f"missing configuration section: {name}")
        return value

    def get(self, dotted_path: str, default: Any = None) -> Any:
        value: Any = self.data
        for part in dotted_path.split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value

    def with_overrides(self, overrides: Mapping[str, Any]) -> "StudyConfig":
        merged = copy.deepcopy(self.data)
        for dotted_path, value in overrides.items():
            parts = dotted_path.split(".")
            cursor = merged
            for part in parts[:-1]:
                cursor = cursor.setdefault(part, {})
            cursor[parts[-1]] = copy.deepcopy(value)
        return resolve_config(merged)

    @property
    def config_id(self) -> str:
        identity = copy.deepcopy(self.data)
        identity.pop("execution", None)
        identity.pop("study", None)
        return hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest()[:16]

    @property
    def slot_count(self) -> int:
        return int(round(self.get("experiment.duration_s") / self.get("experiment.slot_dt_s")))

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)


def resolve_config(raw: Mapping[str, Any]) -> StudyConfig:
    data = copy.deepcopy(dict(raw))
    exp = data["experiment"]
    geometry = data["geometry"]
    constellation = data["constellation"]
    services = data["services"]
    radio = data.get("radio")
    channel = data["channel"]
    delay = data["delay"]
    algorithm = data["algorithm"]
    vc, vnc = float(algorithm["V_c"]), float(algorithm["V_nc"])
    if not all(math.isfinite(v) and v > 0 for v in (vc, vnc)) or abs(vc + vnc - 1.0) > 1e-12:
        raise ConfigurationError("V_c and V_nc must be finite, positive and sum to one")
    for path, value in (
        ("algorithm.V_phi", algorithm["V_phi"]),
        ("channel.normalized_satellite_power", channel["normalized_satellite_power"]),
        ("algorithm.sw_ucb_window_s", algorithm["sw_ucb_window_s"]),
        ("algorithm.regrouping_interval_s", algorithm.get("regrouping_interval_s", exp["slot_dt_s"])),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ConfigurationError(f"{path} must be finite and positive")

    slot_dt = float(exp["slot_dt_s"])
    duration = float(exp["duration_s"])
    if slot_dt <= 0.0 or duration <= 0.0:
        raise ConfigurationError("duration_s and slot_dt_s must be positive")
    slots_exact = duration / slot_dt
    if abs(slots_exact - round(slots_exact)) > 1e-9:
        raise ConfigurationError("duration_s must be an integer multiple of slot_dt_s")
    if not isinstance(radio, dict) or float(radio.get("scs_hz", 0.0)) <= 0.0:
        raise ConfigurationError("radio.scs_hz must be positive")

    earth_radius = float(geometry["earth_radius_km"])
    altitude = float(constellation["altitude_km"])
    derived = data.setdefault("derived", {})
    derived["semi_major_axis_km"] = earth_radius + altitude
    derived["mean_motion_rad_s"] = math.sqrt(
        float(geometry["mu_km3_s2"]) / derived["semi_major_axis_km"] ** 3
    )
    derived["orbital_period_s"] = 2.0 * math.pi / derived["mean_motion_rad_s"]
    derived["slot_count"] = int(round(slots_exact))
    derived["handover_delay_slots"] = float(delay["handover_s"]) / slot_dt
    derived["fixed_isl_per_hop_slots"] = float(delay["fixed_isl_per_hop_s"]) / slot_dt
    derived["handover_ttt_slots"] = int(round(float(data["association"]["time_to_trigger_s"]) / slot_dt))
    derived["sw_ucb_window_slots"] = int(round(float(algorithm["sw_ucb_window_s"]) / slot_dt))
    # Legacy configurations without this field retain per-slot regrouping.
    interval = float(algorithm.setdefault("regrouping_interval_s", slot_dt))
    interval_exact = interval / slot_dt
    window_exact = float(algorithm["sw_ucb_window_s"]) / interval
    if abs(interval_exact - round(interval_exact)) > 1e-9 or round(interval_exact) < 1:
        raise ConfigurationError("algorithm.regrouping_interval_s must be an integer multiple of experiment.slot_dt_s")
    if abs(window_exact - round(window_exact)) > 1e-9 or round(window_exact) < 1:
        raise ConfigurationError("algorithm.sw_ucb_window_s must be an integer multiple of algorithm.regrouping_interval_s")
    derived["regrouping_interval_slots"] = int(round(interval_exact))
    derived["sw_ucb_window_epochs"] = int(round(window_exact))
    sr = channel["shadowed_rician"]
    sr_mean = 2.0 * float(sr["b0"]) + float(sr["Omega"])
    derived["shadowed_rician_mean_power"] = sr_mean
    c_km_s = float(geometry["light_speed_km_s"])
    fc_hz = float(channel["carrier_hz"])
    zeta_ref = (c_km_s / (4.0 * math.pi * altitude * fc_hz)) ** 2
    derived["reference_path_gain"] = zeta_ref
    derived["noise_power"] = (
        sr_mean
        * zeta_ref
        / (10.0 ** (float(channel["reference_snr_db"]) / 10.0))
        * float(services["bandwidth_hz"])
        / float(channel["reference_bandwidth_hz"])
    )
    schedule_capacity_continuous = -0.5 + math.sqrt(
        0.25
        + 2.0
        * float(channel["normalized_satellite_power"])
        / float(algorithm["V_phi"])
    )
    derived["schedule_capacity_continuous"] = schedule_capacity_continuous
    derived["schedule_capacity_integer"] = math.floor(schedule_capacity_continuous)
    derived["schedule_capacity_paper_approximation"] = round(schedule_capacity_continuous)
    derived["schedule_capacity"] = derived["schedule_capacity_integer"]
    derived["schedule_capacity_single_class_floor_only"] = derived["schedule_capacity_integer"]
    for name, fraction in (("edge", algorithm["V_c"]), ("nonedge", algorithm["V_nc"])):
        derived[f"schedule_capacity_{name}_when_mixed_floor_only"] = math.floor(
            -0.5 + math.sqrt(0.25 + 2.0 * float(channel["normalized_satellite_power"]) * float(fraction) / float(algorithm["V_phi"]))
        )
    derived["algorithm2_U"] = derived["schedule_capacity_integer"]
    derived["schedule_capacity_semantics"] = "Algorithm 2 per-satellite cap U; mixed 52/34 limits are diagnostics only"

    action_name = str(algorithm["action_set_name"])
    if action_name not in ACTION_SETS:
        raise ConfigurationError(f"unknown action set: {action_name}")
    action_values = algorithm.get("action_set", ACTION_SETS[action_name])
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in action_values):
        raise ConfigurationError("grouping actions must be positive integers")
    configured_actions = list(action_values)
    if configured_actions != ACTION_SETS[action_name]:
        raise ConfigurationError("action_set content does not match action_set_name")
    for section, key in ((data["devices"], "unique_count"),
                         (services, "messages_per_active_satellite"),
                         (services, "subscribers_per_message")):
        value = section[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigurationError(f"{key} must be a positive integer")
    subscriber_count = services["subscribers_per_message"]
    derived["legal_actions"] = [
        value for value in range(1, subscriber_count + 1) if subscriber_count % value == 0
    ]
    if any(action <= 0 or subscriber_count % action != 0 for action in configured_actions):
        raise ConfigurationError("every action must evenly divide subscribers_per_message")

    if int(constellation["actual_satellite_count"]) != 8:
        raise ConfigurationError("this synthetic study requires exactly 8 active satellites")
    offsets1 = constellation["plane1_phase_offsets_deg"]
    offsets2 = constellation["plane2_phase_offsets_deg"]
    if len(offsets1) != 5 or len(offsets2) != 3:
        raise ConfigurationError("the two planes must contain 5 and 3 satellites")
    if "cohort_count" in services:
        raise ConfigurationError("obsolete cohort_count: use the single-serving message-only schema")
    if (data["devices"]["unique_count"] != 120
            or services["messages_per_active_satellite"] != 3 or subscriber_count != 40
            or data["devices"]["unique_count"] != services["messages_per_active_satellite"] * subscriber_count):
        raise ConfigurationError("the study requires 120 devices, 3 disjoint messages and 40 subscribers per message")
    if data["association"]["mode"] != "single_serving_satellite_for_all_devices":
        raise ConfigurationError("all devices must share one serving satellite")
    if data["association"]["objective"] != "coverage_first_then_minimum_elevation":
        raise ConfigurationError("association must prioritize coverage then minimum elevation")
    if data["association"].get("initial_acquisition_is_handover", False):
        raise ConfigurationError("initial acquisition must not count as handover")
    if not data["devices"].get("stationary"):
        raise ConfigurationError("the 120 devices must be stationary")
    if channel.get("external_interference") != "always_on_non_serving_satellites_expected_cross_link_power":
        raise ConfigurationError("external interference must use expected non-serving cross-link powers")
    if channel.get("interferer_power_mode") != "fixed_full_load":
        raise ConfigurationError("external interferers must use fixed_full_load power")
    if float(channel["normalized_satellite_power"]) != 1.0:
        raise ConfigurationError("fixed full-load interferer power must be 1.0")
    weights = [float(value) for value in services["weights"]]
    if len(weights) != int(services["messages_per_active_satellite"]) or any(value < 0 for value in weights):
        raise ConfigurationError("services.weights must be nonnegative and match message count")
    vc = float(algorithm["V_c"])
    vnc = float(algorithm["V_nc"])
    if vc <= 0.0 or vnc <= 0.0 or abs(vc + vnc - 1.0) > 1e-12:
        raise ConfigurationError("V_c and V_nc must be positive and sum to one")
    if not math.isfinite(float(algorithm["phi_theta_deg"])) or not 0 < float(algorithm["phi_theta_deg"]) <= 90:
        raise ConfigurationError("phi_theta_deg must be finite and in (0,90]")
    if algorithm["power_mode"] not in {POWER_MODE, "twc_aim_staircase", "literal_with_checks", "constraint_projection"}:
        raise ConfigurationError("unsupported power_mode")
    if algorithm.get("infeasible_power_handling", "error") not in {"error", "transmission_failure"}:
        raise ConfigurationError("infeasible_power_handling must be error or transmission_failure")
    if delay["handover_mode"] not in {"paper_additive", "physical_blackout"}:
        raise ConfigurationError("unsupported handover_mode")
    if delay["isl_mode"] not in {"fixed_per_hop", "geometric_isl"}:
        raise ConfigurationError("unsupported isl_mode")
    return StudyConfig(data)


def validate_phi_study_config(config: StudyConfig) -> None:
    """Reject drift from the fixed physical setup of the phi-theta study."""
    expected = {
        "devices.unique_count": 120,
        "constellation.actual_satellite_count": 8,
        "constellation.altitude_km": 800.0,
        "experiment.slot_dt_s": 0.01,
        "experiment.duration_s": 480.0,
        "experiment.warmup_report_s": 10.0,
        "services.bandwidth_hz": 50_000_000.0,
        "services.messages_per_active_satellite": 3,
        "services.subscribers_per_message": 40,
        "channel.reference_bandwidth_hz": 50_000_000.0,
        "channel.carrier_hz": 20_000_000_000.0,
        "channel.normalized_satellite_power": 1.0,
        "radio.scs_hz": 60_000.0,
        "algorithm.V_c": 0.7,
        "algorithm.V_nc": 0.3,
        "algorithm.V_phi": 0.0005,
        "algorithm.R_theta": 0.00001,
        "algorithm.B0_calibration_quantile": 0.75,
        "algorithm.beta_B": 1.0,
        "algorithm.sw_ucb_xi": math.sqrt(0.5),
        "algorithm.sw_ucb_window_s": 10.0,
        "geometry.phi_min_deg": 10.0,
        "delay.handover_s": 0.025,
        "delay.fixed_isl_per_hop_s": 0.005,
        "randomness.fixed_channel_seed": 0,
        "randomness.policy_seed": 0,
    }
    for path, expected_value in expected.items():
        actual = config.get(path)
        if actual is None or not math.isclose(
            float(actual), float(expected_value), rel_tol=0.0, abs_tol=1e-15
        ):
            raise ConfigurationError(
                f"phi-theta study requires {path}={expected_value!r}; got {actual!r}"
            )
    if config.get("services.payload_bytes") is not None:
        raise ConfigurationError("formal phi-theta study must not define payload_bytes")
    forbidden_message_size_fields = (
        "services.message_bytes",
        "services.message_size_bytes",
        "services.message_size_bits",
        "services.payload_bits",
    )
    for path in forbidden_message_size_fields:
        if config.get(path) is not None:
            raise ConfigurationError(f"formal phi-theta study must not define {path}")
    if list(config.get("search.phi_theta_deg", [])) != PHI_THETA_CANDIDATES_DEG:
        raise ConfigurationError(
            f"phi-theta candidates must be exactly {PHI_THETA_CANDIDATES_DEG}"
        )
    sr = config.get("channel.shadowed_rician")
    if not isinstance(sr, dict) or any(
        not math.isclose(float(sr.get(key, math.nan)), expected_value, rel_tol=0.0, abs_tol=1e-15)
        for key, expected_value in {"b0": 0.126, "Omega": 0.835, "m": 10.1}.items()
    ):
        raise ConfigurationError("phi-theta study requires SR=(0.126, 0.835, 10.1)")
    actions = [int(value) for value in config.get("algorithm.action_set")]
    if actions != ACTION_SETS["full"]:
        raise ConfigurationError("phi-theta study requires the full legal N=40 action set")
    if actions != list(config.get("derived.legal_actions")):
        raise ConfigurationError("phi-theta action set must contain every legal divisor of N=40")
    weights = [float(value) for value in config.get("services.weights")]
    if weights != [0.7, 0.2, 0.1]:
        raise ConfigurationError("phi-theta study requires weights=[0.7, 0.2, 0.1]")
    for name in (
        "randomness.pilot_channel_seeds",
        "randomness.validation_channel_seeds",
        "randomness.test_channel_seeds",
    ):
        values = config.get(name)
        if values is not None and [int(value) for value in values] != [0]:
            raise ConfigurationError(f"{name} may only contain the fixed seed [0]")
    layout_seeds = config.get("randomness.robustness_layout_seeds")
    if layout_seeds is not None and [int(value) for value in layout_seeds] != [42]:
        raise ConfigurationError(
            "randomness.robustness_layout_seeds is not allowed in the fixed-layout phi study"
        )


def validate_fixed_study_config(config: StudyConfig, *, allow_tau_sensitivity: bool = False) -> None:
    """Validate the phi=15 main configuration and fixed P1 experiment controls."""
    expected = {
        "devices.unique_count": 120,
        "constellation.actual_satellite_count": 8,
        "constellation.altitude_km": 800.0,
        "experiment.slot_dt_s": 0.01,
        "experiment.duration_s": 480.0,
        "experiment.warmup_report_s": 10.0,
        "services.bandwidth_hz": 50_000_000.0,
        "services.messages_per_active_satellite": 3,
        "services.subscribers_per_message": 40,
        "channel.carrier_hz": 20_000_000_000.0,
        "channel.reference_bandwidth_hz": 50_000_000.0,
        "channel.normalized_satellite_power": 1.0,
        "radio.scs_hz": 60_000.0,
        "algorithm.phi_theta_deg": FIXED_PHI_THETA_DEG,
        "algorithm.B_penalty": FIXED_B_PENALTY,
        "algorithm.V_c": 0.7,
        "algorithm.V_nc": 0.3,
        "algorithm.V_phi": 0.0005,
        "algorithm.R_theta": 0.00001,
        "algorithm.sw_ucb_xi": math.sqrt(0.5),
        "algorithm.regrouping_interval_s": 1.0,
        "geometry.phi_min_deg": 10.0,
        "delay.handover_s": 0.025,
        "delay.fixed_isl_per_hop_s": 0.005,
    }
    if not allow_tau_sensitivity:
        expected["algorithm.sw_ucb_window_s"] = 10.0
    for path, value in expected.items():
        actual = config.get(path)
        if actual is None or not math.isclose(float(actual), value, rel_tol=0.0, abs_tol=1e-15):
            raise ConfigurationError(f"fixed study requires {path}={value!r}; got {actual!r}")
    if config.get("algorithm.action_set_name") != "compact" or config.get("algorithm.action_set") != ACTION_SETS["compact"]:
        raise ConfigurationError("fixed study requires compact action set [1,2,4,5,8,10]")
    if int(config.get("derived.regrouping_interval_slots")) != 100:
        raise ConfigurationError("fixed study requires 100-slot regrouping epochs")
    if config.get("services.weights") != [0.7, 0.2, 0.1]:
        raise ConfigurationError("fixed study requires weights=[0.7, 0.2, 0.1]")
    if config.get("algorithm.reward_normalization") != "pilot_fixed_clip":
        raise ConfigurationError("fixed study requires pilot_fixed_clip reward normalization")
    if config.get("algorithm.power_mode") != POWER_MODE:
        raise ConfigurationError("fixed study requires csi_expanded_protected_pool")
    if config.get("algorithm.power_pool_rule_version") != POOL_RULE_VERSION:
        raise ConfigurationError("power-pool rule version mismatch")
    if config.get("ablation.phi_theta_deg") != P1_PHI_THETA_DEG or config.get("ablation.baselines_phi_theta_deg") != 15.0:
        raise ConfigurationError("P1 matrix requires MIRA at 13,15,20,25 and baselines only at 15")
    for path, expected_seed in (("devices.layout_seed",42),("devices.subscription_seed",1042),("randomness.fixed_channel_seed",0),("randomness.policy_seed",0)):
        if config.get(path) != expected_seed:
            raise ConfigurationError(f"P1 requires {path}={expected_seed}")
    if config.get("algorithm.infeasible_power_handling") != "transmission_failure":
        raise ConfigurationError("fixed study treats proven power infeasibility as a zero-power transmission failure")
    if config.get("channel.shadowed_rician") != {"b0": 0.126, "Omega": 0.835, "m": 10.1}:
        raise ConfigurationError("fixed study requires SR=(0.126, 0.835, 10.1)")
    if config.get("delay.handover_mode") != "paper_additive":
        raise ConfigurationError("fixed study requires paper_additive handover")
    if any(config.get(f"services.{name}") is not None for name in (
        "payload_bytes", "payload_bits", "message_bytes", "message_size_bytes", "message_size_bits"
    )):
        raise ConfigurationError("fixed study uses no payload gate")
    if config.get("search.phi_theta_deg") or config.get("search.beta_B"):
        raise ConfigurationError("phi/B sweeps are disabled in the fixed study")
    tau_values = config.get("formal_results.regrouping_overhead_tau_s", [10.0])
    if not isinstance(tau_values, list) or not tau_values or any(
        not math.isfinite(float(tau)) or float(tau) < float(config.get("algorithm.regrouping_interval_s"))
        or abs(float(tau) / float(config.get("algorithm.regrouping_interval_s")) - round(float(tau) / float(config.get("algorithm.regrouping_interval_s")))) > 1e-9
        for tau in tau_values
    ):
        raise ConfigurationError("regrouping_overhead_tau_s must contain positive integer multiples of the regrouping interval")
    if int(config.get("formal_results.fixed_action", 4)) != 4:
        raise ConfigurationError("formal Fixed grouping must use a=4")


def validate_p1_ablation_config(config: StudyConfig, *, allow_tau_sensitivity: bool = False) -> None:
    if float(config.get("algorithm.phi_theta_deg")) not in P1_PHI_THETA_DEG:
        raise ConfigurationError("P1 angles must be exactly 13,15,20,25 degrees")
    validate_fixed_study_config(config.with_overrides({"algorithm.phi_theta_deg": FIXED_PHI_THETA_DEG}),
                                allow_tau_sensitivity=allow_tau_sensitivity)


def load_config(path: str | Path) -> StudyConfig:
    return resolve_config(_load_mapping(Path(path)))


def save_config(config: StudyConfig, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
