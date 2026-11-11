from __future__ import annotations

from typing import Any


SOURCE_OVERRIDES = {
    "constellation.actual_satellite_count": ("paper", "Synthetic scenario: L=8"),
    "constellation.altitude_km": ("paper", "Simulation geometry: H=800 km"),
    "geometry.phi_min_deg": ("paper", "Simulation geometry note"),
    "channel.carrier_hz": ("paper", "Channel model"),
    "channel.shadowed_rician": ("paper", "Shadowed-Rician parameters"),
    "channel.normalized_satellite_power": ("paper", "Normalized per-satellite budget"),
    "services.messages_per_active_satellite": ("paper", "VII-B baseline"),
    "services.subscribers_per_message": ("paper", "VII-B baseline"),
    "services.weights": ("paper", "Message weight option"),
    "algorithm.V_phi": ("paper", "Equation (36)"),
    "algorithm.sw_ucb_xi": ("paper", "Algorithm 1"),
    "experiment.duration_s": ("user", "Synthetic observation window"),
    "geometry.region_half_width_km": ("paper/user", "1000 km projected square"),
    "devices.layout_seed": ("prior_proposal/user", "Fixed layout seed"),
    "geometry.center_raw_ecef_km": ("reference/user", "Taipei attachment data point"),
    "radio.scs_hz": ("user", "Phi-theta study specification; numerology metadata"),
    "services.bandwidth_hz": ("user", "Phi-theta study specification: 50 MHz"),
    "channel.reference_bandwidth_hz": (
        "user",
        "Phi-theta study specification: reference SNR at 50 MHz",
    ),
    "algorithm.V_c": ("user", "Phi-theta study specification: fixed at 0.7"),
    "algorithm.V_nc": ("user", "Phi-theta study specification: fixed at 0.3"),
    "algorithm.R_theta": (
        "user",
        "Phi-theta study specification using reference-code threshold 1e-5",
    ),
    "algorithm.phi_theta_deg": ("user", "P1 main threshold 15 degrees; MIRA ablation 13/15/20/25 degrees"),
    "algorithm.power_mode": ("user", "Scheduled geographic edge plus raw-CSI promotion; strict 70/30 pools and global SIC"),
    "algorithm.power_pool_rule_version": ("user", "Strict weaker-than-strongest-scheduled-geographic-edge rule"),
    "ablation": ("user", "Exactly four MIRA angles and Random/Fixed only at phi=15"),
    "algorithm.infeasible_power_handling": ("user", "Proven power infeasibility causes a zero-power failed update"),
    "algorithm.B0_calibration_quantile": (
        "new_design",
        "Empirical per-angle q75 calibration convention",
    ),
    "randomness.fixed_channel_seed": ("user", "Forced fixed realization: seed 0"),
    "geometry.earth_radius_km": ("prior_proposal", "Explicit spherical Earth model"),
    "constellation.inclination_deg": ("prior_proposal", "Synthetic orbit design"),
    "experiment.slot_dt_s": ("user", "Phi-theta study decision slot: 10 ms"),
    "association": ("new_design", "Single-serving coverage-first association policy"),
    "devices.unique_count": ("user", "120 stationary active subscribers"),
    "channel.external_interference": ("user", "Visible non-serving expected cross-link powers; sum per UE then group maximum"),
    "channel.interferer_power_mode": ("user", "Always-on cochannel full load at normalized power 1.0"),
    "channel.reference_snr_db": ("new_design", "Normalized link calibration"),
    "delay.isl_max_link_distance_km": ("new_design", "Synthetic ISL graph rule"),
    "algorithm.sw_ucb_window_s": ("new_design", "Ten-second sliding window over ten regrouping epochs"),
    "algorithm.reward_normalization": ("new_design", "Pilot-frozen learner scaling"),
    "derived": ("derived", "Resolved from explicit inputs"),
}


def _walk(prefix: str, value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            dotted = f"{prefix}.{key}" if prefix else key
            yield from _walk(dotted, child)
    else:
        yield prefix, value


def build_source_registry(config) -> dict:
    registry = {}
    for path, value in _walk("", config.to_dict()):
        match = path
        while match and match not in SOURCE_OVERRIDES:
            match = match.rsplit(".", 1)[0] if "." in match else ""
        source, note = SOURCE_OVERRIDES.get(
            match, ("new_design", "Explicit implementation setting")
        )
        registry[path] = {
            "value": value,
            "source": source,
            "reference": note,
            "is_new_assumption": source in {"new_design", "prior_proposal", "prior_proposal/user"},
        }
    return {
        "allowed_source_labels": [
            "paper",
            "user",
            "reference",
            "prior_proposal",
            "new_design",
            "derived",
        ],
        "parameters": registry,
    }
