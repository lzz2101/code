from __future__ import annotations

import math

import numpy as np


def shadowed_rician_samples(
    seed: int,
    slot: int,
    shape: tuple[int, ...],
    b0: float,
    omega: float,
    m_shape: float,
) -> np.ndarray:
    """Generate one policy-independent indexed fading block."""
    if b0 < 0.0 or omega <= 0.0 or m_shape <= 0.0:
        raise ValueError("invalid Shadowed-Rician parameters")
    sequence = np.random.SeedSequence([int(seed), int(slot), 0x4D495241])
    rng = np.random.default_rng(sequence)
    real = rng.normal(0.0, math.sqrt(b0), size=shape)
    imag = rng.normal(0.0, math.sqrt(b0), size=shape)
    los_power = rng.gamma(shape=m_shape, scale=omega / m_shape, size=shape)
    phase = rng.uniform(0.0, 2.0 * math.pi, size=shape)
    return real + 1j * imag + np.sqrt(los_power) * np.exp(1j * phase)


def path_gain(distance_km: np.ndarray, carrier_hz: float, light_speed_km_s: float) -> np.ndarray:
    distance = np.asarray(distance_km, dtype=float)
    if np.any(distance <= 0.0):
        raise ValueError("link distances must be positive")
    return (float(light_speed_km_s) / (4.0 * math.pi * distance * float(carrier_hz))) ** 2


def serving_channel_amplitude(config, seed: int, slot: int, distance_km: np.ndarray) -> np.ndarray:
    sr = config.get("channel.shadowed_rician")
    fading = shadowed_rician_samples(
        seed,
        slot,
        tuple(distance_km.shape),
        float(sr["b0"]),
        float(sr["Omega"]),
        float(sr["m"]),
    )
    zeta = path_gain(
        distance_km,
        float(config.get("channel.carrier_hz")),
        float(config.get("geometry.light_speed_km_s")),
    )
    return np.abs(fading) * np.sqrt(zeta)


def external_interferer_powers(config, satellite_count: int, serving_satellite: int) -> np.ndarray:
    """Always-on full-load cochannel powers, independent of the serving schedule."""
    if not 0 <= serving_satellite < satellite_count:
        raise ValueError("serving satellite is out of range")
    power = float(config.get("channel.normalized_satellite_power"))
    if not math.isfinite(power) or power < 0.0:
        raise ValueError("interferer power must be finite and nonnegative")
    powers = np.full(satellite_count, power, dtype=float)
    powers[serving_satellite] = 0.0
    return powers


def expected_cross_link_interference_by_ue(
    config,
    serving_satellite: int,
    member_indices: np.ndarray,
    path_gains: np.ndarray,
    visible: np.ndarray,
    interferer_power_by_satellite: np.ndarray,
) -> np.ndarray:
    """Sum visible non-serving expected cross-link powers separately for each UE."""
    path_gains = np.asarray(path_gains, dtype=float)
    visible = np.asarray(visible)
    powers = np.asarray(interferer_power_by_satellite, dtype=float)
    members = np.asarray(member_indices)
    if path_gains.ndim != 2 or visible.shape != path_gains.shape or visible.dtype != np.bool_:
        raise ValueError("gains and boolean visibility must have shape [satellite, device]")
    if powers.shape != (path_gains.shape[0],):
        raise ValueError("interferer powers must have one entry per satellite")
    if not 0 <= serving_satellite < path_gains.shape[0]:
        raise ValueError("serving satellite is out of range")
    if members.ndim != 1 or (members.size and (not np.issubdtype(members.dtype, np.integer)
            or np.any(members < 0) or np.any(members >= path_gains.shape[1]))):
        raise ValueError("member indices must be a valid one-dimensional integer array")
    members = members.astype(np.intp)
    if (not np.all(np.isfinite(path_gains)) or np.any(path_gains < 0)
            or not np.all(np.isfinite(powers)) or np.any(powers < 0)):
        raise ValueError("gains and powers must be finite and nonnegative")
    sr = config.get("channel.shadowed_rician")
    mean_power = 2.0 * float(sr["b0"]) + float(sr["Omega"])
    values = np.zeros(len(members), dtype=float)
    for satellite in range(path_gains.shape[0]):
        if satellite == serving_satellite:
            continue
        power = float(powers[satellite])
        if power <= 0.0:
            continue
        reached = visible[satellite, members]
        values += reached * path_gains[satellite, members] * mean_power * power
    return values


def expected_cross_link_interference(
    config, serving_satellite: int, member_indices: np.ndarray,
    path_gains: np.ndarray, visible: np.ndarray, interferer_power_by_satellite: np.ndarray,
) -> float:
    """Worst UE after summing all visible neighbors; serving signal is excluded."""
    values = expected_cross_link_interference_by_ue(
        config, serving_satellite, member_indices, path_gains, visible, interferer_power_by_satellite)
    return float(np.max(values)) if len(values) else 0.0
