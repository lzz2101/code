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


def expected_cross_link_interference(
    config,
    serving_satellite: int,
    member_indices: np.ndarray,
    path_gains: np.ndarray,
    visible: np.ndarray,
    transmitted_power_by_satellite: np.ndarray,
) -> float:
    sr = config.get("channel.shadowed_rician")
    mean_power = 2.0 * float(sr["b0"]) + float(sr["Omega"])
    values = np.zeros(len(member_indices), dtype=float)
    for satellite in range(path_gains.shape[0]):
        if satellite == serving_satellite:
            continue
        power = float(transmitted_power_by_satellite[satellite])
        if power <= 0.0:
            continue
        reached = visible[satellite, member_indices]
        values += reached * path_gains[satellite, member_indices] * mean_power * power
    return float(np.max(values)) if len(values) else 0.0
