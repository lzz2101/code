"""Streaming 960-second physical-link diagnostics, independent of MIRA policies.

Run from the package's parent directory:
    python -m mira_leo_study.csi_elevation_no_interference
    python -m mira_leo_study.csi_elevation_with_interference

Only these tools override duration. The original base config and all models
remain unchanged. Full-population statistics use disk-backed exact quantiles;
the deterministic plotting sample never determines the statistical population.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import tempfile

import numpy as np

from .channel import path_gain, serving_channel_amplitude
from .config import StudyConfig, load_config
from .geometry import link_geometry
from .scenario import StudyScenario

BASE_CONFIG = Path(__file__).with_name("configs") / "base.yaml"
DEFAULT_OUTPUTS = Path("outputs/csi_elevation_16min")
DURATION_S = 960.0
LINK_SHAPE = (8, 120)
BIN_COUNT = 80
BUFFER_SLOTS = 1000
READ_BLOCK_RECORDS = 1_000_000
PLOT_SEED = 0
MAX_PLOT_POINTS = 100_000
GAIN_DTYPE = np.dtype("<f8")
BIN_DTYPE = np.dtype("u1")


def build_analysis_config() -> tuple[StudyConfig, StudyConfig]:
    """Use the original resolver; bypass only the formal 480-second validator."""
    base = load_config(BASE_CONFIG)
    config = base.with_overrides({"experiment.duration_s": DURATION_S})
    if config.slot_count != 96000:
        raise ValueError("the diagnostic requires 96,000 slots of 0.01 seconds")
    before, after = base.to_dict(), config.to_dict()
    before.pop("derived", None)
    after.pop("derived", None)
    before["experiment"]["duration_s"] = DURATION_S
    if before != after:
        raise ValueError("only experiment.duration_s may change")
    return base, config


def check_source_invariants(config: StudyConfig, scenario: StudyScenario) -> None:
    """Fail clearly if the installed source differs from the requested scenario."""
    expected = {
        "experiment.duration_s": 960.0, "experiment.slot_dt_s": 0.01,
        "experiment.epoch_utc": "2023-04-19T11:56:35Z",
        "constellation.actual_satellite_count": 8, "constellation.altitude_km": 800.0,
        "constellation.inclination_deg": 50.0, "constellation.eccentricity": 0.0,
        "constellation.center_overflight_s": 240.0,
        "constellation.plane_raan_difference_deg": 10.0,
        "constellation.plane1_phase_offsets_deg": [-40, -20, 0, 20, 40],
        "constellation.plane2_phase_offsets_deg": [-10, 10, 30],
        "geometry.earth_model": "sphere", "geometry.earth_radius_km": 6371.0,
        "geometry.projection": "spherical_aeqd", "geometry.region_half_width_km": 500.0,
        "geometry.center_raw_ecef_km": [-3023.722874, 4928.116736, 2683.88077],
        "geometry.phi_min_deg": 10.0,
        "devices.unique_count": 120, "devices.stationary": True,
        "devices.distribution": "iid_uniform_in_projected_square",
        "devices.layout_seed": 42, "devices.subscription_seed": 1042,
        "randomness.fixed_channel_seed": 0, "randomness.policy_seed": 0,
        "channel.carrier_hz": 20e9, "radio.scs_hz": 60e3,
        "services.bandwidth_hz": 50e6, "channel.reference_bandwidth_hz": 50e6,
        "channel.reference_snr_db": 10.0, "channel.noise_model": "normalized_reference_snr",
        "channel.shadowed_rician": {"b0": 0.126, "Omega": 0.835, "m": 10.1},
        "channel.sr_mean_normalization": False,
        "channel.fading_model": "iid_per_link_per_slot",
        "channel.normalized_satellite_power": 1.0,
        "channel.interferer_power_mode": "fixed_full_load",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"source invariant {key} must be {value!r}; got {config.get(key)!r}")
    if scenario.devices.ecef_km.shape != (120, 3) or len(scenario.constellation.elements) != 8:
        raise ValueError("expected eight original satellites and 120 original UE coordinates")
    if not np.all(np.isfinite(scenario.devices.ecef_km)):
        raise ValueError("UE coordinates must be finite")
    noise = float(config.get("derived.noise_power"))
    if not math.isfinite(noise) or noise <= 0:
        raise ValueError("derived.noise_power must be finite and positive")


def elevation_bin_indices(angles_deg: np.ndarray) -> np.ndarray:
    """Map visible angles to [10,11), ..., [89,90]; do not round bin centers."""
    angles = np.asarray(angles_deg, dtype=float)
    if angles.ndim != 1 or not np.all(np.isfinite(angles)):
        raise ValueError("angles must be a finite one-dimensional array")
    if np.any(angles < 10.0) or np.any(angles > 90.0):
        raise ValueError("only visible elevations in [10,90] belong to the population")
    return np.minimum(np.floor(angles).astype(np.int64) - 10, BIN_COUNT - 1).astype(BIN_DTYPE)


def compute_external_interference(config, path_gains, visible) -> np.ndarray:
    """Algebraic equivalent of the original expected per-UE cross-link helper."""
    gains = np.asarray(path_gains, dtype=float)
    visible = np.asarray(visible)
    if gains.shape != LINK_SHAPE or visible.shape != LINK_SHAPE or visible.dtype != np.bool_:
        raise ValueError("expected gains and boolean visibility of shape (8,120)")
    if not np.all(np.isfinite(gains)) or np.any(gains < 0):
        raise ValueError("path gains must be finite and nonnegative")
    mean_sr = float(config.get("derived.shadowed_rician_mean_power"))
    power = float(config.get("channel.normalized_satellite_power"))
    if not math.isfinite(mean_sr) or mean_sr <= 0 or not math.isfinite(power) or power < 0:
        raise ValueError("expected fading and external power must be valid")
    q = np.where(visible, gains * mean_sr * power, 0.0)
    return np.maximum(q.sum(axis=0, keepdims=True) - q, 0.0)


def interference_adjusted_gain(raw_gain, expected_raw_gain, interference, noise_power):
    """Noise-referenced equivalent gain; never mutate the physical raw channel."""
    noise = float(noise_power)
    interference = np.asarray(interference, dtype=float)
    if not math.isfinite(noise) or noise <= 0:
        raise ValueError("noise power must be finite and positive")
    if not np.all(np.isfinite(interference)) or np.any(interference < 0):
        raise ValueError("interference must be finite and nonnegative")
    attenuation = noise / (noise + interference)
    if np.any(attenuation <= 0) or np.any(attenuation > 1):
        raise ValueError("interference attenuation must be in (0,1]")
    return raw_gain * attenuation, expected_raw_gain * attenuation


@dataclass(frozen=True)
class SlotMetric:
    raw_gain: np.ndarray
    elevation_deg: np.ndarray
    visible: np.ndarray
    angles_deg: np.ndarray
    angle_bins: np.ndarray
    gain_db: np.ndarray
    expected_gain_linear: np.ndarray
    observation_ids: np.ndarray
    interference_power: np.ndarray | None


def compute_slot_metric(config, scenario, slot: int, with_interference: bool) -> SlotMetric:
    """One full original fading call, followed by visibility-only filtering."""
    _, satellite_ecef = scenario.constellation.state(slot * float(config.get("experiment.slot_dt_s")))
    distances, elevations = link_geometry(satellite_ecef, scenario.devices.ecef_km)
    if distances.shape != LINK_SHAPE or elevations.shape != LINK_SHAPE:
        raise ValueError("original geometry must return shape (8,120)")
    if not np.all(np.isfinite(elevations)):
        raise ValueError("geometry returned non-finite elevations")
    visible = elevations >= float(config.get("geometry.phi_min_deg"))
    amplitude = serving_channel_amplitude(
        config, int(config.get("randomness.fixed_channel_seed")), slot, distances)
    if amplitude.shape != LINK_SHAPE:
        raise ValueError("fading must use the entire original (8,120) matrix")
    raw_gain = amplitude ** 2
    if not np.all(np.isfinite(raw_gain)) or np.any(raw_gain < 0):
        raise ValueError("raw channel gains must be finite and nonnegative")
    zeta = path_gain(distances, float(config.get("channel.carrier_hz")),
                     float(config.get("geometry.light_speed_km_s")))
    expected = zeta * float(config.get("derived.shadowed_rician_mean_power"))
    metric_gain = raw_gain
    interference = None
    if with_interference:
        interference = compute_external_interference(config, zeta, visible)
        metric_gain, expected = interference_adjusted_gain(
            raw_gain, expected, interference, config.get("derived.noise_power"))
    angles = elevations[visible]
    gain_db = 10.0 * np.log10(np.maximum(metric_gain[visible], np.finfo(float).tiny))
    expected_visible = expected[visible]
    if not np.all(np.isfinite(gain_db)) or not np.all(np.isfinite(expected_visible)):
        raise ValueError("visible observations must be finite")
    if np.any(expected_visible <= 0):
        raise ValueError("expected visible gains must be positive")
    ids = slot * (8 * 120) + np.flatnonzero(visible)
    return SlotMetric(raw_gain, elevations, visible, angles, elevation_bin_indices(angles),
                      gain_db, expected_visible, ids.astype(np.uint64),
                      None if interference is None else interference[visible])


class PlotReservoir:
    """Uniform bottom-K independent random keys, updated in vectorized chunks."""

    def __init__(self, maximum: int):
        self.maximum = maximum
        self.rng = np.random.default_rng(PLOT_SEED)
        self.keys = np.empty(0)
        self.angles = np.empty(0)
        self.gains_db = np.empty(0)
        self.ids = np.empty(0, dtype=np.uint64)

    def update(self, angles, gains_db, observation_ids):
        size = len(gains_db)
        if not size:
            return
        keys = np.concatenate((self.keys, self.rng.random(size)))
        angles = np.concatenate((self.angles, angles))
        gains_db = np.concatenate((self.gains_db, gains_db))
        ids = np.concatenate((self.ids, observation_ids))
        if len(keys) > self.maximum:
            keep = np.argpartition(keys, self.maximum - 1)[:self.maximum]
            boundary = keys[keep].max()
            tied = np.flatnonzero(keys == boundary)
            if len(tied) > 1:
                lower = np.flatnonzero(keys < boundary)
                tie_order = np.argsort(ids[tied], kind="stable")
                keep = np.concatenate((lower, tied[tie_order[:self.maximum - len(lower)]]))
        else:
            keep = np.arange(len(keys))
        self.keys, self.angles = keys[keep], angles[keep]
        self.gains_db, self.ids = gains_db[keep], ids[keep]

    def ordered(self):
        order = np.argsort(self.ids, kind="stable")
        return self.angles[order], self.gains_db[order], self.ids[order]


def compute_exact_quantile_from_memmap(gain_path: Path, sample_count: int,
                                       quantile: float = 0.20) -> float:
    """NumPy linear quantile via two order statistics on an on-disk COPY.

    ndarray.partition is in place. Never call np.partition (which returns a
    whole-population copy), nor partition the angle-aligned original file.
    """
    gain_path = Path(gain_path)
    if not math.isfinite(quantile) or not 0 <= quantile <= 1:
        raise ValueError("quantile must be finite and between zero and one")
    if sample_count <= 0 or gain_path.stat().st_size != sample_count * GAIN_DTYPE.itemsize:
        raise ValueError("gain file size must match a nonempty float64 population")
    copy_path = gain_path.with_name("gain_for_quantile.bin")
    if copy_path.exists():
        raise FileExistsError(copy_path)
    mapping = None
    try:
        shutil.copyfile(gain_path, copy_path)
        mapping = np.memmap(copy_path, dtype=GAIN_DTYPE, mode="r+", shape=(sample_count,))
        position = quantile * (sample_count - 1)
        lo, hi = math.floor(position), math.ceil(position)
        np.ndarray.partition(mapping, (lo, hi))
        endpoints = np.array([float(mapping[lo]), float(mapping[hi])], dtype=np.float64)
        threshold = float(np.quantile(endpoints, position - lo, method="linear"))
        if not math.isfinite(threshold):
            raise ValueError("non-finite global quantile")
        return threshold
    finally:
        if mapping is not None:
            try:
                mapping.flush()
            finally:
                # Explicit release matters on Windows before unlinking this file.
                mapping._mmap.close()
                del mapping
        copy_path.unlink(missing_ok=True)


def accumulate_bottom_by_angle(gain_path, bin_path, sample_count, threshold_db,
                               block_records=READ_BLOCK_RECORDS):
    """Read paired original records in bounded blocks; quantile never reorders them."""
    if Path(gain_path).stat().st_size != sample_count * 8 or Path(bin_path).stat().st_size != sample_count:
        raise ValueError("gain/bin files are not aligned")
    totals = np.zeros(BIN_COUNT, dtype=np.int64)
    bottom = np.zeros(BIN_COUNT, dtype=np.int64)
    ties = 0
    with Path(gain_path).open("rb") as gain_file, Path(bin_path).open("rb") as bin_file:
        for start in range(0, sample_count, block_records):
            size = min(block_records, sample_count - start)
            values = np.frombuffer(gain_file.read(size * 8), dtype=GAIN_DTYPE)
            bins = np.frombuffer(bin_file.read(size), dtype=BIN_DTYPE)
            if len(values) != size or len(bins) != size or not np.all(np.isfinite(values)):
                raise ValueError("incomplete/non-finite raw observations")
            if np.any(bins >= BIN_COUNT):
                raise ValueError("raw angle bin outside 0..79")
            totals += np.bincount(bins, minlength=BIN_COUNT)
            bottom += np.bincount(bins[values <= threshold_db], minlength=BIN_COUNT)
            ties += int(np.count_nonzero(values == threshold_db))
    return totals, bottom, ties


def accumulate_bottom20_by_angle(gain_path, bin_path, sample_count, threshold_db,
                                  block_records=READ_BLOCK_RECORDS):
    """Compatibility name; the supplied global threshold determines the subset."""
    return accumulate_bottom_by_angle(gain_path, bin_path, sample_count, threshold_db, block_records)


def bin_statistics(total_count, bottom_count, expected_linear_sum):
    total = np.asarray(total_count, dtype=np.int64)
    bottom = np.asarray(bottom_count, dtype=np.int64)
    sums = np.asarray(expected_linear_sum, dtype=float)
    if any(x.shape != (BIN_COUNT,) for x in (total, bottom, sums)):
        raise ValueError("statistics require exactly 80 angle bins")
    if np.any(total < 0) or np.any(bottom < 0) or np.any(bottom > total):
        raise ValueError("inconsistent bin counts")
    fractions = np.divide(100.0 * bottom, total, out=np.full(BIN_COUNT, np.nan), where=total > 0)
    expected = np.divide(sums, total, out=np.full(BIN_COUNT, np.nan), where=total > 0)
    expected_db = np.full(BIN_COUNT, np.nan)
    populated = total > 0
    if np.any(expected[populated] <= 0) or not np.all(np.isfinite(expected[populated])):
        raise ValueError("populated expected means must be finite and positive")
    expected_db[populated] = 10.0 * np.log10(expected[populated])
    return fractions, expected, expected_db


def save_statistics_csv(path, totals, bottom, fractions, expected, expected_db, *, bottom_percent=20):
    fields = ["angle_start_deg", "angle_end_deg", "angle_center_deg", "total_count",
              f"bottom{bottom_percent}_count", f"bottom{bottom_percent}_fraction_pct",
              "mean_expected_gain_linear", "mean_expected_gain_db"]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for index in range(BIN_COUNT):
            values = [10 + index, 11 + index, 10.5 + index, int(totals[index]), int(bottom[index]),
                      fractions[index], expected[index], expected_db[index]]
            writer.writerow(["" if isinstance(v, (float, np.floating)) and not math.isfinite(v) else v for v in values])


def save_plots(directory, reservoir, expected_db, fractions, *, with_interference, slots, dt_s, is_debug,
               bottom_percent=20):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    angles, values, _ = reservoir.ordered()
    centers = np.arange(10.5, 90.0, 1.0)
    kind = "interference-adjusted gain" if with_interference else "CSI power gain"
    scope = f"{slots} slots, {slots * dt_s:g} s; DEBUG" if is_debug else "16 min"
    prefix = "MIRA-LEO"
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 12}):
        fig, ax = plt.subplots(figsize=(14, 7), facecolor="white")
        try:
            ax.scatter(angles, values, color="0.55", s=1.5, alpha=0.14, linewidths=0,
                       rasterized=True, label=("Interference-adjusted link gain (sampled satellite-UE links)"
                                              if with_interference else "Instantaneous CSI (sampled satellite-UE links)"))
            ax.plot(centers, expected_db, color="#e68613", linewidth=3,
                    label=("Expected equivalent gain (geometry + mean fading + interference)"
                           if with_interference else "Expected CSI power gain (geometry + mean fading)"))
            ax.set(xlim=(10, 90), xlabel="Satellite elevation angle (degrees)",
                   ylabel=("Interference-adjusted equivalent gain (dB)" if with_interference
                           else "CSI power gain (dB, 10 log10 |h|²)"),
                   title=f"{prefix}: elevation angle vs. {kind} (120 stationary UE, 8 LEO, {scope})")
            ax.grid(True, color="0.9", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.legend(loc="best", markerscale=4)
            fig.tight_layout()
            fig.savefig(Path(directory) / "01_elevation_vs_gain.png", dpi=160,
                        facecolor="white", metadata={"Software": "mira_leo_study"})
        finally:
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(14, 7), facecolor="white")
        try:
            ax.plot(centers, fractions, color="#e68613", linewidth=2, marker="o", markersize=3)
            ax.set(xlim=(10, 90), ylim=(0, 100), xlabel="Satellite elevation angle (degrees; 1-degree bins)",
                   ylabel=f"Fraction in this version's global bottom {bottom_percent}% (%)",
                   title=f"{prefix}: global bottom-{bottom_percent}% {kind} by elevation ({scope})")
            ax.grid(True, color="0.9", linewidth=0.7)
            fig.tight_layout()
            fig.savefig(Path(directory) / f"02_bottom{bottom_percent}_fraction_by_elevation.png", dpi=160,
                        facecolor="white", metadata={"Software": "mira_leo_study"})
        finally:
            plt.close(fig)


def run_csi_elevation_study(*, with_interference: bool, outputs: Path = DEFAULT_OUTPUTS,
                            smoke_slots: int | None = None, keep_raw: bool = False,
                            plot_sample_max: int = MAX_PLOT_POINTS,
                            bottom_percent: int = 20) -> dict:
    """Three passes: original per-slot channel, exact disk quantile, aligned bins."""
    if isinstance(bottom_percent, bool) or not isinstance(bottom_percent, int) or bottom_percent not in (20, 30):
        raise ValueError("bottom_percent must be 20 or 30")
    quantile = bottom_percent / 100.0
    base, config = build_analysis_config()
    if isinstance(plot_sample_max, bool) or not isinstance(plot_sample_max, int) or not 1 <= plot_sample_max <= MAX_PLOT_POINTS:
        raise ValueError("plot_sample_max must be an integer between 1 and 100000")
    if smoke_slots is not None and (isinstance(smoke_slots, bool) or not isinstance(smoke_slots, int)
                                   or not 1 <= smoke_slots < config.slot_count):
        raise ValueError("smoke_slots must be between 1 and 95999; omit it for the full study")
    slots = config.slot_count if smoke_slots is None else smoke_slots
    is_debug = smoke_slots is not None
    mode = "with_interference" if with_interference else "no_interference"
    if bottom_percent != 20:
        mode += f"_bottom{bottom_percent}"
    output = Path(outputs) / (Path("debug") / f"{mode}_{slots}slots" if is_debug else Path(mode))
    scenario = StudyScenario(config)
    check_source_invariants(config, scenario)
    original_coordinates = scenario.devices.ecef_km.copy()
    output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="_raw_", dir=output))
    gain_path, bin_path = work / "gain_db.bin", work / "angle_bin.bin"
    totals = np.zeros(BIN_COUNT, dtype=np.int64)
    expected_sums = np.zeros(BIN_COUNT)
    reservoir = PlotReservoir(plot_sample_max)
    raw_hash, elevation_hash = hashlib.sha256(), hashlib.sha256()
    count = 0
    interference_count = 0
    interference_sum = 0.0
    interference_min, interference_max = math.inf, -math.inf
    buffer = []

    def flush_buffers(gain_file, bin_file):
        if not buffer:
            return
        values = np.concatenate([x[0] for x in buffer]).astype(GAIN_DTYPE, copy=False)
        bins = np.concatenate([x[1] for x in buffer])
        angles = np.concatenate([x[2] for x in buffer])
        ids = np.concatenate([x[3] for x in buffer])
        values.tofile(gain_file)
        bins.tofile(bin_file)
        reservoir.update(angles, values, ids)
        buffer.clear()

    try:
        with gain_path.open("wb") as gain_file, bin_path.open("wb") as bin_file:
            for slot in range(slots):
                item = compute_slot_metric(config, scenario, slot, with_interference)
                raw_hash.update(np.ascontiguousarray(item.raw_gain, dtype=GAIN_DTYPE).tobytes())
                elevation_hash.update(np.ascontiguousarray(item.elevation_deg, dtype=GAIN_DTYPE).tobytes())
                size = len(item.gain_db)
                count += size
                totals += np.bincount(item.angle_bins, minlength=BIN_COUNT)
                expected_sums += np.bincount(item.angle_bins, weights=item.expected_gain_linear, minlength=BIN_COUNT)
                if with_interference and size:
                    powers = item.interference_power
                    interference_count += int(np.count_nonzero(powers > 0))
                    interference_sum += float(np.sum(powers))
                    interference_min = min(interference_min, float(np.min(powers)))
                    interference_max = max(interference_max, float(np.max(powers)))
                # Store only fields needed by this bounded 1000-slot buffer.
                buffer.append((item.gain_db, item.angle_bins, item.angles_deg, item.observation_ids))
                if (slot + 1) % BUFFER_SLOTS == 0:
                    flush_buffers(gain_file, bin_file)
                    print(f"[{mode}] {slot + 1}/{slots} slots; {count:,} visible observations", flush=True)
            flush_buffers(gain_file, bin_file)
        if not count:
            raise ValueError("no visible links in this observation window; no quantile or fabricated zero gains")
        if not np.array_equal(original_coordinates, scenario.devices.ecef_km):
            raise RuntimeError("UE coordinates changed during the diagnostic")
        print(f"[{mode}] {slots}/{slots} slots; {count:,} observations; exact disk-backed q{bottom_percent}", flush=True)
        threshold = compute_exact_quantile_from_memmap(gain_path, count, quantile)
        reread_totals, bottom, ties = accumulate_bottom_by_angle(gain_path, bin_path, count, threshold)
        if not np.array_equal(totals, reread_totals) or int(totals.sum()) != count:
            raise RuntimeError("streaming and paired-file counts do not reconcile")
        fraction = int(bottom.sum()) / count
        tolerance = 1.0 / count + 1e-12
        if abs(fraction - quantile) > tolerance:
            raise RuntimeError(f"global weak fraction {fraction:.12g} differs from {quantile:g}; threshold ties={ties}")
        fractions, expected, expected_db = bin_statistics(totals, bottom, expected_sums)
        save_statistics_csv(work / "elevation_bin_statistics.csv", totals, bottom, fractions, expected, expected_db,
                            bottom_percent=bottom_percent)
        dt_s = float(config.get("experiment.slot_dt_s"))
        save_plots(work, reservoir, expected_db, fractions, with_interference=with_interference,
                   slots=slots, dt_s=dt_s, is_debug=is_debug, bottom_percent=bottom_percent)
        import matplotlib
        _, _, selected_ids = reservoir.ordered()
        summary = {
            "status": "COMPLETED_DEBUG" if is_debug else "COMPLETED",
            "study": "MIRA-LEO elevation-CSI 16-minute physical-link diagnostic",
            "mode": mode, "sample_scope": "all_visible_satellite_UE_candidate_links",
            "not_a_single_serving_scheduling_performance_comparison": True,
            "is_debug": is_debug, "is_full_horizon": slots == config.slot_count,
            "configured_duration_s": DURATION_S, "time_s": slots * dt_s,
            "last_processed_time_s": (slots - 1) * dt_s, "slot_dt_s": dt_s,
            "slots_completed": slots, "satellite_count": 8, "ue_count": 120,
            "layout_seed": int(config.get("devices.layout_seed")),
            "subscription_seed": int(config.get("devices.subscription_seed")),
            "channel_seed": int(config.get("randomness.fixed_channel_seed")),
            "policy_used": False, "plot_sample_seed": PLOT_SEED,
            "phi_min_deg": float(config.get("geometry.phi_min_deg")),
            "phi_theta_used_for_analysis": False,
            "sr_mean_power": float(config.get("derived.shadowed_rician_mean_power")),
            "noise_power": float(config.get("derived.noise_power")),
            "number_of_valid_link_observations": count,
            "number_of_plotted_scatter_points": len(selected_ids),
            "bottom_percent": bottom_percent,
            f"global_bottom{bottom_percent}_threshold_db": threshold,
            f"global_bottom{bottom_percent}_threshold_linear": float(10.0 ** (threshold / 10.0)),
            f"global_bottom{bottom_percent}_fraction": fraction,
            f"global_bottom{bottom_percent}_count": int(bottom.sum()),
            "threshold_tie_count": ties, "quantile_method": "linear",
            "statistics_use_all_visible_observations": True,
            "angle_bin_edges_deg": list(range(10, 91)), "angle_bin_total_counts": totals.tolist(),
            "bin_interval_convention": "[start,end), with final bin [89,90]",
            "layout_hash": scenario.identity["layout_hash"],
            "subscription_hash": scenario.identity["subscription_hash"],
            "ue_positions_unchanged": True,
            "base_config_id": base.config_id, "analysis_config_id": config.config_id,
            "raw_channel_trace_sha256": raw_hash.hexdigest(),
            "elevation_trace_sha256": elevation_hash.hexdigest(),
            "trace_array_format": "full (8,120), slot-major/satellite-major/UE-minor, little-endian float64",
            "plot_sample_observation_ids_sha256": hashlib.sha256(selected_ids.astype("<u8").tobytes()).hexdigest(),
            "plot_sample_id_encoding": "slot*960 + satellite_index*120 + ue_index; ordered ascending",
            "plot_sample_method": "uniform independent random keys; bottom-K; seed 0; chunk updates",
            "scatter_y_axis_clipped": False,
            "interference_model": "fixed_full_load_expected_cross_link_power" if with_interference else "none",
            "metric_semantics": "interference_adjusted_equivalent_power_gain" if with_interference else "raw_channel_power_gain",
            "interferer_power_normalized": float(config.get("channel.normalized_satellite_power")) if with_interference else None,
            "interference_positive_count": interference_count if with_interference else None,
            "interference_positive_fraction": interference_count / count if with_interference else None,
            "interference_min": interference_min if with_interference else None,
            "interference_max": interference_max if with_interference else None,
            "interference_mean": interference_sum / count if with_interference else None,
            "streaming_buffer_slots": BUFFER_SLOTS,
            "raw_pair_bytes": count * 9,
            "quantile_working_copy_bytes": count * 8,
            "peak_temporary_binary_bytes": count * 17,
            "raw_files_kept": keep_raw,
            "raw_files": {"gain_db": {"path": str(gain_path.resolve()), "dtype": "<f8"},
                          "angle_bin": {"path": str(bin_path.resolve()), "dtype": "uint8"},
                          "record_count": count, "record_order": "visible entries in slot/satellite/UE order"} if keep_raw else None,
            "software_versions": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
            "output_directory": str(output.resolve()),
        }
        (work / "run_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for name in ("01_elevation_vs_gain.png", f"02_bottom{bottom_percent}_fraction_by_elevation.png",
                     "elevation_bin_statistics.csv", "run_summary.json"):
            (work / name).replace(output / name)
        if not keep_raw:
            gain_path.unlink()
            bin_path.unlink()
            work.rmdir()
        print(f"[{mode}] complete: {output.resolve()}", flush=True)
        return summary
    except BaseException:
        print(f"[{mode}] diagnostic incomplete; intermediate files retained at {work.resolve()}", file=sys.stderr)
        raise


def main(argv=None, *, with_interference: bool = False, bottom_percent: int = 20) -> int:
    parser = argparse.ArgumentParser(description=(
        f"16-minute all-visible satellite-UE physical-link diagnostic, global bottom {bottom_percent}%; "
        "no association, multicast scheduling or AoI simulation"))
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS, help="parent directory for both independent modes")
    parser.add_argument("--smoke-slots", type=int, default=None, help="explicit short DEBUG run, stored separately; omit for 96000 slots")
    parser.add_argument("--keep-raw", action="store_true", help="keep paired gain/bin binary files after success")
    parser.add_argument("--plot-sample-max", type=int, default=MAX_PLOT_POINTS, help="1..100000 points; never changes population statistics")
    args = parser.parse_args(argv)
    try:
        summary = run_csi_elevation_study(with_interference=with_interference, outputs=args.outputs,
                                          smoke_slots=args.smoke_slots, keep_raw=args.keep_raw,
                                          plot_sample_max=args.plot_sample_max, bottom_percent=bottom_percent)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
