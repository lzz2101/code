"""16-minute serving-only raw CSI diagnostic with global bottom-20/30 curves.

Run from the package parent: python -m mira_leo_study.csi_elevation_serving_satellite
Association follows the existing coverage-first/hysteresis/TTT implementation.
No multicast scheduling, power allocation, interference adjustment or AoI runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np

from .association import AssociationTracker
from .csi_elevation_no_interference import (
    BIN_COUNT, BIN_DTYPE, BUFFER_SLOTS, DURATION_S, GAIN_DTYPE, LINK_SHAPE,
    MAX_PLOT_POINTS, PLOT_SEED, PlotReservoir, SlotMetric,
    accumulate_bottom_by_angle, bin_statistics, build_analysis_config,
    check_source_invariants, compute_exact_quantile_from_memmap, compute_slot_metric,
)
from .scenario import StudyScenario

DEFAULT_OUTPUTS = Path(__file__).with_name("outputs") / "csi_elevation_16min_serving_only"
FIGURE_NAMES = (
    "01_elevation_vs_gain.png",
    "02_bottom20_fraction_by_elevation.png",
    "03_bottom30_fraction_by_elevation.png",
)
ASSOCIATION_FIELDS = (
    "slot", "time_s", "serving_satellite_id", "handover", "initial_acquisition",
    "reason", "fallback_used", "outage_receivers", "visible_ue_count",
    "min_serving_elevation_deg", "visible_ue_below_15_count",
)


def select_serving_links(item: SlotMetric, satellite: int) -> SlotMetric:
    """Keep only visible links of the chosen satellite, retaining full trace arrays."""
    if isinstance(satellite, bool) or not isinstance(satellite, (int, np.integer)) or not 0 <= satellite < LINK_SHAPE[0]:
        raise ValueError("serving satellite index must be an integer in 0..7")
    if item.visible.shape != LINK_SHAPE or item.raw_gain.shape != LINK_SHAPE or item.elevation_deg.shape != LINK_SHAPE:
        raise ValueError("original channel/geometry must retain shape (8,120)")
    if item.interference_power is not None:
        raise ValueError("serving-only diagnostic requires raw CSI without interference adjustment")
    size = len(item.observation_ids)
    if any(len(values) != size for values in (item.angles_deg, item.angle_bins, item.gain_db, item.expected_gain_linear)):
        raise ValueError("CSI/elevation/observation IDs must stay aligned")
    # The old diagnostic encodes slot*960 + satellite*120 + UE, in row-major order.
    keep = (item.observation_ids % np.prod(LINK_SHAPE)) // LINK_SHAPE[1] == satellite
    visible = np.zeros_like(item.visible, dtype=bool)
    visible[satellite] = item.visible[satellite]
    if int(np.count_nonzero(keep)) != int(np.count_nonzero(visible)):
        raise ValueError("selected IDs do not match selected-satellite visibility")
    return SlotMetric(item.raw_gain, item.elevation_deg, visible,
                      item.angles_deg[keep], item.angle_bins[keep], item.gain_db[keep],
                      item.expected_gain_linear[keep], item.observation_ids[keep], None)


def summarize_population(gain_path, bin_path, count, totals, expected_sums):
    """Both exact global cutoffs use the same serving-only population on disk."""
    results = {}
    for percent in (20, 30):
        threshold = compute_exact_quantile_from_memmap(gain_path, count, percent / 100.)
        reread, bottom, ties = accumulate_bottom_by_angle(gain_path, bin_path, count, threshold)
        if not np.array_equal(reread, totals) or int(reread.sum()) != count:
            raise RuntimeError("streaming and paired-file populations differ")
        fraction = int(bottom.sum()) / count
        within_tolerance = abs(fraction-percent/100.) <= 1/count+1e-12
        if not within_tolerance and not ties:
            raise RuntimeError("global quantile fraction mismatch without threshold ties")
        fractions, expected, expected_db = bin_statistics(totals, bottom, expected_sums)
        results[percent] = dict(threshold_db=threshold, count=int(bottom.sum()), fraction=fraction,
                                threshold_tie_count=ties, fraction_matches_nominal=within_tolerance,
                                bottom_counts=bottom, fractions=fractions,
                                expected=expected, expected_db=expected_db)
    if results[20]["threshold_db"] > results[30]["threshold_db"] or np.any(results[20]["bottom_counts"] > results[30]["bottom_counts"]):
        raise RuntimeError("global bottom-20 must be a subset of global bottom-30")
    return results


def save_statistics(path, totals, statistics):
    fields = ["angle_start_deg", "angle_end_deg", "angle_center_deg", "total_count",
              "bottom20_count", "bottom20_fraction_pct", "bottom30_count", "bottom30_fraction_pct",
              "mean_expected_gain_linear", "mean_expected_gain_db"]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for i in range(BIN_COUNT):
            values = [10+i, 11+i, 10.5+i, int(totals[i]),
                      int(statistics[20]["bottom_counts"][i]), statistics[20]["fractions"][i],
                      int(statistics[30]["bottom_counts"][i]), statistics[30]["fractions"][i],
                      statistics[20]["expected"][i], statistics[20]["expected_db"][i]]
            writer.writerow(["" if isinstance(v, (float, np.floating)) and not math.isfinite(v) else v for v in values])


def save_serving_plots(directory, reservoir, statistics, *, slots, dt_s, is_debug):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    angles, gains, _ = reservoir.ordered()
    centers = np.arange(10.5, 90., 1.)
    scope = f"DEBUG: {slots} slots / {slots*dt_s:g} s" if is_debug else "16 min / 96,000 slots"
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
        fig, ax = plt.subplots(figsize=(13, 7), facecolor="white")
        try:
            ax.scatter(angles, gains, s=1.5, alpha=.15, color=".5", linewidths=0,
                       rasterized=True, label="Instantaneous raw CSI (sampled serving links)")
            ax.plot(centers, statistics[20]["expected_db"], lw=2.5, color="#df811c",
                    label="Expected power gain of serving links (geometry + mean fading)")
            ax.set(xlim=(10, 90), xlabel="Elevation to the current serving satellite (degrees)",
                   ylabel="Raw CSI power gain (dB, 10 log10 |h|^2)",
                   title=f"Serving satellite only: elevation vs. CSI\n120 stationary UE; {scope}; no interference adjustment")
            ax.grid(alpha=.25); ax.set_axisbelow(True); ax.legend(markerscale=4)
            fig.tight_layout(); fig.savefig(Path(directory)/FIGURE_NAMES[0], dpi=160)
        finally:
            plt.close(fig)
        for percent, name in zip((20, 30), FIGURE_NAMES[1:]):
            fig, ax = plt.subplots(figsize=(13, 7), facecolor="white")
            try:
                ax.plot(centers, statistics[percent]["fractions"], color="#df811c", marker="o", ms=3, lw=2)
                ax.set(xlim=(10, 90), ylim=(0, 100),
                       xlabel="Elevation to the current serving satellite (degrees; 1-degree bins)",
                       ylabel=f"Within-bin fraction in global bottom {percent}% of serving CSI (%)",
                       title=f"Serving satellite only: global bottom-{percent}% CSI by elevation\n{scope}; global cutoff = {statistics[percent]['threshold_db']:.4f} dB")
                ax.grid(alpha=.25)
                fig.text(.5, .015, "Empty angle bins are gaps, not zero. Cutoff ties are included; realized fractions are saved in run_summary.json.",
                         ha="center", fontsize=9, color=".35")
                fig.tight_layout(rect=(0, .035, 1, 1)); fig.savefig(Path(directory)/name, dpi=160)
            finally:
                plt.close(fig)


def _save_scatter_sample(path, reservoir, scenario, dt_s):
    angles, gains, ids = reservoir.ordered()
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["observation_id", "slot", "time_s", "serving_satellite_id", "ue_id", "elevation_deg", "gain_db"])
        for identity, angle, gain in zip(ids, angles, gains):
            identity = int(identity)
            slot, remainder = divmod(identity, 960)
            satellite, device = divmod(remainder, 120)
            writer.writerow([identity, slot, slot*dt_s, scenario.constellation.elements[satellite].satellite_id,
                             scenario.devices.ids[device], angle, gain])


def run_serving_csi_elevation_study(*, outputs=DEFAULT_OUTPUTS, smoke_slots=None,
                                   keep_raw=False, plot_sample_max=MAX_PLOT_POINTS):
    """Stream 96,000 slots once; compute two quantiles and exactly three figures."""
    if isinstance(plot_sample_max, bool) or not isinstance(plot_sample_max, int) or not 1 <= plot_sample_max <= MAX_PLOT_POINTS:
        raise ValueError("plot_sample_max must be an integer in 1..100000")
    base, config = build_analysis_config()
    if smoke_slots is not None and (isinstance(smoke_slots, bool) or not isinstance(smoke_slots, int)
                                    or not 1 <= smoke_slots < config.slot_count):
        raise ValueError("smoke_slots must be in 1..95999; omit for 16 minutes")
    if config.get("association.mode") != "single_serving_satellite_for_all_devices":
        raise ValueError("this diagnostic requires the existing single-serving association model")
    scenario = StudyScenario(config)
    check_source_invariants(config, scenario)
    tracker = AssociationTracker(config, scenario.subscriptions)
    slots = config.slot_count if smoke_slots is None else smoke_slots
    is_debug = smoke_slots is not None
    dt_s = float(config.get("experiment.slot_dt_s"))
    output = Path(outputs) / "debug" / f"slots_{slots}" if is_debug else Path(outputs)
    output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="_serving_raw_", dir=output))
    gain_path, bin_path = work/"gain_db.bin", work/"angle_bin.bin"
    original_coordinates = scenario.devices.ecef_km.copy()
    totals = np.zeros(BIN_COUNT, dtype=np.int64)
    expected_sums = np.zeros(BIN_COUNT)
    reservoir = PlotReservoir(plot_sample_max)
    count = handovers = low_angle_count = 0
    min_angle, max_angle = math.inf, -math.inf
    raw_hash, elevation_hash, association_hash = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    satellite_slots = {element.satellite_id: 0 for element in scenario.constellation.elements}
    buffer = []

    def flush_buffers(gain_file, bin_file):
        if not buffer:
            return
        values = np.concatenate([part[0] for part in buffer]).astype(GAIN_DTYPE, copy=False)
        bins = np.concatenate([part[1] for part in buffer]).astype(BIN_DTYPE, copy=False)
        angles = np.concatenate([part[2] for part in buffer])
        ids = np.concatenate([part[3] for part in buffer])
        values.tofile(gain_file); bins.tofile(bin_file)
        reservoir.update(angles, values, ids)
        buffer.clear()

    try:
        with gain_path.open("wb") as gain_file, bin_path.open("wb") as bin_file, \
                (work/"association_trace.csv").open("w", encoding="utf-8", newline="") as trace:
            writer = csv.DictWriter(trace, fieldnames=ASSOCIATION_FIELDS)
            writer.writeheader()
            for slot in range(slots):
                full = compute_slot_metric(config, scenario, slot, with_interference=False)
                decision = tracker.update(slot, full.elevation_deg)
                satellite = decision.serving_satellite
                item = select_serving_links(full, satellite)
                satellite_id = scenario.constellation.elements[satellite].satellite_id
                raw_hash.update(np.ascontiguousarray(full.raw_gain, dtype=GAIN_DTYPE).tobytes())
                elevation_hash.update(np.ascontiguousarray(full.elevation_deg, dtype=GAIN_DTYPE).tobytes())
                association_hash.update(np.asarray([slot, satellite, int(decision.handover)], dtype="<i8").tobytes())
                handovers += int(decision.handover)
                satellite_slots[satellite_id] += 1
                size = len(item.gain_db)
                count += size
                low_count = int(np.count_nonzero(item.angles_deg < 15.))
                low_angle_count += low_count
                if size:
                    min_angle = min(min_angle, float(item.angles_deg.min()))
                    max_angle = max(max_angle, float(item.angles_deg.max()))
                writer.writerow(dict(slot=slot, time_s=slot*dt_s, serving_satellite_id=satellite_id,
                    handover=int(decision.handover), initial_acquisition=int(decision.initial_acquisition),
                    reason=decision.reason, fallback_used=int(decision.fallback_used),
                    outage_receivers=decision.outage_receivers, visible_ue_count=size,
                    min_serving_elevation_deg=float(full.elevation_deg[satellite].min()),
                    visible_ue_below_15_count=low_count))
                totals += np.bincount(item.angle_bins, minlength=BIN_COUNT)
                expected_sums += np.bincount(item.angle_bins, weights=item.expected_gain_linear, minlength=BIN_COUNT)
                buffer.append((item.gain_db, item.angle_bins, item.angles_deg, item.observation_ids))
                if (slot+1) % BUFFER_SLOTS == 0:
                    flush_buffers(gain_file, bin_file)
                    print(f"[serving-only] {slot+1}/{slots} slots; {count:,} visible serving-link observations", flush=True)
            flush_buffers(gain_file, bin_file)
        if not count:
            raise ValueError("no visible serving links; no fabricated quantiles or figures")
        if not np.array_equal(original_coordinates, scenario.devices.ecef_km):
            raise RuntimeError("UE coordinates changed during the diagnostic")
        statistics = summarize_population(gain_path, bin_path, count, totals, expected_sums)
        save_statistics(work/"elevation_bin_statistics.csv", totals, statistics)
        _save_scatter_sample(work/"sampled_scatter_points.csv", reservoir, scenario, dt_s)
        save_serving_plots(work, reservoir, statistics, slots=slots, dt_s=dt_s, is_debug=is_debug)
        import matplotlib
        _, _, selected_ids = reservoir.ordered()
        summary = dict(status="COMPLETED_DEBUG" if is_debug else "COMPLETED",
            study="16-minute serving-satellite-only raw-CSI diagnostic", mode="serving_only_no_interference",
            sample_scope="visible links from the current single serving satellite to the 120 UE only",
            configured_duration_s=DURATION_S, time_s=slots*dt_s, last_processed_time_s=(slots-1)*dt_s,
            slot_dt_s=dt_s, slots_completed=slots, is_debug=is_debug, is_full_horizon=not is_debug,
            satellite_count=8, serving_satellite_count_per_slot=1, ue_count=120,
            association_mode=config.get("association.mode"), hysteresis_score_deg=config.get("association.hysteresis_score_deg"),
            time_to_trigger_s=config.get("association.time_to_trigger_s"), handover_count=handovers,
            serving_satellite_slot_counts=satellite_slots, phi_min_deg=float(config.get("geometry.phi_min_deg")),
            phi_theta_used_for_analysis=False, scheduler_used=False, grouping_policy_used=False, aoi_computed=False,
            metric_semantics="raw_channel_power_gain", interference_model="none",
            number_of_valid_link_observations=count, excluded_invisible_serving_link_observations=slots*120-count,
            number_of_plotted_scatter_points=len(selected_ids), plot_sample_seed=PLOT_SEED,
            statistics_use_all_visible_serving_observations=True,
            minimum_visible_serving_elevation_deg=min_angle, maximum_visible_serving_elevation_deg=max_angle,
            visible_serving_observations_below_15_deg=low_angle_count,
            angle_bin_edges_deg=list(range(10, 91)), angle_bin_total_counts=totals.tolist(),
            bin_interval_convention="[start,end), final bin [89,90]; empty bins are missing, not zero",
            bottom_percentages=[20, 30], quantile_method="linear on gain_db; exact disk-backed order statistics",
            quantile_tie_policy="include all values <= threshold; report realized fraction if ties exceed nominal fraction",
            bottom_fraction_denominator="all serving observations within the given angle bin",
            layout_hash=scenario.identity["layout_hash"], subscription_hash=scenario.identity["subscription_hash"],
            ue_positions_unchanged=True, layout_seed=int(config.get("devices.layout_seed")),
            subscription_seed=int(config.get("devices.subscription_seed")), channel_seed=int(config.get("randomness.fixed_channel_seed")),
            policy_seed=int(config.get("randomness.policy_seed")), base_config_id=base.config_id, analysis_config_id=config.config_id,
            raw_channel_trace_sha256=raw_hash.hexdigest(), elevation_trace_sha256=elevation_hash.hexdigest(),
            raw_trace_scope="full original (8,120) arrays; identical draws before serving-only selection",
            association_trace_sha256=association_hash.hexdigest(),
            plot_sample_observation_ids_sha256=hashlib.sha256(selected_ids.astype("<u8").tobytes()).hexdigest(),
            plot_sample_id_encoding="slot*960 + satellite_index*120 + ue_index",
            plot_sample_method="uniform bottom-K independent random keys; does not determine statistics",
            figure_files=list(FIGURE_NAMES), streaming_buffer_slots=BUFFER_SLOTS,
            raw_pair_bytes=count*9, quantile_working_copy_bytes=count*8, peak_temporary_binary_bytes=count*17,
            raw_files_kept=keep_raw,
            raw_files={"gain_db": str(gain_path.resolve()), "angle_bin": str(bin_path.resolve())} if keep_raw else None,
            software_versions={"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
            output_directory=str(output.resolve()))
        for percent, item in statistics.items():
            for key in ("threshold_db", "count", "fraction", "threshold_tie_count", "fraction_matches_nominal"):
                summary[f"global_bottom{percent}_{key}"] = item[key]
            summary[f"global_bottom{percent}_threshold_linear"] = float(10.**(item["threshold_db"]/10.))
        (work/"run_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        for name in (*FIGURE_NAMES, "elevation_bin_statistics.csv", "sampled_scatter_points.csv", "association_trace.csv", "run_summary.json"):
            (work/name).replace(output/name)
        if not keep_raw:
            gain_path.unlink(); bin_path.unlink(); work.rmdir()
        print(f"[serving-only] complete: {output.resolve()}", flush=True)
        return summary
    except BaseException:
        print(f"[serving-only] incomplete; intermediate files retained at {work.resolve()}", file=sys.stderr)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(
        "16-minute serving-satellite-only raw CSI: existing association/handover; "
        "three figures (angle vs CSI, global bottom 20%, global bottom 30%); no multicast/AoI simulation"))
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS, help="independent serving-only output directory")
    parser.add_argument("--smoke-slots", type=int, default=None, help="explicit DEBUG run: 1..95999; omit for 96000 slots")
    parser.add_argument("--plot-sample-max", type=int, default=MAX_PLOT_POINTS, help="1..100000 plot points; statistics use the full serving population")
    parser.add_argument("--keep-raw", action="store_true", help="retain paired binary files (max about 104 MB for 96000*120 visible links)")
    args = parser.parse_args(argv)
    try:
        result = run_serving_csi_elevation_study(outputs=args.outputs, smoke_slots=args.smoke_slots,
                                               keep_raw=args.keep_raw, plot_sample_max=args.plot_sample_max)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
