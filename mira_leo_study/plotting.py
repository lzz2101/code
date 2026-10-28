from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from .config import PHI_THETA_CANDIDATES_DEG
from .errors import OptionalDependencyError
from .geometry import aeqd_forward, aeqd_inverse, ecef_to_spherical, footprint_central_angle_rad, radial_project_to_sphere


def _libraries():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise OptionalDependencyError(
            "Plot generation requires matplotlib and pandas; install requirements.txt"
        ) from exc
    return plt, pd


def _save(figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")


def _service_square(axis):
    axis.plot([-500, 500, 500, -500, -500], [-500, -500, 500, 500, -500], color="black", linewidth=1.0)
    axis.set_xlim(-550, 550)
    axis.set_ylim(-550, 550)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("East (km)")
    axis.set_ylabel("North (km)")


def plot_geometry(config, output_directory: str | Path) -> list[str]:
    plt, pd = _libraries()
    root = Path(output_directory)
    geometry = root / "geometry"
    figures = root / "figures"
    generated = []
    if not (geometry / "satellites_1s.parquet").exists():
        return generated
    satellites = pd.read_parquet(geometry / "satellites_1s.parquet")
    devices = pd.read_csv(geometry / "devices.csv")
    radius = float(config.get("geometry.earth_radius_km"))
    center = radial_project_to_sphere(config.get("geometry.center_raw_ecef_km"), radius)
    center_lat, center_lon = ecef_to_spherical(center)

    sat_xyz = satellites[["ecef_x_km", "ecef_y_km", "ecef_z_km"]].to_numpy()
    sat_lat, sat_lon = ecef_to_spherical(sat_xyz)
    east, north = aeqd_forward(sat_lat, sat_lon, float(center_lat), float(center_lon), radius)
    satellites = satellites.assign(east_km=east, north_km=north)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(devices["east_km"], devices["north_km"], s=7, color="#777777", alpha=0.6, label="devices")
    for satellite_id, frame in satellites.groupby("satellite_id"):
        plane = int(frame["plane"].iloc[0])
        ax.plot(frame["east_km"], frame["north_km"], linewidth=1.0, alpha=0.8, color="#1769aa" if plane == 1 else "#d1495b")
    _service_square(ax)
    ax.set_title("Synthetic ground tracks and fixed devices")
    _save(fig, figures / "01_ground_tracks_devices")
    plt.close(fig)
    generated.append("01_ground_tracks_devices")

    central_angle = footprint_central_angle_rad(
        radius,
        float(config.get("constellation.altitude_km")),
        float(config.get("geometry.phi_min_deg")),
    )
    snapshots = [0.0, 120.0, 240.0, 360.0, 480.0]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True)
    azimuth = np.linspace(0.0, 2.0 * math.pi, 181)
    for axis, time_s in zip(axes, snapshots):
        frame = satellites[np.isclose(satellites["time_s"], time_s)]
        for _, row in frame.iterrows():
            position = np.array([row.ecef_x_km, row.ecef_y_km, row.ecef_z_km])
            lat, lon = ecef_to_spherical(position)
            boundary_lat = np.arcsin(
                np.sin(lat) * math.cos(central_angle)
                + np.cos(lat) * math.sin(central_angle) * np.cos(azimuth)
            )
            boundary_lon = lon + np.arctan2(
                np.sin(azimuth) * math.sin(central_angle) * np.cos(lat),
                math.cos(central_angle) - np.sin(lat) * np.sin(boundary_lat),
            )
            x, y = aeqd_forward(boundary_lat, boundary_lon, float(center_lat), float(center_lon), radius)
            axis.plot(x, y, linewidth=0.6, alpha=0.7)
        _service_square(axis)
        axis.set_title(f"t={int(time_s)} s")
    fig.suptitle("Visibility footprints at 10 degree minimum elevation")
    _save(fig, figures / "02_footprint_snapshots")
    plt.close(fig)
    generated.append("02_footprint_snapshots")

    grid_file = geometry / "grid_coverage.npz"
    if grid_file.exists():
        grid = np.load(grid_file)
        counts = grid["visible_satellite_count"]
        side = int(round(math.sqrt(counts.shape[1])))
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for axis, values, title in zip(
            axes,
            (np.min(counts, axis=0), np.mean(counts, axis=0), np.max(counts, axis=0)),
            ("Minimum", "Mean", "Maximum"),
        ):
            image = axis.imshow(values.reshape(side, side), origin="lower", extent=(-500, 500, -500, 500), vmin=0, vmax=8)
            axis.set_title(title)
            axis.set_xlabel("East (km)")
        axes[0].set_ylabel("North (km)")
        fig.colorbar(image, ax=axes.ravel().tolist(), label="Visible satellites")
        _save(fig, figures / "03_grid_coverage")
        plt.close(fig)
        generated.append("03_grid_coverage")

    association_file = geometry / "association.parquet"
    if association_file.exists():
        association = pd.read_parquet(association_file)
        fig, ax = plt.subplots(figsize=(11, 4))
        for cohort, frame in association.groupby("cohort_id"):
            satellite_number = frame["satellite_id"].str.split("_").str[-1].astype(int)
            ax.step(frame["time_s"], satellite_number + 0.08 * int(cohort), where="post", label=f"cohort_{cohort}")
            handovers = frame[frame["handover"] == 1]
            ax.scatter(handovers["time_s"], satellite_number.loc[handovers.index] + 0.08 * int(cohort), marker="x", s=20)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Serving satellite number")
        ax.legend()
        ax.set_title("Serving-satellite and handover timeline")
        _save(fig, figures / "04_association_handover_timeline")
        plt.close(fig)
        generated.append("04_association_handover_timeline")
    return generated


def plot_parameter_comparisons(config, output_directory: str | Path) -> list[str]:
    plt, pd = _libraries()
    root = Path(output_directory)
    aggregate_path = root / "comparison" / "aggregate_by_config.csv"
    if not aggregate_path.exists():
        return []
    data = pd.read_csv(aggregate_path)
    data = data[data["legal_for_ranking"] == 1]
    if data.empty:
        return []
    figures = root / "figures"
    generated = []
    dimensions = [
        ("phi_theta_deg", "05_phi_comparison", "Edge threshold (degree)"),
        ("action_set_name", "06_action_set_comparison", "Action set"),
        ("V_c", "07_power_fraction_comparison", "V_c"),
        ("R_theta", "08_rate_threshold_comparison", "R_theta (bit/s/Hz)"),
    ]
    for column, filename, label in dimensions:
        subset = data.dropna(subset=[column, "weighted_mean_aoi_ms"])
        if subset.empty:
            continue
        grouped = subset.groupby(column, as_index=False).agg(
            weighted_mean_aoi_ms=("weighted_mean_aoi_ms", "min"),
            goodput=("capacity_goodput_bps", "max"),
            tail_aoi=("aoi_p95_ms", "min"),
        )
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        x = np.arange(len(grouped))
        labels = [str(value) for value in grouped[column]]
        axes[0].plot(x, grouped["weighted_mean_aoi_ms"], marker="o")
        axes[0].set_ylabel("Weighted mean AoI (ms)")
        axes[1].plot(x, grouped["goodput"], marker="o", color="#198754")
        axes[1].set_ylabel("Capacity goodput (bit/s)")
        axes[2].plot(x, grouped["tail_aoi"], marker="o", color="#d1495b")
        axes[2].set_ylabel("AoI p95 (ms)")
        for axis in axes:
            axis.set_xticks(x, labels, rotation=25 if column == "action_set_name" else 0)
            axis.set_xlabel(label)
            axis.grid(alpha=0.25)
        if column == "R_theta":
            threshold = float(config.get("algorithm.R_theta"))
            fig.suptitle(f"Spectral-efficiency threshold comparison; R_theta={threshold:.6g}")
        _save(fig, figures / filename)
        plt.close(fig)
        generated.append(filename)

    for columns, filename in [
        (("phi_theta_deg", "V_c"), "10a_joint_phi_v"),
    ]:
        subset = data.dropna(subset=[*columns, "weighted_mean_aoi_ms"])
        if subset.empty:
            continue
        pivot = subset.pivot_table(index=columns[0], columns=columns[1], values="weighted_mean_aoi_ms", aggfunc="min")
        fig, ax = plt.subplots(figsize=(7, 5))
        image = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower")
        ax.set_xticks(np.arange(len(pivot.columns)), [str(value) for value in pivot.columns], rotation=25)
        ax.set_yticks(np.arange(len(pivot.index)), [str(value) for value in pivot.index])
        ax.set_xlabel(columns[1])
        ax.set_ylabel(columns[0])
        fig.colorbar(image, ax=ax, label="Weighted mean AoI (ms)")
        _save(fig, figures / filename)
        plt.close(fig)
        generated.append(filename)
    return generated


def generate_all_plots(config, output_directory: str | Path) -> list[str]:
    return plot_geometry(config, output_directory) + plot_parameter_comparisons(config, output_directory)


def load_phi_aoi_trace(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an exact raw fixed-seed trace; never substitute or synthesize data."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"required baseline trace is absent: {source}")
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise OptionalDependencyError("Phi trace plotting requires pyarrow") from exc
    table = pq.read_table(
        source,
        columns=["slot", "aggregated_weighted_sum_aoi_ms"],
    )
    rows = table.to_pylist()
    by_slot: dict[int, float] = {}
    for row in rows:
        slot = int(row["slot"])
        value = float(row["aggregated_weighted_sum_aoi_ms"])
        if slot in by_slot and not math.isclose(by_slot[slot], value, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"inconsistent aggregate AoI rows for slot {slot}")
        by_slot[slot] = value
    if not by_slot:
        raise ValueError(f"trace contains no AoI rows: {source}")
    slots = np.asarray(sorted(by_slot), dtype=int)
    values = np.asarray([by_slot[int(slot)] for slot in slots], dtype=float)
    return slots, values


def plot_phi_study_results(output_directory: str | Path) -> list[str]:
    """Render only traces that genuinely exist in a completed phi-study directory."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise OptionalDependencyError("Phi-study plotting requires matplotlib") from exc
    root = Path(output_directory)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    policies = (
        ("mira_sw_ucb", "MIRA-LEO"),
        ("aim_static", "AIM-static"),
        ("ppo_multicast", "PPO-multicast"),
        ("ga_grouping", "GA-grouping"),
        ("random_grouping", "Random grouping"),
        ("fixed_grouping", "Fixed(a=4)"),
    )
    generated: list[str] = []
    mira_traces: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for phi in PHI_THETA_CANDIDATES_DEG:
        traces = []
        for policy, label in policies:
            trace_path = (
                root
                / "validation"
                / f"phi_{phi}"
                / policy
                / "fixed_seed_0"
                / "slot_metrics.parquet"
            )
            summary_path = trace_path.with_name("summary.json")
            if not trace_path.exists() or not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                summary.get("status") != "COMPLETED"
                or int(summary.get("channel_seed", -1)) != 0
                or int(summary.get("slots_completed", 0)) <= 0
            ):
                continue
            slots, values = load_phi_aoi_trace(trace_path)
            expected_slots = int(summary["slots_completed"])
            if (
                slots.size != expected_slots
                or int(slots[0]) != 0
                or int(slots[-1]) != expected_slots - 1
            ):
                raise ValueError(f"incomplete slot trace for completed run: {trace_path}")
            traces.append((label, slots, values))
            if policy == "mira_sw_ucb":
                mira_traces[phi] = (slots, values)
        if not traces:
            continue
        figure, axis = plt.subplots(figsize=(11, 6))
        for label, slots, values in traces:
            axis.plot(slots, values, linewidth=1.1, label=label)
        axis.set_xlabel("Simulation slot")
        axis.set_ylabel("Aggregated weighted-sum AoI (ms)")
        axis.set_title(f"Convergence and Long-Run AoI, phi_theta={phi} deg")
        axis.grid(alpha=0.25)
        axis.legend()
        base = figures / f"phi_{phi}_long_run_weighted_sum_aoi"
        _save(figure, base)
        plt.close(figure)
        generated.extend([str(base.with_suffix(".png")), str(base.with_suffix(".pdf"))])

    if mira_traces:
        figure, axis = plt.subplots(figsize=(11, 6))
        for phi, (slots, values) in sorted(mira_traces.items()):
            axis.plot(slots, values, linewidth=1.0, label=f"{phi} deg")
        axis.set_xlabel("Simulation slot")
        axis.set_ylabel("Aggregated weighted-sum AoI (ms)")
        axis.set_title("MIRA-LEO phi_theta Long-Run Comparison")
        axis.grid(alpha=0.25)
        axis.legend(ncol=2)
        base = figures / "mira_phi_theta_long_run_comparison"
        _save(figure, base)
        plt.close(figure)
        generated.extend([str(base.with_suffix(".png")), str(base.with_suffix(".pdf"))])

    summary_path = root / "comparison" / "phi_fixed_seed_summary.csv"
    if summary_path.exists():
        with summary_path.open("r", newline="", encoding="utf-8") as handle:
            rows = [
                row
                for row in csv.DictReader(handle)
                if row.get("policy") == "mira_sw_ucb"
                and row.get("status") == "COMPLETED"
                and row.get("weighted_mean_aoi_after_warmup_ms") not in {None, ""}
            ]
        if rows:
            rows.sort(key=lambda row: int(float(row["phi_theta_deg"])))
            x = [int(float(row["phi_theta_deg"])) for row in rows]
            y = [float(row["weighted_mean_aoi_after_warmup_ms"]) for row in rows]
            figure, axis = plt.subplots(figsize=(9, 5.5))
            axis.plot(x, y, marker="o", linewidth=1.5)
            axis.set_xticks(x)
            axis.set_xlabel("phi_theta (degree)")
            axis.set_ylabel("Post-warmup weighted mean AoI (ms)")
            axis.set_title("Fixed-Seed Phi-Theta Summary")
            axis.grid(alpha=0.25)
            path = figures / "phi_theta_vs_long_run_aoi.png"
            figure.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(figure)
            generated.append(str(path))
    return generated
