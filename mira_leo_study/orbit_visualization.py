from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .geometry import ecef_to_spherical, radial_project_to_sphere
from .orbit import SyntheticConstellation


PLANE_COLORS = {1: "#1f77b4", 2: "#e4572e"}
FONT_SIZE_PT = 20


def sample_eci_trajectory(config, duration_s: float, frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample the configured constellation without changing the orbit model."""
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    if frames < 2:
        raise ValueError("frames must be at least 2")
    constellation = SyntheticConstellation(config)
    times = np.linspace(0.0, float(duration_s), int(frames))
    positions = np.stack([constellation.state(float(time_s))[0] for time_s in times])
    return times, positions


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.animation as animation
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Orbit visualization requires matplotlib and Pillow; install requirements.txt first"
        ) from exc
    return plt, animation


def _apply_presentation_fonts(plt) -> None:
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE_PT,
            "axes.titlesize": FONT_SIZE_PT,
            "axes.labelsize": FONT_SIZE_PT,
            "xtick.labelsize": FONT_SIZE_PT,
            "ytick.labelsize": FONT_SIZE_PT,
            "legend.fontsize": FONT_SIZE_PT,
            "figure.titlesize": FONT_SIZE_PT,
        }
    )


def _study_center_eci(config, constellation: SyntheticConstellation, time_s: float) -> np.ndarray:
    earth_radius = float(config.get("geometry.earth_radius_km"))
    center_ecef = radial_project_to_sphere(
        config.get("geometry.center_raw_ecef_km"), earth_radius
    )
    angle = constellation.theta0 + constellation.earth_rotation * float(time_s)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    return rotation @ center_ecef


def _camera_angles(point: np.ndarray) -> tuple[float, float]:
    radius = float(np.linalg.norm(point))
    elevation = math.degrees(math.asin(float(point[2]) / radius))
    azimuth = math.degrees(math.atan2(float(point[1]), float(point[0])))
    return elevation, azimuth


def _earth_surface(ax, radius_km: float) -> None:
    longitude = np.linspace(0.0, 2.0 * math.pi, 73)
    colatitude = np.linspace(0.0, math.pi, 37)
    lon, colat = np.meshgrid(longitude, colatitude)
    x = radius_km * np.sin(colat) * np.cos(lon)
    y = radius_km * np.sin(colat) * np.sin(lon)
    z = radius_km * np.cos(colat)
    ax.plot_surface(x, y, z, color="#5f9ec7", alpha=0.48, linewidth=0.0, shade=True)
    ax.plot_wireframe(
        x,
        y,
        z,
        rstride=6,
        cstride=8,
        color="#e8f3f8",
        alpha=0.35,
        linewidth=0.45,
    )


def _format_3d_axes(ax, orbit_radius_km: float) -> None:
    limit = orbit_radius_km * 1.12
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    major_ticks = (-6000.0, 0.0, 6000.0)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.set_zticks(major_ticks)
    ax.set_xlabel("ECI x (km)", labelpad=20)
    ax.set_ylabel("ECI y (km)", labelpad=20)
    ax.set_zlabel("ECI z (km)", labelpad=24)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE_PT, pad=6)
    ax.grid(True, alpha=0.2)


def _full_plane_paths(config, points: int = 361) -> tuple[SyntheticConstellation, dict[int, np.ndarray]]:
    constellation = SyntheticConstellation(config)
    period_s = float(config.get("derived.orbital_period_s"))
    times = np.linspace(0.0, period_s, points)
    representative: dict[int, int] = {}
    for index, element in enumerate(constellation.elements):
        representative.setdefault(int(element.plane), index)
    paths = {
        plane: np.stack([constellation.state(float(time_s))[0][index] for time_s in times])
        for plane, index in representative.items()
    }
    return constellation, paths


def _break_longitude_wrap(longitude_deg: np.ndarray, latitude_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    longitude = np.asarray(longitude_deg, dtype=float).copy()
    latitude = np.asarray(latitude_deg, dtype=float).copy()
    jumps = np.flatnonzero(np.abs(np.diff(longitude)) > 180.0) + 1
    longitude[jumps] = np.nan
    latitude[jumps] = np.nan
    return longitude, latitude


def create_orbit_plot(config, output_directory: str | Path, duration_s: float, dpi: int = 180) -> dict:
    plt, _ = _load_matplotlib()
    _apply_presentation_fonts(plt)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    constellation, plane_paths = _full_plane_paths(config)
    earth_radius = float(config.get("geometry.earth_radius_km"))
    orbit_radius = float(config.get("derived.semi_major_axis_km"))

    figure = plt.figure(figsize=(21.0, 10.5), constrained_layout=True)
    orbit_axis = figure.add_subplot(1, 2, 1, projection="3d")
    _earth_surface(orbit_axis, earth_radius)
    for plane, path in sorted(plane_paths.items()):
        orbit_axis.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            color=PLANE_COLORS[plane],
            linewidth=2.0,
            label=f"Orbital plane {plane}",
        )
    initial_eci, _ = constellation.state(0.0)
    for index, element in enumerate(constellation.elements):
        color = PLANE_COLORS[int(element.plane)]
        orbit_axis.scatter(*initial_eci[index], color=color, edgecolor="white", s=88, linewidth=1.1)
    study_center_eci = _study_center_eci(config, constellation, 0.0)
    orbit_axis.scatter(
        *study_center_eci,
        marker="*",
        s=300,
        color="#202020",
        edgecolor="white",
        linewidth=1.0,
        label="Study-region center",
        zorder=8,
    )
    _format_3d_axes(orbit_axis, orbit_radius)
    camera_elevation, camera_azimuth = _camera_angles(study_center_eci)
    orbit_axis.view_init(elev=camera_elevation, azim=camera_azimuth)
    orbit_axis.set_title("(a) ECI orbital geometry")
    orbit_axis.legend(loc="upper left", fontsize=FONT_SIZE_PT)

    ground_axis = figure.add_subplot(1, 2, 2)
    sample_count = max(121, min(601, int(math.ceil(duration_s)) + 1))
    sample_times = np.linspace(0.0, duration_s, sample_count)
    ecef_samples = np.stack(
        [constellation.state(float(time_s))[1] for time_s in sample_times]
    )
    lat, lon = ecef_to_spherical(ecef_samples.reshape(-1, 3))
    latitude_deg = np.degrees(lat).reshape(sample_count, -1)
    longitude_deg = np.degrees(lon).reshape(sample_count, -1)
    labeled_planes: set[int] = set()
    for index, element in enumerate(constellation.elements):
        plane = int(element.plane)
        longitude, latitude = _break_longitude_wrap(
            longitude_deg[:, index], latitude_deg[:, index]
        )
        label = f"Plane {plane} ground tracks" if plane not in labeled_planes else None
        ground_axis.plot(
            longitude,
            latitude,
            color=PLANE_COLORS[plane],
            alpha=0.82,
            linewidth=1.15,
            label=label,
        )
        labeled_planes.add(plane)
    center = radial_project_to_sphere(
        config.get("geometry.center_raw_ecef_km"), earth_radius
    )
    center_lat, center_lon = ecef_to_spherical(center)
    center_lat_deg = math.degrees(float(center_lat))
    center_lon_deg = math.degrees(float(center_lon))
    all_relative_longitudes = []
    all_relative_latitudes = []
    ground_axis.clear()
    labeled_planes.clear()
    for index, element in enumerate(constellation.elements):
        plane = int(element.plane)
        relative_longitude = (
            longitude_deg[:, index] - center_lon_deg + 180.0
        ) % 360.0 - 180.0
        relative_latitude = latitude_deg[:, index] - center_lat_deg
        relative_longitude, relative_latitude = _break_longitude_wrap(
            relative_longitude, relative_latitude
        )
        all_relative_longitudes.append(relative_longitude)
        all_relative_latitudes.append(relative_latitude)
        label = f"Plane {plane} ground tracks" if plane not in labeled_planes else None
        ground_axis.plot(
            relative_longitude,
            relative_latitude,
            color=PLANE_COLORS[plane],
            alpha=0.86,
            linewidth=2.2,
            label=label,
        )
        labeled_planes.add(plane)
    ground_axis.scatter(
        0.0,
        0.0,
        marker="*",
        s=360,
        color="#2b2b2b",
        label="Study-region center",
        zorder=5,
    )
    finite_longitudes = np.concatenate(all_relative_longitudes)
    finite_latitudes = np.concatenate(all_relative_latitudes)
    longitude_span = max(60.0, math.ceil(np.nanmax(np.abs(finite_longitudes)) / 10.0) * 10.0 + 10.0)
    latitude_span = max(50.0, math.ceil(np.nanmax(np.abs(finite_latitudes)) / 10.0) * 10.0 + 10.0)
    ground_axis.set_xlim(-min(longitude_span, 180.0), min(longitude_span, 180.0))
    ground_axis.set_ylim(-min(latitude_span, 90.0), min(latitude_span, 90.0))
    ground_axis.set_xlabel("Longitude relative to study center (deg)")
    ground_axis.set_ylabel("Latitude relative to study center (deg)")
    ground_axis.set_title(
        f"(b) Ground tracks centered at ({center_lat_deg:.1f} deg, {center_lon_deg:.1f} deg)"
    )
    ground_axis.grid(True, alpha=0.28)
    ground_axis.axhline(0.0, color="#555555", linewidth=0.8, alpha=0.35)
    ground_axis.axvline(0.0, color="#555555", linewidth=0.8, alpha=0.35)
    ground_axis.legend(loc="lower left", fontsize=FONT_SIZE_PT)

    figure.suptitle(
        "MIRA-LEO Synthetic Constellation\n"
        f"8 satellites, altitude {config.get('constellation.altitude_km'):g} km, "
        f"inclination {config.get('constellation.inclination_deg'):g} deg",
        fontsize=FONT_SIZE_PT,
    )
    png_path = output / "mira_leo_synthetic_constellation_orbits.png"
    pdf_path = output / "mira_leo_synthetic_constellation_orbits.pdf"
    figure.savefig(png_path, dpi=int(dpi), facecolor="white")
    figure.savefig(pdf_path, facecolor="white")
    plt.close(figure)
    return {"png": str(png_path.resolve()), "pdf": str(pdf_path.resolve())}


def create_orbit_animation(
    config,
    output_directory: str | Path,
    duration_s: float,
    frames: int = 121,
    fps: int = 20,
    video_format: str = "auto",
    dpi: int = 120,
) -> dict:
    if fps <= 0:
        raise ValueError("fps must be positive")
    plt, animation = _load_matplotlib()
    _apply_presentation_fonts(plt)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    times, positions = sample_eci_trajectory(config, duration_s, frames)
    constellation, plane_paths = _full_plane_paths(config)
    earth_radius = float(config.get("geometry.earth_radius_km"))
    orbit_radius = float(config.get("derived.semi_major_axis_km"))

    requested = video_format.lower()
    if requested not in {"auto", "gif", "mp4"}:
        raise ValueError("video_format must be auto, gif, or mp4")
    ffmpeg_available = bool(animation.writers.is_available("ffmpeg"))
    selected = "mp4" if requested == "auto" and ffmpeg_available else requested
    if selected == "auto":
        selected = "gif"
    if selected == "mp4" and not ffmpeg_available:
        raise RuntimeError("MP4 output requires ffmpeg; use --video-format gif")

    figure = plt.figure(figsize=(10.5, 10.5))
    axis = figure.add_subplot(111, projection="3d")
    _earth_surface(axis, earth_radius)
    for plane, path in sorted(plane_paths.items()):
        axis.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            color=PLANE_COLORS[plane],
            alpha=0.7,
            linewidth=1.5,
            label=f"Orbital plane {plane}",
        )
    _format_3d_axes(axis, orbit_radius)
    axis.view_init(elev=24.0, azim=38.0)

    plane_indices: dict[int, list[int]] = {}
    for index, element in enumerate(constellation.elements):
        plane_indices.setdefault(int(element.plane), []).append(index)
    scatters = {}
    for plane, indices in sorted(plane_indices.items()):
        current = positions[0, indices]
        scatters[plane] = axis.scatter(
            current[:, 0],
            current[:, 1],
            current[:, 2],
            s=110,
            color=PLANE_COLORS[plane],
            edgecolor="white",
            linewidth=0.7,
        )
    trails = [
        axis.plot([], [], [], color=PLANE_COLORS[int(element.plane)], alpha=0.42, linewidth=1.0)[0]
        for element in constellation.elements
    ]
    study_centers = np.stack(
        [_study_center_eci(config, constellation, float(time_s)) for time_s in times]
    )
    study_marker = axis.scatter(
        *study_centers[0],
        marker="*",
        s=360,
        color="#202020",
        edgecolor="white",
        linewidth=1.0,
        label="Study-region center",
        zorder=8,
    )
    title = axis.set_title("")
    axis.legend(loc="upper left", fontsize=FONT_SIZE_PT)
    trail_frames = max(2, int(round(frames * min(0.20, 40.0 / max(duration_s, 1.0)))))

    def update(frame_index: int):
        for plane, indices in plane_indices.items():
            current = positions[frame_index, indices]
            scatters[plane]._offsets3d = (current[:, 0], current[:, 1], current[:, 2])
        start = max(0, frame_index - trail_frames)
        for satellite_index, trail in enumerate(trails):
            segment = positions[start : frame_index + 1, satellite_index]
            trail.set_data(segment[:, 0], segment[:, 1])
            trail.set_3d_properties(segment[:, 2])
        center = study_centers[frame_index]
        study_marker._offsets3d = ([center[0]], [center[1]], [center[2]])
        title.set_text(
            "MIRA-LEO Synthetic Constellation\n"
            f"t = {times[frame_index]:.1f} s / {duration_s:.1f} s"
        )
        camera_elevation, camera_azimuth = _camera_angles(center)
        axis.view_init(elev=camera_elevation, azim=camera_azimuth)
        return [*scatters.values(), *trails, study_marker, title]

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=len(times),
        interval=1000.0 / fps,
        blit=False,
    )
    if selected == "mp4":
        video_path = output / "mira_leo_synthetic_constellation_orbit.mp4"
        writer = animation.FFMpegWriter(fps=fps, bitrate=2200, metadata={"title": "MIRA-LEO Synthetic Constellation"})
    else:
        video_path = output / "mira_leo_synthetic_constellation_orbit.gif"
        writer = animation.PillowWriter(fps=fps)
    movie.save(video_path, writer=writer, dpi=int(dpi))
    plt.close(figure)
    return {
        "video": str(video_path.resolve()),
        "video_format": selected,
        "ffmpeg_available": ffmpeg_available,
        "frames": int(frames),
        "fps": int(fps),
        "playback_duration_s": float(frames / fps),
    }


def generate_orbit_visuals(
    config,
    output_directory: str | Path,
    duration_s: float | None = None,
    frames: int = 121,
    fps: int = 20,
    video_format: str = "auto",
    dpi: int = 180,
) -> dict:
    duration = (
        float(config.get("experiment.duration_s"))
        if duration_s is None
        else float(duration_s)
    )
    output = Path(output_directory) / "orbit_visualization"
    plots = create_orbit_plot(config, output, duration, dpi=dpi)
    animation_result = create_orbit_animation(
        config,
        output,
        duration,
        frames=frames,
        fps=fps,
        video_format=video_format,
        dpi=min(int(dpi), 140),
    )
    constellation = SyntheticConstellation(config)
    metadata = {
        "model": "synthetic two-plane circular LEO constellation",
        "reference_frame": "ECI for orbit plot and animation; ECEF for ground tracks",
        "satellite_count": len(constellation.elements),
        "plane_satellite_counts": {
            str(plane): sum(int(element.plane) == plane for element in constellation.elements)
            for plane in sorted({int(element.plane) for element in constellation.elements})
        },
        "altitude_km": float(config.get("constellation.altitude_km")),
        "semi_major_axis_km": float(config.get("derived.semi_major_axis_km")),
        "inclination_deg": float(config.get("constellation.inclination_deg")),
        "orbital_period_s": float(config.get("derived.orbital_period_s")),
        "visualized_duration_s": duration,
        "font_size_pt": FONT_SIZE_PT,
        "view_center": "configured study-region center",
        **plots,
        **animation_result,
    }
    metadata_path = output / "orbit_visualization_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {**metadata, "metadata": str(metadata_path.resolve())}
