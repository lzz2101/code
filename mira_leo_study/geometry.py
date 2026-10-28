from __future__ import annotations

import hashlib
import math
from typing import Iterable

import numpy as np

from .models import DeviceTable


def radial_project_to_sphere(raw_ecef_km: Iterable[float], radius_km: float) -> np.ndarray:
    raw = np.asarray(tuple(raw_ecef_km), dtype=float)
    if raw.shape != (3,) or not np.all(np.isfinite(raw)):
        raise ValueError("raw ECEF center must contain three finite coordinates")
    norm = float(np.linalg.norm(raw))
    if norm <= 0.0 or radius_km <= 0.0:
        raise ValueError("center norm and radius must be positive")
    return raw * (float(radius_km) / norm)


def ecef_to_spherical(ecef_km: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(ecef_km, dtype=float)
    radii = np.linalg.norm(xyz, axis=-1)
    lat = np.arcsin(np.clip(xyz[..., 2] / radii, -1.0, 1.0))
    lon = np.arctan2(xyz[..., 1], xyz[..., 0])
    return lat, lon


def spherical_to_ecef(latitude_rad: np.ndarray, longitude_rad: np.ndarray, radius_km: float) -> np.ndarray:
    lat = np.asarray(latitude_rad, dtype=float)
    lon = np.asarray(longitude_rad, dtype=float)
    cos_lat = np.cos(lat)
    return float(radius_km) * np.stack(
        (cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)), axis=-1
    )


def aeqd_inverse(
    east_km: np.ndarray,
    north_km: np.ndarray,
    center_lat_rad: float,
    center_lon_rad: float,
    radius_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    east = np.asarray(east_km, dtype=float)
    north = np.asarray(north_km, dtype=float)
    distance = np.hypot(east, north)
    rho = distance / float(radius_km)
    azimuth = np.arctan2(east, north)
    sin_lat0 = math.sin(center_lat_rad)
    cos_lat0 = math.cos(center_lat_rad)
    sin_rho = np.sin(rho)
    cos_rho = np.cos(rho)
    latitude = np.arcsin(
        np.clip(
            sin_lat0 * cos_rho + cos_lat0 * sin_rho * np.cos(azimuth),
            -1.0,
            1.0,
        )
    )
    longitude = center_lon_rad + np.arctan2(
        np.sin(azimuth) * sin_rho * cos_lat0,
        cos_rho - sin_lat0 * np.sin(latitude),
    )
    origin = distance == 0.0
    latitude = np.where(origin, center_lat_rad, latitude)
    longitude = np.where(origin, center_lon_rad, longitude)
    longitude = (longitude + math.pi) % (2.0 * math.pi) - math.pi
    return latitude, longitude


def aeqd_forward(
    latitude_rad: np.ndarray,
    longitude_rad: np.ndarray,
    center_lat_rad: float,
    center_lon_rad: float,
    radius_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(latitude_rad, dtype=float)
    lon = np.asarray(longitude_rad, dtype=float)
    delta_lon = (lon - center_lon_rad + math.pi) % (2.0 * math.pi) - math.pi
    sin_lat0 = math.sin(center_lat_rad)
    cos_lat0 = math.cos(center_lat_rad)
    cos_c = np.clip(
        sin_lat0 * np.sin(lat) + cos_lat0 * np.cos(lat) * np.cos(delta_lon),
        -1.0,
        1.0,
    )
    c = np.arccos(cos_c)
    sin_c = np.sin(c)
    scale = np.divide(c, sin_c, out=np.ones_like(c), where=np.abs(sin_c) > 1e-15)
    east = float(radius_km) * scale * np.cos(lat) * np.sin(delta_lon)
    north = float(radius_km) * scale * (
        cos_lat0 * np.sin(lat) - sin_lat0 * np.cos(lat) * np.cos(delta_lon)
    )
    return east, north


def generate_devices(config, layout_seed: int | None = None) -> DeviceTable:
    count = int(config.get("devices.unique_count"))
    half_width = float(config.get("geometry.region_half_width_km"))
    radius = float(config.get("geometry.earth_radius_km"))
    center = radial_project_to_sphere(config.get("geometry.center_raw_ecef_km"), radius)
    center_lat, center_lon = ecef_to_spherical(center)
    seed = int(config.get("devices.layout_seed") if layout_seed is None else layout_seed)
    rng = np.random.default_rng(seed)
    east = rng.uniform(-half_width, half_width, count)
    north = rng.uniform(-half_width, half_width, count)
    lat, lon = aeqd_inverse(east, north, float(center_lat), float(center_lon), radius)
    ecef = spherical_to_ecef(lat, lon, radius)
    ids = tuple(f"ue_{index:03d}" for index in range(count))
    digest = hashlib.sha256()
    digest.update(np.asarray(east, dtype="<f8").tobytes())
    digest.update(np.asarray(north, dtype="<f8").tobytes())
    digest.update("|".join(ids).encode("ascii"))
    return DeviceTable(ids, east, north, lat, lon, ecef, digest.hexdigest())


def generate_coverage_grid(config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    half = float(config.get("geometry.region_half_width_km"))
    spacing = float(config.get("geometry.grid_spacing_km"))
    axis = np.arange(-half, half + spacing * 0.5, spacing, dtype=float)
    east, north = np.meshgrid(axis, axis, indexing="xy")
    radius = float(config.get("geometry.earth_radius_km"))
    center = radial_project_to_sphere(config.get("geometry.center_raw_ecef_km"), radius)
    center_lat, center_lon = ecef_to_spherical(center)
    lat, lon = aeqd_inverse(east.ravel(), north.ravel(), float(center_lat), float(center_lon), radius)
    return east.ravel(), north.ravel(), spherical_to_ecef(lat, lon, radius)


def link_geometry(satellite_ecef_km: np.ndarray, ground_ecef_km: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    satellites = np.asarray(satellite_ecef_km, dtype=float)
    ground = np.asarray(ground_ecef_km, dtype=float)
    relative = satellites[:, None, :] - ground[None, :, :]
    distance = np.linalg.norm(relative, axis=2)
    normals = ground / np.linalg.norm(ground, axis=1)[:, None]
    projection = np.einsum("sui,ui->su", relative, normals)
    elevation = np.degrees(np.arcsin(np.clip(projection / distance, -1.0, 1.0)))
    return distance, elevation


def footprint_central_angle_rad(radius_km: float, altitude_km: float, phi_min_deg: float) -> float:
    phi = math.radians(phi_min_deg)
    return math.acos(radius_km / (radius_km + altitude_km) * math.cos(phi)) - phi
