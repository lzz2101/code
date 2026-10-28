from __future__ import annotations

import datetime as dt
import math

import numpy as np

from .geometry import ecef_to_spherical, radial_project_to_sphere
from .models import SatelliteElement


def _rz(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def gmst_rad(epoch_utc: str) -> float:
    epoch = dt.datetime.fromisoformat(epoch_utc.replace("Z", "+00:00"))
    unix = epoch.timestamp()
    jd = 2440587.5 + unix / 86400.0
    centuries = (jd - 2451545.0) / 36525.0
    degrees = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * centuries**2
        - centuries**3 / 38710000.0
    ) % 360.0
    return math.radians(degrees)


class SyntheticConstellation:
    def __init__(self, config):
        self.config = config
        self.radius_km = float(config.get("derived.semi_major_axis_km"))
        self.mean_motion = float(config.get("derived.mean_motion_rad_s"))
        self.inclination_rad = math.radians(float(config.get("constellation.inclination_deg")))
        self.earth_rotation = float(config.get("geometry.earth_rotation_rad_s"))
        self.theta0 = gmst_rad(str(config.get("experiment.epoch_utc")))

        earth_radius = float(config.get("geometry.earth_radius_km"))
        center = radial_project_to_sphere(config.get("geometry.center_raw_ecef_km"), earth_radius)
        center_lat, center_lon = ecef_to_spherical(center)
        t_mid = float(config.get("constellation.center_overflight_s"))
        ratio = math.sin(float(center_lat)) / math.sin(self.inclination_rad)
        if abs(ratio) > 1.0:
            raise ValueError("center latitude cannot be reached by the configured inclination")
        u_mid = math.asin(ratio)
        self.u_ref0 = u_mid - self.mean_motion * t_mid
        az_term = math.atan2(
            math.cos(self.inclination_rad) * math.sin(u_mid), math.cos(u_mid)
        )
        raan1 = (
            float(center_lon)
            + self.theta0
            + self.earth_rotation * t_mid
            - az_term
        ) % (2.0 * math.pi)
        raan2 = raan1 + math.radians(float(config.get("constellation.plane_raan_difference_deg")))
        offsets = [
            (1, value) for value in config.get("constellation.plane1_phase_offsets_deg")
        ] + [(2, value) for value in config.get("constellation.plane2_phase_offsets_deg")]
        self.elements = tuple(
            SatelliteElement(
                satellite_id=f"sat_{index + 1:02d}",
                plane=plane,
                raan_rad=raan1 if plane == 1 else raan2,
                u0_rad=(self.u_ref0 + math.radians(float(offset))) % (2.0 * math.pi),
            )
            for index, (plane, offset) in enumerate(offsets)
        )

    def state(self, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        eci = np.empty((len(self.elements), 3), dtype=float)
        for index, element in enumerate(self.elements):
            u = element.u0_rad + self.mean_motion * float(time_s)
            orbital = self.radius_km * np.array([math.cos(u), math.sin(u), 0.0])
            eci[index] = _rz(element.raan_rad) @ _rx(self.inclination_rad) @ orbital
        ecef = (_rz(-(self.theta0 + self.earth_rotation * float(time_s))) @ eci.T).T
        return eci, ecef

    def initial_rows(self) -> list[dict[str, float | int | str]]:
        eci, ecef = self.state(0.0)
        rows = []
        for index, element in enumerate(self.elements):
            rows.append(
                {
                    "satellite_id": element.satellite_id,
                    "plane": element.plane,
                    "raan_deg": math.degrees(element.raan_rad) % 360.0,
                    "u0_deg": math.degrees(element.u0_rad) % 360.0,
                    "eci_x_km": float(eci[index, 0]),
                    "eci_y_km": float(eci[index, 1]),
                    "eci_z_km": float(eci[index, 2]),
                    "ecef_x_km": float(ecef[index, 0]),
                    "ecef_y_km": float(ecef[index, 1]),
                    "ecef_z_km": float(ecef[index, 2]),
                }
            )
        return rows
