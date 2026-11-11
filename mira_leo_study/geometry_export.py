from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .association import AssociationTracker
from .geometry import generate_coverage_grid, link_geometry
from .isl import compute_routes
from .scenario import StudyScenario
from .sources import build_source_registry
from .storage import AppendCsvWriter, atomic_write_json, csv_to_parquet, write_csv, write_npz


def export_geometry(config, output_directory: str | Path) -> dict:
    output_root = Path(output_directory)
    geometry_dir = output_root / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    scenario = StudyScenario(config)
    atomic_write_json(output_root / "source_registry.json", build_source_registry(config))
    write_csv(geometry_dir / "constellation.csv", scenario.constellation.initial_rows())

    device_rows = [
        {
            "device_id": scenario.devices.ids[index],
            "east_km": scenario.devices.east_km[index],
            "north_km": scenario.devices.north_km[index],
            "spherical_latitude_deg": math.degrees(scenario.devices.latitude_rad[index]),
            "longitude_deg": math.degrees(scenario.devices.longitude_rad[index]),
            "ecef_x_km": scenario.devices.ecef_km[index, 0],
            "ecef_y_km": scenario.devices.ecef_km[index, 1],
            "ecef_z_km": scenario.devices.ecef_km[index, 2],
            "layout_hash": scenario.devices.layout_hash,
        }
        for index in range(len(scenario.devices.ids))
    ]
    write_csv(geometry_dir / "devices.csv", device_rows)
    subscription_rows = []
    for index, device_id in enumerate(scenario.devices.ids):
        message = int(scenario.subscriptions.message_by_device[index])
        subscription_rows.append(
            {
                "device_id": device_id,
                "active_subscription": int(message >= 0),
                "message_id": message if message >= 0 else "",
                "subscription_hash": scenario.subscriptions.subscription_hash,
            }
        )
    write_csv(geometry_dir / "subscriptions.csv", subscription_rows)

    satellite_partial = geometry_dir / "_satellites_1s.partial.csv"
    device_partial = geometry_dir / "_device_geometry_1s.partial.csv"
    satellite_writer = AppendCsvWriter(satellite_partial)
    device_writer = AppendCsvWriter(device_partial)
    export_dt = float(config.get("experiment.geometry_export_dt_s"))
    sample_times = np.arange(0.0, float(config.get("experiment.duration_s")) + 0.5 * export_dt, export_dt)
    grid_east, grid_north, grid_ecef = generate_coverage_grid(config)
    grid_counts = np.empty((len(sample_times), len(grid_ecef)), dtype=np.uint8)
    routes_rows = []
    for time_index, time_s in enumerate(sample_times):
        eci, ecef = scenario.constellation.state(float(time_s))
        distance, elevation = link_geometry(ecef, scenario.devices.ecef_km)
        _, grid_elevation = link_geometry(ecef, grid_ecef)
        grid_counts[time_index] = np.count_nonzero(
            grid_elevation >= float(config.get("geometry.phi_min_deg")), axis=0
        )
        satellite_writer.append(
            [
                {
                    "time_s": time_s,
                    "satellite_id": element.satellite_id,
                    "plane": element.plane,
                    "eci_x_km": eci[index, 0],
                    "eci_y_km": eci[index, 1],
                    "eci_z_km": eci[index, 2],
                    "ecef_x_km": ecef[index, 0],
                    "ecef_y_km": ecef[index, 1],
                    "ecef_z_km": ecef[index, 2],
                }
                for index, element in enumerate(scenario.constellation.elements)
            ]
        )
        rows = []
        for satellite, element in enumerate(scenario.constellation.elements):
            for device, device_id in enumerate(scenario.devices.ids):
                rows.append(
                    {
                        "time_s": time_s,
                        "satellite_id": element.satellite_id,
                        "device_id": device_id,
                        "slant_range_km": distance[satellite, device],
                        "elevation_deg": elevation[satellite, device],
                        "visible": int(elevation[satellite, device] >= float(config.get("geometry.phi_min_deg"))),
                    }
                )
        device_writer.append(rows)
        for target, route in compute_routes(config, ecef).items():
            routes_rows.append(
                {
                    "time_s": time_s,
                    "target_satellite_id": scenario.constellation.elements[target].satellite_id,
                    "reachable": int(route.reachable),
                    "path": ">".join(scenario.constellation.elements[index].satellite_id for index in route.path),
                    "hop_count": route.hop_count,
                    "distance_km": route.distance_km,
                    "fixed_delay_s": route.fixed_delay_s,
                    "geometric_delay_s": route.geometric_delay_s,
                }
            )
    csv_to_parquet(satellite_partial, geometry_dir / "satellites_1s.parquet")
    csv_to_parquet(device_partial, geometry_dir / "device_geometry_1s.parquet")
    satellite_partial.unlink(missing_ok=True)
    device_partial.unlink(missing_ok=True)
    write_npz(
        geometry_dir / "grid_coverage.npz",
        time_s=sample_times,
        east_km=grid_east,
        north_km=grid_north,
        visible_satellite_count=grid_counts,
    )

    association_partial = geometry_dir / "_association.partial.csv"
    association_writer = AppendCsvWriter(association_partial)
    tracker = AssociationTracker(config, scenario.subscriptions)
    handover_rows = []
    dt_s = float(config.get("experiment.slot_dt_s"))
    chunk = []
    for slot in range(config.slot_count):
        time_s = slot * dt_s
        _, ecef = scenario.constellation.state(time_s)
        _, elevation = link_geometry(ecef, scenario.devices.ecef_km)
        decision = tracker.update(slot, elevation)
        satellite = decision.serving_satellite
        chunk.append(
            {
                "slot": slot,
                "time_s": time_s,
                "satellite_id": scenario.constellation.elements[satellite].satellite_id,
                "handover": int(decision.handover),
                "initial_acquisition": int(decision.initial_acquisition),
                "outage_receivers": decision.outage_receivers,
                "fallback_used": int(decision.fallback_used),
                "assignment_score_deg": decision.score_deg,
                "reason": decision.reason,
            }
        )
        if decision.handover or decision.initial_acquisition:
            handover_rows.append(dict(chunk[-1]))
        if len(chunk) >= 2000:
            association_writer.append(chunk)
            chunk = []
    association_writer.append(chunk)
    csv_to_parquet(association_partial, geometry_dir / "association.parquet")
    association_partial.unlink(missing_ok=True)
    write_csv(geometry_dir / "handover_events.csv", handover_rows)
    from .storage import write_parquet

    write_parquet(geometry_dir / "isl_routes.parquet", routes_rows)
    summary = {
        "layout_hash": scenario.devices.layout_hash,
        "subscription_hash": scenario.subscriptions.subscription_hash,
        "satellite_count": len(scenario.constellation.elements),
        "device_count": len(scenario.devices.ids),
        "grid_point_count": len(grid_ecef),
        "geometry_time_count": len(sample_times),
        "communication_slot_count": config.slot_count,
        "grid_visible_min": int(np.min(grid_counts)),
        "grid_visible_mean": float(np.mean(grid_counts)),
        "grid_visible_max": int(np.max(grid_counts)),
    }
    atomic_write_json(geometry_dir / "geometry_summary.json", summary)
    return summary
