from __future__ import annotations

import heapq
import math

import numpy as np

from .models import Route


def segment_clear_of_sphere(point_a: np.ndarray, point_b: np.ndarray, radius_km: float) -> bool:
    delta = point_b - point_a
    denominator = float(np.dot(delta, delta))
    if denominator == 0.0:
        return False
    fraction = float(np.clip(-np.dot(point_a, delta) / denominator, 0.0, 1.0))
    closest = point_a + fraction * delta
    return float(np.linalg.norm(closest)) > float(radius_km)


def build_isl_graph(satellite_ecef_km: np.ndarray, radius_km: float, max_distance_km: float) -> dict[int, list[tuple[int, float]]]:
    positions = np.asarray(satellite_ecef_km, dtype=float)
    graph = {index: [] for index in range(len(positions))}
    for left in range(len(positions)):
        for right in range(left + 1, len(positions)):
            distance = float(np.linalg.norm(positions[left] - positions[right]))
            if distance <= max_distance_km and segment_clear_of_sphere(
                positions[left], positions[right], radius_km
            ):
                graph[left].append((right, distance))
                graph[right].append((left, distance))
    for edges in graph.values():
        edges.sort(key=lambda item: item[0])
    return graph


def shortest_route(graph: dict[int, list[tuple[int, float]]], source: int, target: int, c_km_s: float, fixed_per_hop_s: float) -> Route:
    if source == target:
        return Route(True, (source,), 0.0, 0, 0.0, 0.0)
    queue: list[tuple[float, tuple[int, ...], int]] = [(0.0, (source,), source)]
    best: dict[int, tuple[float, tuple[int, ...]]] = {source: (0.0, (source,))}
    while queue:
        distance, path, node = heapq.heappop(queue)
        if node == target:
            hops = len(path) - 1
            return Route(
                True,
                path,
                distance,
                hops,
                hops * float(fixed_per_hop_s),
                distance / float(c_km_s),
            )
        if best.get(node) != (distance, path):
            continue
        for neighbor, edge_distance in graph[node]:
            candidate = (distance + edge_distance, path + (neighbor,))
            if neighbor not in best or candidate < best[neighbor]:
                best[neighbor] = candidate
                heapq.heappush(queue, (candidate[0], candidate[1], neighbor))
    return Route(False, (), math.inf, 0, math.inf, math.inf)


def compute_routes(config, satellite_ecef_km: np.ndarray) -> dict[int, Route]:
    radius = float(config.get("geometry.earth_radius_km"))
    graph = build_isl_graph(
        satellite_ecef_km,
        radius,
        float(config.get("delay.isl_max_link_distance_km")),
    )
    ingress_name = str(config.get("delay.ingress_satellite"))
    ingress = int(ingress_name.split("_")[-1]) - 1
    return {
        target: shortest_route(
            graph,
            ingress,
            target,
            float(config.get("geometry.light_speed_km_s")),
            float(config.get("delay.fixed_isl_per_hop_s")),
        )
        for target in graph
    }


def route_delay_s(config, route: Route) -> float:
    if not route.reachable:
        return math.inf
    if config.get("delay.isl_mode") == "fixed_per_hop":
        return route.fixed_delay_s
    return route.geometric_delay_s
