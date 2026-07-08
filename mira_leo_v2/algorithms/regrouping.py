from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Group:
    satellite_id: int
    message_id: int
    group_id: int
    users: list[dict]
    group_gain: float = 0.0
    group_elevation: float = 0.0
    is_edge: bool = False
    scheduled: bool = False
    power: float = 0.0
    sinr: float = 0.0
    rate: float = 0.0
    success: bool = False
    priority: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def user_ids(self) -> list[int]:
        return [int(user["id"]) for user in self.users]


def make_equal_cardinality_groups(
    users: list[dict], satellite_id: int, action_a: int, num_messages: int
) -> list[Group]:
    """
    Sort users by CSI in ascending order and split each message into action_a groups.
    """
    if action_a <= 0:
        raise ValueError("action_a must be positive")

    groups: list[Group] = []
    for message_id in range(num_messages):
        subscribers = [
            user
            for user in users
            if user["message_id"] == message_id
            and user["serving_sat"] == satellite_id
        ]
        if not subscribers:
            continue

        subscribers = sorted(
            subscribers, key=lambda user: float(user["channel_gain"][satellite_id])
        )
        chunks = np.array_split(np.array(subscribers, dtype=object), action_a)

        for group_id, chunk in enumerate(chunks):
            if len(chunk) == 0:
                continue

            chunk_users = list(chunk)
            gains = [
                float(user["channel_gain"][satellite_id]) for user in chunk_users
            ]
            elevations = [
                float(user["elevation"][satellite_id]) for user in chunk_users
            ]
            groups.append(
                Group(
                    satellite_id=satellite_id,
                    message_id=message_id,
                    group_id=group_id,
                    users=chunk_users,
                    group_gain=float(min(gains)),
                    group_elevation=float(max(elevations)),
                )
            )

    return groups


def classify_edge_groups(
    groups: list[Group], phi_edge: float
) -> tuple[list[Group], list[Group]]:
    edge_groups: list[Group] = []
    nonedge_groups: list[Group] = []

    for group in groups:
        sat_id = group.satellite_id
        elevations = [float(user["elevation"][sat_id]) for user in group.users]
        group.group_elevation = max(elevations) if elevations else 0.0
        group.is_edge = bool(group.group_elevation < phi_edge)

        if group.is_edge:
            edge_groups.append(group)
        else:
            nonedge_groups.append(group)

    return edge_groups, nonedge_groups
