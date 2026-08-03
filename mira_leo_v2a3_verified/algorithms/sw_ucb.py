from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite

import numpy as np


@dataclass(frozen=True)
class SWUCBObservation:
    satellite_id: int | None
    slot: int
    arm: int
    reward: float


class SWUCB:
    """
    Sliding-Window UCB for MIRA-LEO regrouping.

    Each arm is a candidate number of multicast groups per message.
    """

    def __init__(
        self,
        action_set,
        tau: int,
        xi: float = 1.0 / np.sqrt(2.0),
        satellite_id: int | None = None,
    ):
        if tau <= 0:
            raise ValueError("tau must be positive")
        if len(action_set) == 0:
            raise ValueError("action_set must not be empty")

        self.action_set = list(action_set)
        self.tau = int(tau)
        self.xi = float(xi)
        self.satellite_id = satellite_id
        self.observations: deque[SWUCBObservation] = deque()
        self.selected_actions: list[int] = []
        self.observed_rewards: list[float] = []
        self.observed_slots: list[int] = []

    def _window_start(self, slot: int) -> int:
        return max(0, int(slot) - self.tau)

    def _prune(self, slot: int) -> None:
        window_start = self._window_start(slot)
        while self.observations and self.observations[0].slot < window_start:
            self.observations.popleft()

    def get_window_records(self, slot: int) -> list[SWUCBObservation]:
        self._prune(slot)
        return [record for record in self.observations if record.slot < slot]

    def select_arm(self, t: int) -> int:
        """Select action a_l,*(t), using t = 0, 1, ..., T-1."""
        best_arm = None
        best_index = -np.inf
        log_term = np.log(max(1, min(t, self.tau)))

        for arm in self.action_set:
            count = self.get_count(t, arm)
            if count == 0:
                return arm

            mean_reward = self.get_mean_reward(t, arm)
            bonus = self.xi * np.sqrt(log_term / count)
            index = mean_reward + bonus

            if index > best_index:
                best_index = index
                best_arm = arm

        return int(best_arm)

    def update(self, slot: int, arm: int, reward: float) -> None:
        """Update sliding-window statistics after observing reward."""
        if arm not in self.action_set:
            raise ValueError(f"unknown arm: {arm}")
        if slot < 0:
            raise ValueError("slot must be non-negative")
        if self.observed_slots and slot <= self.observed_slots[-1]:
            raise ValueError("SW-UCB updates must use strictly increasing slots")
        reward = float(reward)
        if not isfinite(reward) or reward < 0.0 or reward > 1.0:
            raise ValueError(
                f"SW-UCB reward must be finite and in [0,1]: "
                f"satellite={self.satellite_id}, slot={slot}, arm={arm}, reward={reward}"
            )
        self._prune(slot + 1)
        self.observations.append(
            SWUCBObservation(self.satellite_id, int(slot), int(arm), reward)
        )
        self.selected_actions.append(int(arm))
        self.observed_rewards.append(reward)
        self.observed_slots.append(int(slot))

    def get_mean_reward(self, slot: int, arm: int) -> float:
        rewards = [
            record.reward
            for record in self.get_window_records(slot)
            if record.arm == arm
        ]
        if not rewards:
            return 0.0
        return float(np.mean(rewards))

    def get_count(self, slot: int, arm: int) -> int:
        return sum(
            record.arm == arm for record in self.get_window_records(slot)
        )

    def get_debug_info(self, slot: int) -> dict[int, dict[str, float]]:
        return {
            arm: {
                "count": float(self.get_count(slot, arm)),
                "mean_reward": self.get_mean_reward(slot, arm),
            }
            for arm in self.action_set
        }
