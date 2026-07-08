from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


class SWUCB:
    """
    Sliding-Window UCB for MIRA-LEO regrouping.

    Each arm is a candidate number of multicast groups per message.
    """

    def __init__(self, action_set, tau: int, xi: float = 1.0 / np.sqrt(2.0)):
        if tau <= 0:
            raise ValueError("tau must be positive")
        if len(action_set) == 0:
            raise ValueError("action_set must not be empty")

        self.action_set = list(action_set)
        self.tau = int(tau)
        self.xi = float(xi)
        self.reward_history = defaultdict(lambda: deque(maxlen=self.tau))
        self.selected_actions: list[int] = []
        self.observed_rewards: list[float] = []

    def select_arm(self, t: int) -> int:
        """Select action a_l,*(t), using t = 0, 1, ..., T-1."""
        if t < len(self.action_set):
            return self.action_set[t]

        best_arm = None
        best_index = -np.inf
        log_term = np.log(max(1, min(t + 1, self.tau)))

        for arm in self.action_set:
            count = len(self.reward_history[arm])
            if count == 0:
                return arm

            mean_reward = float(np.mean(self.reward_history[arm]))
            bonus = self.xi * np.sqrt(log_term / count)
            index = mean_reward + bonus

            if index > best_index:
                best_index = index
                best_arm = arm

        return int(best_arm)

    def update(self, arm: int, reward: float) -> None:
        """Update sliding-window statistics after observing reward."""
        if arm not in self.action_set:
            raise ValueError(f"unknown arm: {arm}")
        reward = float(reward)
        self.reward_history[arm].append(reward)
        self.selected_actions.append(int(arm))
        self.observed_rewards.append(reward)

    def get_mean_reward(self, arm: int) -> float:
        if len(self.reward_history[arm]) == 0:
            return 0.0
        return float(np.mean(self.reward_history[arm]))

    def get_count(self, arm: int) -> int:
        return len(self.reward_history[arm])

    def get_debug_info(self) -> dict[int, dict[str, float]]:
        return {
            arm: {
                "count": float(self.get_count(arm)),
                "mean_reward": self.get_mean_reward(arm),
            }
            for arm in self.action_set
        }
