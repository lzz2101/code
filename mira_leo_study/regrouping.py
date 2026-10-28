"""One-second held partitions and epoch-indexed bandit bookkeeping."""

from __future__ import annotations

import hashlib
from collections import Counter

from .algorithms import build_group_partition, materialize_groups_from_partition


def initial_epoch_state() -> dict:
    return {
        "held_actions_by_cohort": {},
        "held_partitions": {},
        "held_partition_hash": {},
        "epoch_action_owner_satellite": {},
        "epoch_reward_sum": {},
        "epoch_reward_count": {},
        "epoch_reward_finalized": {},
        "last_regrouping_slot": {},
        "action_decision_histogram": Counter(),
        "bandit_observation_count": 0,
    }


def partition_hash(partition) -> str:
    digest = hashlib.sha256()
    for entry in partition:
        digest.update(repr((entry.message_id, entry.group_id, tuple(sorted(entry.member_indices)))).encode("ascii"))
    return digest.hexdigest()


def groups_for_slot(
    config, state, *, cohort, satellite, slot, policy, policy_seed,
    fixed_action, message_members, channel_amplitude, elevation_deg, select_action,
) -> tuple[int, list, int]:
    """A handover changes materialization satellite, never the held partition."""
    interval = int(config.get("derived.regrouping_interval_slots"))
    epoch = slot // interval
    boundary = slot % interval == 0
    action_changed = 0
    if boundary or cohort not in state["held_partitions"]:
        agent = state["agents"][satellite] if state["agents"] is not None else None
        action = select_action(
            policy, actions=config.get("algorithm.action_set"), slot=epoch,
            satellite=satellite, policy_seed=policy_seed, fixed_action=fixed_action, agent=agent,
        )
        partition = build_group_partition(config, satellite, cohort, action,
                                          message_members, channel_amplitude)
        previous = state["previous_actions"].get(cohort)
        action_changed = int(previous is not None and previous != action)
        state["previous_actions"][cohort] = action
        state["held_actions_by_cohort"][cohort] = action
        state["held_partitions"][cohort] = partition
        state["held_partition_hash"][cohort] = partition_hash(partition)
        state["epoch_action_owner_satellite"][cohort] = satellite
        state["last_regrouping_slot"][cohort] = slot
        state["epoch_reward_sum"][cohort] = 0.0
        state["epoch_reward_count"][cohort] = 0
        state["epoch_reward_finalized"][cohort] = False
        state["action_decision_histogram"][(satellite, action)] += 1
    action = state["held_actions_by_cohort"][cohort]
    partition = state["held_partitions"][cohort]
    state["action_histogram"][(satellite, action)] += 1
    state["action_change_events"] += action_changed
    groups = materialize_groups_from_partition(config, satellite, cohort, partition,
                                               channel_amplitude, elevation_deg)
    return action, groups, action_changed


def observe_slot_reward(config, state, *, cohort, slot, slot_limit, raw_reward) -> tuple[float, float | None]:
    """Normalize per slot, then update the original action owner once per epoch.

    Only slot_limit marks the true end. A checkpoint pause must not finalize a
    partial epoch, so stop_after_slots is deliberately absent from this API.
    """
    owner = state["epoch_action_owner_satellite"][cohort]
    normalized = state["normalizers"][owner].transform(raw_reward)
    if state["epoch_reward_finalized"].get(cohort):
        raise RuntimeError("epoch reward was already finalized")
    state["epoch_reward_sum"][cohort] += normalized
    state["epoch_reward_count"][cohort] += 1
    interval = int(config.get("derived.regrouping_interval_slots"))
    epoch_reward = None
    if (slot + 1) % interval == 0 or slot + 1 == slot_limit:
        epoch_reward = state["epoch_reward_sum"][cohort] / state["epoch_reward_count"][cohort]
        state["agents"][owner].update(slot // interval, state["held_actions_by_cohort"][cohort], epoch_reward)
        state["epoch_reward_finalized"][cohort] = True
        state["bandit_observation_count"] += 1
    return normalized, epoch_reward
