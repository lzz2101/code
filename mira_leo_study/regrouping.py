"""One-second held partitions and epoch-indexed bandit bookkeeping."""

from __future__ import annotations

import hashlib
from collections import Counter

from .algorithms import build_group_partition, materialize_groups_from_partition


def initial_epoch_state() -> dict:
    return {
        "held_action": None,
        "held_partition": None,
        "held_partition_hash": None,
        "epoch_action_owner_satellite": None,
        "epoch_reward_sum": 0.0,
        "epoch_reward_count": 0,
        "epoch_reward_finalized": False,
        "last_regrouping_slot": None,
        "action_decision_histogram": Counter(),
        "bandit_observation_count": 0,
        "diagnostic_boundary_csi": None,
    }


def partition_hash(partition) -> str:
    digest = hashlib.sha256()
    for entry in partition:
        digest.update(repr((entry.message_id, entry.group_id, tuple(sorted(entry.member_indices)))).encode("ascii"))
    return digest.hexdigest()


def groups_for_slot(
    config, state, *, satellite, slot, policy, policy_seed,
    fixed_action, message_members, channel_amplitude, elevation_deg, select_action,
    diagnostics_enabled=False,
) -> tuple[int, list, int]:
    """A handover changes materialization satellite, never the held partition."""
    interval = int(config.get("derived.regrouping_interval_slots"))
    epoch = slot // interval
    boundary = slot % interval == 0
    action_changed = 0
    if boundary or state["held_partition"] is None:
        agent = state["agents"][satellite] if state["agents"] is not None else None
        action = select_action(
            policy, actions=config.get("algorithm.action_set"), slot=epoch,
            satellite=satellite, policy_seed=policy_seed, fixed_action=fixed_action, agent=agent,
        )
        partition = build_group_partition(config, satellite, action,
                                          message_members, channel_amplitude)
        previous = state["previous_action"]
        action_changed = int(previous is not None and previous != action)
        state["previous_action"] = action
        state["held_action"] = action
        state["held_partition"] = partition
        state["held_partition_hash"] = partition_hash(partition)
        state["epoch_action_owner_satellite"] = satellite
        state["last_regrouping_slot"] = slot
        state["epoch_reward_sum"] = 0.0
        state["epoch_reward_count"] = 0
        state["epoch_reward_finalized"] = False
        state["action_decision_histogram"][(satellite, action)] += 1
    action = state["held_action"]
    partition = state["held_partition"]
    state["action_histogram"][(satellite, action)] += 1
    state["action_change_events"] += action_changed
    groups = materialize_groups_from_partition(config, satellite, partition,
                                               channel_amplitude, elevation_deg)
    if diagnostics_enabled:
        from .diagnostic_trace import csi_ranks
        if boundary or state["diagnostic_boundary_csi"] is None:
            ranks = csi_ranks(groups)
            state["diagnostic_boundary_csi"] = {
                g.key: (g.gain, ranks[g.key]) for g in groups}
        for group in groups:
            gain, rank = state["diagnostic_boundary_csi"][group.key]
            group.metadata.update(boundary_group_gain=gain, boundary_group_csi_rank=rank)
    return action, groups, action_changed


def observe_slot_reward(config, state, *, slot, slot_limit, raw_reward) -> tuple[float, float | None]:
    """Normalize per slot, then update the original action owner once per epoch.

    Only slot_limit marks the true end. A checkpoint pause must not finalize a
    partial epoch, so stop_after_slots is deliberately absent from this API.
    """
    owner = state["epoch_action_owner_satellite"]
    normalized = state["normalizers"][owner].transform(raw_reward)
    if state["epoch_reward_finalized"]:
        raise RuntimeError("epoch reward was already finalized")
    state["epoch_reward_sum"] += normalized
    state["epoch_reward_count"] += 1
    interval = int(config.get("derived.regrouping_interval_slots"))
    epoch_reward = None
    if (slot + 1) % interval == 0 or slot + 1 == slot_limit:
        epoch_reward = state["epoch_reward_sum"] / state["epoch_reward_count"]
        state["agents"][owner].update(slot // interval, state["held_action"], epoch_reward)
        state["epoch_reward_finalized"] = True
        state["bandit_observation_count"] += 1
    return normalized, epoch_reward
