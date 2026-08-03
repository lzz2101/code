from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from mira_leo_v1.channel import compute_channel_gain
from mira_leo_v1.geometry import (
    assign_serving_satellite,
    compute_distance_and_elevation,
    initialize_users,
    update_satellite_positions,
)
from mira_leo_v2a3_verified.algorithms.power_allocation import (
    mira_leo_power_allocation,
)
from mira_leo_v2a3_verified.algorithms.regrouping import (
    classify_edge_groups,
    make_equal_cardinality_groups,
)
from mira_leo_v2a3_verified.algorithms.scheduler import handover_aware_scheduler
from mira_leo_v2a3_verified.algorithms.sw_ucb import SWUCB
from mira_leo_v2a3_verified.aoi.metrics import (
    compute_regrouping_reward_details,
    compute_weighted_sum_aoi,
    update_user_aoi,
)
from mira_leo_v2a3_verified.config import V2Config
from mira_leo_v2a3_verified.noma.transmission import transmit_noma
from mira_leo_v2a3_verified.validation.validators import (
    validate_allocation_summary,
    validate_power_constraints,
    validate_reward_record,
    validate_sic_success,
    validate_time_config,
)


def empty_histories(action_set: tuple[int, ...]) -> dict[str, list]:
    histories: dict[str, list] = {
        "action": [],
        "actions_by_satellite": [],
        "reward": [],
        "raw_reward": [],
        "rewards_by_satellite": [],
        "raw_rewards_by_satellite": [],
        "weighted_sum_aoi": [],
        "success_count": [],
        "edge_count": [],
        "total_groups": [],
        "handover_count": [],
        "scheduled_edge_count": [],
        "scheduled_nonedge_count": [],
        "positive_priority_count": [],
        "avg_priority": [],
        "edge_power": [],
        "nonedge_power": [],
        "total_allocated_power": [],
        "per_satellite_total_power": [],
        "power_floor_infeasible_count": [],
        "sic_power_order_violations": [],
        "max_sic_power_order_violation": [],
        "sic_decode_violations": [],
    }
    for action in action_set:
        histories[f"arm_{action}_count"] = []
        histories[f"arm_{action}_mean_reward"] = []
    return histories


def attach_slot_observations(
    users: list[dict], channel_gain: np.ndarray, elevation: np.ndarray, cfg: V2Config
) -> None:
    for user in users:
        user_id = int(user["id"])
        user["channel_gain"] = channel_gain[:, user_id]
        user["elevation"] = elevation[:, user_id]
        user["weight"] = (
            cfg.message_weight[user["message_id"]]
            if len(cfg.message_weight) == cfg.M
            else 1.0 / cfg.M
        )


def select_action(
    policy: str,
    slot: int,
    sw_ucb: SWUCB,
    rng: np.random.Generator,
    cfg: V2Config,
    fixed_action: int,
) -> int:
    if policy == "sw-ucb":
        return sw_ucb.select_arm(slot)
    if policy == "fixed":
        if fixed_action not in cfg.action_set:
            raise ValueError(f"fixed_action must be in {cfg.action_set}")
        return int(fixed_action)
    if policy == "random":
        return int(rng.choice(cfg.action_set))
    raise ValueError(f"unknown policy: {policy}")


def update_policy(
    policy: str, sw_ucb: SWUCB, slot: int, action: int, reward: float
) -> None:
    if policy == "sw-ucb":
        sw_ucb.update(slot=slot, arm=action, reward=reward)


def _set_handover_event_ids(users: list[dict], slot: int) -> None:
    for user in users:
        if bool(user.get("handover", 0)):
            user["handover_event_id"] = (
                int(slot), user.get("prev_serving_sat"), user.get("serving_sat")
            )


def _group_validation_record(group, slot: int) -> dict[str, object]:
    return {
        "slot": slot,
        "satellite_id": int(group.satellite_id),
        "message_id": int(group.message_id),
        "group_id": int(group.group_id),
        "is_edge": bool(group.is_edge),
        "scheduled": bool(group.scheduled),
        "channel_gain": float(group.group_gain),
        "sic_rank": int(group.metadata.get("global_sic_rank", 0)),
        "power_floor": float(group.metadata.get("floor_power", 0.0)),
        "allocated_power": float(group.power),
        "sinr": float(group.sinr),
        "rate": float(group.rate),
        "success": bool(group.success),
        "violation_type": "",
    }


def run_mira_leo_v2a3_verified(
    cfg: V2Config | None = None,
    policy: str = "sw-ucb",
    fixed_action: int = 4,
    fading: bool | None = None,
    validate: bool = False,
    strict_validation: bool = False,
) -> dict[str, object]:
    """Run Algorithms 1-3 with per-satellite policies and runtime validation."""
    cfg = cfg or V2Config()
    cfg.validate_time_settings()
    cfg.validate_power_settings()
    validate_time_config(cfg, strict=True)

    fading = cfg.fading_enabled if fading is None else fading
    rng = np.random.default_rng(cfg.random_seed)
    users = initialize_users(cfg, rng)
    sw_ucb_policies = [
        SWUCB(cfg.action_set, cfg.tau, cfg.xi, satellite_id=sat_id)
        for sat_id in range(cfg.L)
    ]
    histories = empty_histories(cfg.action_set)
    logs: list[dict] = []
    validation_satellite_records: list[dict[str, object]] = []
    validation_group_records: list[dict[str, object]] = []
    validation_violations: list[dict[str, object]] = []
    last_groups = []
    reward_min, reward_max = cfg.reward_raw_bounds

    for slot in range(cfg.T):
        sat_positions = update_satellite_positions(slot, cfg)
        distance, elevation, _ = compute_distance_and_elevation(
            users, sat_positions, cfg
        )
        handover_count = assign_serving_satellite(users, elevation, cfg)
        _set_handover_event_ids(users, slot)
        channel_gain = compute_channel_gain(distance, cfg, rng, fading=fading)
        attach_slot_observations(users, channel_gain, elevation, cfg)

        all_groups = []
        all_scheduled_groups = []
        satellite_contexts: list[dict[str, object]] = []

        for sat_id in range(cfg.L):
            sw_ucb = sw_ucb_policies[sat_id]
            action = select_action(
                policy, slot, sw_ucb, rng, cfg, fixed_action
            )
            groups_l = make_equal_cardinality_groups(
                users=users,
                satellite_id=sat_id,
                action_a=action,
                num_messages=cfg.M,
            )
            classify_edge_groups(groups_l, cfg.phi_theta)
            scheduled_groups_l = handover_aware_scheduler(groups_l, cfg, distance)
            allocation_summary = mira_leo_power_allocation(
                groups_l,
                P_sat=cfg.P_sat,
                V_c=cfg.V_c,
                V_nc=cfg.V_nc,
                V_phi=cfg.V_phi,
                slot=slot,
                satellite_id=sat_id,
                enforce_global_sic_order=True,
            )

            validate_power_constraints(
                groups_l,
                P_sat=cfg.P_sat,
                slot=slot,
                satellite_id=sat_id,
                strict=True,
            )
            validate_allocation_summary(allocation_summary, strict=True)
            transmission_summary = transmit_noma(
                scheduled_groups_l,
                noise_power=cfg.noise_power,
                rate_threshold=cfg.R_threshold,
            )
            validate_sic_success(
                groups_l,
                noise_power=cfg.noise_power,
                rate_threshold=cfg.R_threshold,
                slot=slot,
                strict=True,
            )

            all_groups.extend(groups_l)
            all_scheduled_groups.extend(scheduled_groups_l)
            satellite_contexts.append(
                {
                    "satellite_id": sat_id,
                    "action": action,
                    "groups": groups_l,
                    "scheduled_groups": scheduled_groups_l,
                    "allocation": allocation_summary,
                    "transmission": transmission_summary,
                }
            )

        update_user_aoi(users, all_groups, distance, cfg, slot=slot)

        actions_by_satellite: list[int] = []
        rewards_by_satellite: list[float] = []
        raw_rewards_by_satellite: list[float] = []
        satellite_records: list[dict[str, object]] = []

        for context in satellite_contexts:
            sat_id = int(context["satellite_id"])
            action = int(context["action"])
            groups_l = context["groups"]
            scheduled_groups_l = context["scheduled_groups"]
            allocation = context["allocation"]
            success_count_l = sum(group.success for group in scheduled_groups_l)
            weighted_sum_aoi_l = compute_weighted_sum_aoi(users, cfg, sat_id)
            edge_count_l = sum(group.is_edge for group in groups_l)
            total_groups_l = len(groups_l)

            reward_details = compute_regrouping_reward_details(
                success_count=success_count_l,
                weighted_sum_aoi=weighted_sum_aoi_l,
                edge_count=edge_count_l,
                B=cfg.B_edge,
                normalize=cfg.normalize_reward,
                reward_min=reward_min,
                reward_max=reward_max,
            )
            reward_record = {
                "slot": slot,
                "satellite_id": sat_id,
                "action": action,
                "success_count": success_count_l,
                "total_groups": total_groups_l,
                "raw_weighted_sum_aoi": weighted_sum_aoi_l,
                "edge_count": edge_count_l,
                **reward_details,
            }
            reward_violations = validate_reward_record(
                reward_record, strict=strict_validation
            )
            validation_violations.extend(
                violation.to_dict() for violation in reward_violations
            )
            normalized_reward = float(reward_details["normalized_reward"])
            update_policy(
                policy,
                sw_ucb_policies[sat_id],
                slot,
                action,
                normalized_reward,
            )
            debug_info = sw_ucb_policies[sat_id].get_debug_info(slot + 1)
            arm_counts = {
                arm: int(debug_info[arm]["count"]) for arm in cfg.action_set
            }

            sat_record = {
                **reward_record,
                "window_start": max(0, slot + 1 - cfg.tau),
                "window_end": slot,
                "arm_counts": arm_counts,
                "scheduled_group_count": len(scheduled_groups_l),
                "scheduled_edge_count": sum(
                    group.is_edge for group in scheduled_groups_l
                ),
                "scheduled_nonedge_count": sum(
                    not group.is_edge for group in scheduled_groups_l
                ),
                "edge_budget": float(allocation["edge_budget"]),
                "nonedge_budget": float(allocation["nonedge_budget"]),
                "edge_power": float(allocation["edge_power"]),
                "nonedge_power": float(allocation["nonedge_power"]),
                "total_allocated_power": float(
                    allocation["total_allocated_power"]
                ),
                "power_budget": cfg.P_sat,
                "power_floor_infeasible_count": int(
                    allocation["power_floor_infeasible_count"]
                ),
                "negative_power_count": 0,
                "unscheduled_positive_power_count": 0,
                "sic_power_order_violations": int(
                    allocation["sic_power_order_violations"]
                ),
                "sic_decode_violations": 0,
                "handover_event_count": sum(
                    bool(user.get("handover", 0))
                    and user.get("serving_sat") == sat_id
                    for user in users
                ),
                "handover_delay_slots": cfg.handover_delay_slots,
            }
            satellite_records.append(sat_record)
            if validate:
                validation_satellite_records.append(sat_record)
                validation_group_records.extend(
                    _group_validation_record(group, slot) for group in groups_l
                )
            actions_by_satellite.append(action)
            rewards_by_satellite.append(normalized_reward)
            raw_rewards_by_satellite.append(float(reward_details["raw_reward"]))

        success_count = sum(group.success for group in all_scheduled_groups)
        weighted_sum_aoi = compute_weighted_sum_aoi(users, cfg)
        edge_count = sum(group.is_edge for group in all_groups)
        total_groups = len(all_groups)
        scheduled_edge_count = sum(
            group.is_edge for group in all_scheduled_groups
        )
        scheduled_nonedge_count = sum(
            not group.is_edge for group in all_scheduled_groups
        )
        positive_priority_count = sum(group.priority > 0.0 for group in all_groups)
        avg_priority = (
            float(np.mean([group.priority for group in all_groups]))
            if all_groups
            else 0.0
        )
        allocations = [context["allocation"] for context in satellite_contexts]
        edge_power = float(sum(item["edge_power"] for item in allocations))
        nonedge_power = float(sum(item["nonedge_power"] for item in allocations))
        per_satellite_power = [
            float(item["total_allocated_power"]) for item in allocations
        ]
        total_allocated_power = float(sum(per_satellite_power))
        power_floor_infeasible_count = float(
            sum(item["power_floor_infeasible_count"] for item in allocations)
        )
        sic_power_order_violations = float(
            sum(item["sic_power_order_violations"] for item in allocations)
        )
        max_sic_power_order_violation = (
            float(max(item["max_sic_power_order_violation"] for item in allocations))
            if allocations
            else 0.0
        )
        reward = float(np.mean(rewards_by_satellite))
        raw_reward = float(np.mean(raw_rewards_by_satellite))

        histories["action"].append(float(actions_by_satellite[0]))
        histories["actions_by_satellite"].append(actions_by_satellite)
        histories["reward"].append(reward)
        histories["raw_reward"].append(raw_reward)
        histories["rewards_by_satellite"].append(rewards_by_satellite)
        histories["raw_rewards_by_satellite"].append(raw_rewards_by_satellite)
        histories["weighted_sum_aoi"].append(float(weighted_sum_aoi))
        histories["success_count"].append(float(success_count))
        histories["edge_count"].append(float(edge_count))
        histories["total_groups"].append(float(total_groups))
        histories["handover_count"].append(float(handover_count))
        histories["scheduled_edge_count"].append(float(scheduled_edge_count))
        histories["scheduled_nonedge_count"].append(float(scheduled_nonedge_count))
        histories["positive_priority_count"].append(float(positive_priority_count))
        histories["avg_priority"].append(float(avg_priority))
        histories["edge_power"].append(edge_power)
        histories["nonedge_power"].append(nonedge_power)
        histories["total_allocated_power"].append(total_allocated_power)
        histories["per_satellite_total_power"].append(per_satellite_power)
        histories["power_floor_infeasible_count"].append(
            power_floor_infeasible_count
        )
        histories["sic_power_order_violations"].append(
            sic_power_order_violations
        )
        histories["max_sic_power_order_violation"].append(
            max_sic_power_order_violation
        )
        histories["sic_decode_violations"].append(0.0)

        for arm in cfg.action_set:
            counts = [
                policy_state.get_count(slot + 1, arm)
                for policy_state in sw_ucb_policies
            ]
            means = [
                policy_state.get_mean_reward(slot + 1, arm)
                for policy_state in sw_ucb_policies
                if policy_state.get_count(slot + 1, arm) > 0
            ]
            histories[f"arm_{arm}_count"].append(float(sum(counts)))
            histories[f"arm_{arm}_mean_reward"].append(
                float(np.mean(means)) if means else 0.0
            )

        logs.append(
            {
                "t": slot,
                "action": actions_by_satellite[0],
                "actions_by_satellite": actions_by_satellite,
                "reward": reward,
                "raw_reward": raw_reward,
                "rewards_by_satellite": rewards_by_satellite,
                "weighted_sum_aoi": weighted_sum_aoi,
                "success_count": success_count,
                "edge_count": edge_count,
                "total_groups": total_groups,
                "handover_count": handover_count,
                "P_sat": cfg.P_sat,
                "V_phi": cfg.V_phi,
                "V_c": cfg.V_c,
                "V_nc": cfg.V_nc,
                "total_allocated_power": total_allocated_power,
                "allocation_summaries": allocations,
                "satellites": satellite_records,
            }
        )
        last_groups = all_groups

    return {
        "config": cfg,
        "users": users,
        "histories": histories,
        "logs": logs,
        "sw_ucb": sw_ucb_policies[0],
        "sw_ucb_policies": sw_ucb_policies,
        "last_groups": last_groups,
        "validation_satellite_records": validation_satellite_records,
        "validation_group_records": validation_group_records,
        "validation_violations": validation_violations,
    }


def run_mira_leo_v2a3(*args, **kwargs) -> dict[str, object]:
    """Compatibility alias for callers using the source-version runner name."""
    return run_mira_leo_v2a3_verified(*args, **kwargs)


def run_algorithm1(*args, **kwargs) -> dict[str, object]:
    """Backward-compatible alias for the complete verified runner."""
    return run_mira_leo_v2a3_verified(*args, **kwargs)


def _print_summary(result: dict[str, object]) -> None:
    cfg = result["config"]
    histories = result["histories"]
    assert isinstance(cfg, V2Config)
    assert isinstance(histories, dict)

    print("MIRA-LEO v2a3 verified Algorithms 1 + 2 + 3 finished.")
    print(f"T={cfg.T}, users={cfg.N_total}, action_set={list(cfg.action_set)}")
    print(f"P_sat={cfg.P_sat}, V_phi={cfg.V_phi}, V_c={cfg.V_c}, V_nc={cfg.V_nc}")
    print(
        f"slot_time_s={cfg.slot_time_s}, handover_delay_slots="
        f"{cfg.handover_delay_slots}, isl_delay_slots={cfg.isl_delay_slots}"
    )
    print(
        "reward normalization=fixed linear implementation choice, "
        f"bounds={cfg.reward_raw_bounds}"
    )
    print(f"final actions={histories['actions_by_satellite'][-1]}")
    print(f"final reward={histories['reward'][-1]:.6g}")
    print(f"final weighted-sum AoI={histories['weighted_sum_aoi'][-1]:.6g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run validated MIRA-LEO v2a3 Algorithms 1, 2, and 3."
    )
    parser.add_argument("--T", type=int, default=None, help="number of time slots")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--policy", choices=["sw-ucb", "fixed", "random"], default="sw-ucb"
    )
    parser.add_argument("--fixed-action", type=int, default=4)
    parser.add_argument("--P-sat", type=float, default=None)
    parser.add_argument("--V-c", type=float, default=None)
    parser.add_argument("--V-nc", type=float, default=None)
    parser.add_argument("--V-phi", type=float, default=None)
    parser.add_argument("--slot-time-ms", type=float, default=None)
    parser.add_argument("--handover-delay-ms", type=float, default=None)
    parser.add_argument("--isl-delay-ms", type=float, default=None)
    parser.add_argument("--no-fading", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--strict-validation", action="store_true")
    parser.add_argument("--validation-output", default=None)
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "results" / "figures"),
    )
    return parser.parse_args()


def _apply_cli_overrides(cfg: V2Config, args: argparse.Namespace) -> V2Config:
    overrides = {}
    for argument, field_name in [
        ("T", "T"),
        ("seed", "random_seed"),
        ("P_sat", "P_sat"),
        ("V_c", "V_c"),
        ("V_nc", "V_nc"),
        ("V_phi", "V_phi"),
    ]:
        value = getattr(args, argument, None)
        if value is not None:
            overrides[field_name] = value
    for argument, field_name in [
        ("slot_time_ms", "slot_time_s"),
        ("handover_delay_ms", "handover_delay_s"),
        ("isl_delay_ms", "isl_delay_s"),
    ]:
        value = getattr(args, argument, None)
        if value is not None:
            overrides[field_name] = float(value) / 1000.0
    return replace(cfg, **overrides) if overrides else cfg


def main() -> int:
    args = parse_args()
    cfg = _apply_cli_overrides(V2Config(), args)
    result = run_mira_leo_v2a3_verified(
        cfg,
        policy=args.policy,
        fixed_action=args.fixed_action,
        fading=not args.no_fading,
        validate=args.validate or bool(args.validation_output),
        strict_validation=args.strict_validation,
    )

    if args.validation_output:
        from mira_leo_v2a3_verified.validation.run_validation import (
            write_runtime_outputs,
        )

        write_runtime_outputs(result, Path(args.validation_output))

    if args.save_plots:
        from mira_leo_v2a3_verified.plot_results import plot_histories

        saved_paths = plot_histories(result["histories"], args.output_dir)
        if not args.quiet:
            for path in saved_paths:
                print(f"saved {path}")
    if not args.quiet:
        _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
