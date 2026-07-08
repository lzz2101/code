from __future__ import annotations

import argparse
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
from mira_leo_v2.algorithms.power_allocation import power_allocation_placeholder
from mira_leo_v2.algorithms.regrouping import (
    classify_edge_groups,
    make_equal_cardinality_groups,
)
from mira_leo_v2.algorithms.scheduler import scheduler_placeholder
from mira_leo_v2.algorithms.sw_ucb import SWUCB
from mira_leo_v2.aoi.metrics import (
    compute_regrouping_reward,
    compute_weighted_sum_aoi,
    update_user_aoi,
)
from mira_leo_v2.config import V2Config
from mira_leo_v2.noma.transmission import transmit_noma


def empty_histories(action_set: tuple[int, ...]) -> dict[str, list[float]]:
    histories = {
        "action": [],
        "reward": [],
        "weighted_sum_aoi": [],
        "success_count": [],
        "edge_count": [],
        "total_groups": [],
        "handover_count": [],
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
    t: int,
    sw_ucb: SWUCB,
    rng: np.random.Generator,
    cfg: V2Config,
    fixed_action: int,
) -> int:
    if policy == "sw-ucb":
        return sw_ucb.select_arm(t)
    if policy == "fixed":
        return int(fixed_action)
    if policy == "random":
        return int(rng.choice(cfg.action_set))
    raise ValueError(f"unknown policy: {policy}")


def update_policy(policy: str, sw_ucb: SWUCB, action: int, reward: float) -> None:
    if policy == "sw-ucb":
        sw_ucb.update(action, reward)


def run_algorithm1(
    cfg: V2Config | None = None,
    policy: str = "sw-ucb",
    fixed_action: int = 4,
    fading: bool | None = None,
) -> dict[str, object]:
    """Run Algorithm 1 without writing result files."""
    cfg = cfg or V2Config()
    fading = cfg.fading_enabled if fading is None else fading
    rng = np.random.default_rng(cfg.random_seed)
    users = initialize_users(cfg, rng)
    sw_ucb = SWUCB(cfg.action_set, cfg.tau, cfg.xi)
    histories = empty_histories(cfg.action_set)
    logs: list[dict] = []
    last_groups = []

    for t in range(cfg.T):
        sat_positions = update_satellite_positions(t, cfg)
        distance, elevation, _ = compute_distance_and_elevation(users, sat_positions, cfg)
        handover_count = assign_serving_satellite(users, elevation, cfg)
        channel_gain = compute_channel_gain(distance, cfg, rng, fading=fading)
        attach_slot_observations(users, channel_gain, elevation, cfg)

        action = select_action(policy, t, sw_ucb, rng, cfg, fixed_action)
        all_groups = []
        all_scheduled_groups = []

        for sat_id in range(cfg.L):
            groups_l = make_equal_cardinality_groups(
                users=users,
                satellite_id=sat_id,
                action_a=action,
                num_messages=cfg.M,
            )
            classify_edge_groups(groups_l, cfg.phi_theta)
            scheduled_groups_l = scheduler_placeholder(
                groups_l, cfg.max_scheduled_groups
            )
            power_allocation_placeholder(scheduled_groups_l, cfg.P_sat)
            transmit_noma(
                scheduled_groups_l,
                noise_power=cfg.noise_power,
                rate_threshold=cfg.R_threshold,
            )

            all_groups.extend(groups_l)
            all_scheduled_groups.extend(scheduled_groups_l)

        update_user_aoi(users, all_groups, distance, cfg)

        success_count = sum(1 for group in all_scheduled_groups if group.success)
        weighted_sum_aoi = compute_weighted_sum_aoi(users, cfg)
        edge_count = sum(1 for group in all_groups if group.is_edge)
        total_groups = len(all_groups)
        reward = compute_regrouping_reward(
            success_count=success_count,
            weighted_sum_aoi=weighted_sum_aoi,
            edge_count=edge_count,
            total_groups=total_groups,
            num_users=len(users),
            B=cfg.B_edge,
            normalize=cfg.normalize_reward,
            aoi_norm_factor=cfg.aoi_norm_factor,
        )

        update_policy(policy, sw_ucb, action, reward)
        debug_info = sw_ucb.get_debug_info()

        histories["action"].append(float(action))
        histories["reward"].append(float(reward))
        histories["weighted_sum_aoi"].append(float(weighted_sum_aoi))
        histories["success_count"].append(float(success_count))
        histories["edge_count"].append(float(edge_count))
        histories["total_groups"].append(float(total_groups))
        histories["handover_count"].append(float(handover_count))

        log_record = {
            "t": t,
            "action": action,
            "reward": reward,
            "weighted_sum_aoi": weighted_sum_aoi,
            "success_count": success_count,
            "edge_count": edge_count,
            "total_groups": total_groups,
            "handover_count": handover_count,
        }
        for arm in cfg.action_set:
            count = debug_info[arm]["count"]
            mean_reward = debug_info[arm]["mean_reward"]
            histories[f"arm_{arm}_count"].append(float(count))
            histories[f"arm_{arm}_mean_reward"].append(float(mean_reward))
            log_record[f"arm_{arm}_count"] = count
            log_record[f"arm_{arm}_mean_reward"] = mean_reward
        logs.append(log_record)
        last_groups = all_groups

    return {
        "config": cfg,
        "users": users,
        "histories": histories,
        "logs": logs,
        "sw_ucb": sw_ucb,
        "last_groups": last_groups,
    }


def _print_summary(result: dict[str, object]) -> None:
    cfg = result["config"]
    histories = result["histories"]
    assert isinstance(cfg, V2Config)
    assert isinstance(histories, dict)

    print("MIRA-LEO Version 2 Algorithm 1 finished.")
    print(f"T={cfg.T}, users={cfg.N_total}, action_set={list(cfg.action_set)}")
    print(f"final action={histories['action'][-1]:.0f}")
    print(f"final reward={histories['reward'][-1]:.6g}")
    print(f"final weighted-sum AoI={histories['weighted_sum_aoi'][-1]:.6g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MIRA-LEO Version 2 Algorithm 1 SW-UCB regrouping."
    )
    parser.add_argument("--T", type=int, default=None, help="number of time slots")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--policy",
        choices=["sw-ucb", "fixed", "random"],
        default="sw-ucb",
        help="regrouping policy",
    )
    parser.add_argument("--fixed-action", type=int, default=4, help="fixed group count")
    parser.add_argument("--no-fading", action="store_true", help="disable fading")
    parser.add_argument("--quiet", action="store_true", help="suppress summary")
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="save diagnostic figures under output directory",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "results" / "figures"),
        help="directory used when --save-plots is enabled",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = V2Config()
    if args.T is not None:
        cfg = replace(cfg, T=args.T)
    if args.seed is not None:
        cfg = replace(cfg, random_seed=args.seed)

    result = run_algorithm1(
        cfg,
        policy=args.policy,
        fixed_action=args.fixed_action,
        fading=not args.no_fading,
    )

    if args.save_plots:
        from mira_leo_v2.plot_results import plot_histories

        saved_paths = plot_histories(result["histories"], args.output_dir)
        if not args.quiet:
            for path in saved_paths:
                print(f"saved {path}")

    if not args.quiet:
        _print_summary(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
