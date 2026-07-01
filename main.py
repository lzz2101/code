from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from .aoi import update_user_aoi, weighted_sum_aoi
from .channel import compute_channel_gain
from .config import SimConfig
from .geometry import (
    assign_serving_satellite,
    compute_distance_and_elevation,
    initialize_users,
    update_satellite_positions,
)
from .grouping import build_groups_for_satellite, classify_edge_groups
from .noma import allocate_power, compute_noma_success
from .scheduler import schedule_top_aoi_groups


def empty_histories() -> dict[str, list[float]]:
    return {
        "weighted_sum_aoi": [],
        "success_count": [],
        "handover_count": [],
        "edge_group_count": [],
        "scheduled_group_count": [],
        "total_group_count": [],
        "average_rate": [],
        "covered_user_count": [],
    }


def run_simulation(
    cfg: SimConfig | None = None, fading: bool | None = None
) -> dict[str, object]:
    """Run the Version 1 simulator and return histories without writing files."""
    cfg = cfg or SimConfig()
    fading = cfg.fading_enabled if fading is None else fading
    rng = np.random.default_rng(cfg.random_seed)
    users = initialize_users(cfg, rng)
    histories = empty_histories()
    last_groups: list[dict] = []

    for t in range(cfg.T):
        sat_positions = update_satellite_positions(t, cfg)
        distance, elevation, _ = compute_distance_and_elevation(users, sat_positions, cfg)
        handover_count = assign_serving_satellite(users, elevation, cfg)
        channel_gain = compute_channel_gain(distance, cfg, rng, fading=fading)

        all_groups: list[dict] = []
        for sat_id in range(cfg.L):
            groups = build_groups_for_satellite(
                users, sat_id, channel_gain, elevation, cfg
            )
            classify_edge_groups(groups, elevation, cfg)
            scheduled_groups = schedule_top_aoi_groups(groups, cfg)
            allocate_power(scheduled_groups, cfg, mode=cfg.power_mode)
            compute_noma_success(scheduled_groups, cfg)
            all_groups.extend(groups)

        update_user_aoi(users, all_groups, distance, cfg)

        scheduled_groups = [group for group in all_groups if group["scheduled"]]
        rates = [float(group["rate"]) for group in scheduled_groups]

        histories["weighted_sum_aoi"].append(weighted_sum_aoi(users, cfg))
        histories["success_count"].append(
            float(sum(1 for group in all_groups if group["success"]))
        )
        histories["handover_count"].append(float(handover_count))
        histories["edge_group_count"].append(
            float(sum(1 for group in all_groups if group["is_edge"]))
        )
        histories["scheduled_group_count"].append(float(len(scheduled_groups)))
        histories["total_group_count"].append(float(len(all_groups)))
        histories["average_rate"].append(float(np.mean(rates)) if rates else 0.0)
        histories["covered_user_count"].append(
            float(sum(1 for user in users if user["serving_sat"] is not None))
        )

        last_groups = all_groups

    return {
        "config": cfg,
        "users": users,
        "histories": histories,
        "last_groups": last_groups,
    }


def _print_summary(result: dict[str, object]) -> None:
    cfg = result["config"]
    histories = result["histories"]
    assert isinstance(cfg, SimConfig)
    assert isinstance(histories, dict)

    print("MIRA-LEO Version 1 simulation finished.")
    print(f"T={cfg.T}, users={cfg.N_total}, satellites={cfg.L}, messages={cfg.M}")
    print(f"final weighted-sum AoI={histories['weighted_sum_aoi'][-1]:.6g}")
    print(f"total handovers={sum(histories['handover_count']):.0f}")
    print(f"max scheduled groups/slot={max(histories['scheduled_group_count']):.0f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MIRA-LEO Version 1 simulator.")
    parser.add_argument("--T", type=int, default=None, help="number of time slots")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--power-mode",
        choices=["equal", "ordered"],
        default=None,
        help="power allocation mode",
    )
    parser.add_argument("--no-fading", action="store_true", help="disable Rayleigh fading")
    parser.add_argument("--quiet", action="store_true", help="suppress summary printing")
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="save diagnostic figures under the output directory",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "results" / "figures"),
        help="directory used when --save-plots is enabled",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = SimConfig()
    if args.T is not None:
        cfg = replace(cfg, T=args.T)
    if args.seed is not None:
        cfg = replace(cfg, random_seed=args.seed)
    if args.power_mode is not None:
        cfg = replace(cfg, power_mode=args.power_mode)

    result = run_simulation(cfg, fading=not args.no_fading)

    if args.save_plots:
        from .plot_results import plot_histories

        saved_paths = plot_histories(result["histories"], args.output_dir)
        if not args.quiet:
            for path in saved_paths:
                print(f"saved {path}")

    if not args.quiet:
        _print_summary(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
