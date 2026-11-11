from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config, validate_fixed_study_config
from .errors import ArtifactReadError
from .geometry_export import export_geometry
from .orbit_visualization import generate_orbit_visuals
from .paper_results import run_paper_results
from .reward_pilot import run_reward_bounds_pilot
from .simulation import run_simulation

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "base.yaml"


def _common_config(parser):
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--outputs", default="outputs/120ue_single_serving_full_load_interference")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mira_leo_study",
        description="phi=15 main, MIRA 13/15/20/25 P1 ablation, CSI-expanded protected power pools",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-config", help="validate fixed parameters without simulation")
    validate.add_argument("--config", default=str(DEFAULT_CONFIG))
    p1 = commands.add_parser("p1-ablation", help="exactly six runs: MIRA(13,15,20,25), Random(15), Fixed a=4(15)",
                             description="Exactly six runs: MIRA(13,15,20,25 degrees), Random(15), Fixed a=4(15). 48000 slots/run, seeds layout42/subscription1042/channel0/policy0; one CSI-expanded 70/30 allocator")
    p1.add_argument("--config", default=str(DEFAULT_CONFIG))
    p1.add_argument("--outputs", default="outputs/p1_ablation_13_15_20_25")
    p1.add_argument("--debug-slots", type=int, default=None, help="1..47999, separate debug directory and short pilot bounds")
    p1.add_argument("--seed", type=int, choices=[0], default=0)
    p1.add_argument("--resume", action="store_true")
    p1.add_argument("--save-group-trace", action="store_true", help="optional full enriched group traces for all six runs")
    geometry = commands.add_parser("geometry", help="export synthetic geometry and association")
    _common_config(geometry)
    orbit = commands.add_parser("orbit-visuals", help="render synthetic constellation orbit visuals")
    _common_config(orbit)
    orbit.add_argument("--duration-s", type=float, default=None)
    orbit.add_argument("--frames", type=int, default=121)
    orbit.add_argument("--fps", type=int, default=20)
    orbit.add_argument("--video-format", choices=("auto", "gif", "mp4"), default="auto")
    orbit.add_argument("--dpi", type=int, default=180)
    pilot = commands.add_parser("calibrate-reward-bounds", help="pilot normalization bounds at fixed phi/B only")
    _common_config(pilot)
    pilot.add_argument("--seed", type=int, default=None)
    paper = commands.add_parser("paper-results", help="run MIRA / Random / Fixed AoI and regrouping overhead")
    _common_config(paper)
    paper.add_argument("--slots", type=int, default=None, help="debug truncation; omit for full 480 seconds")
    paper.add_argument("--seed", type=int, default=None)
    paper.add_argument("--reward-bounds", default=None, help="frozen reward_bounds.json from the fixed pilot")
    paper.add_argument("--tau-s", nargs="+", type=float, default=None, help="overhead sensitivity windows; nominal 10 is always included")
    paper.add_argument("--resume", action="store_true")
    paper.add_argument("--save-group-trace", action="store_true")
    paper.add_argument("--diagnostics", action="store_true", help="save enriched group traces and diagnose after the comparison")
    diagnose = commands.add_parser("diagnose-results", help="analyze saved traces without running any simulation")
    _common_config(diagnose)
    diagnose.add_argument("--seed", type=int, default=0)
    diagnose.add_argument("--slots", type=int, default=None, help="select debug/slots_N_seed_SEED when several debug runs exist")
    smoke = commands.add_parser("smoke", help="explicitly run a short fixed-a=4 simulation")
    smoke.add_argument("--config", default=str(DEFAULT_CONFIG))
    smoke.add_argument("--slots", type=int, default=3)
    smoke.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "validate-config":
        validate_fixed_study_config(config)
        result = {"status": "VALID", "total_ue": 120, "message_count": 3, "subscribers_per_message": 40, "serving_satellite_count": 1, "interference_mode": "fixed_full_load", "interferer_power_normalized": 1.0, "phi_theta_deg": config.get("algorithm.phi_theta_deg"), "B_penalty": 0.05,
                  "action_set": config.get("algorithm.action_set"),
                  "regrouping_interval_slots": config.get("derived.regrouping_interval_slots"),
                  "sw_ucb_window_epochs": config.get("derived.sw_ucb_window_epochs"),
                  "scheduler_algorithm": "Algorithm2_strict", "power_allocator": config.get("algorithm.power_mode"),
                  "infeasible_power_handling": config.get("algorithm.infeasible_power_handling"),
                  "reward_bounds_available": all(config.get("algorithm.reward_bounds." + k) is not None for k in ("lower", "upper", "identity"))}
    elif args.command == "p1-ablation":
        from .p1_ablation import run_p1_ablation
        result = run_p1_ablation(config,args.outputs,debug_slots=args.debug_slots,seed=args.seed,
                                 resume=args.resume,save_group_trace=args.save_group_trace)
    elif args.command == "geometry":
        result = export_geometry(config, args.outputs)
    elif args.command == "orbit-visuals":
        result = generate_orbit_visuals(config, args.outputs, duration_s=args.duration_s,
                                        frames=args.frames, fps=args.fps,
                                        video_format=args.video_format, dpi=args.dpi)
    elif args.command == "calibrate-reward-bounds":
        seed = int(config.get("randomness.fixed_channel_seed", 0) if args.seed is None else args.seed)
        config = config.with_overrides({"randomness.fixed_channel_seed": seed})
        result = run_reward_bounds_pilot(config, args.outputs, seed=seed)["calibration"]
    elif args.command == "paper-results":
        try:
            result = run_paper_results(config, args.outputs, slots=args.slots, seed=args.seed,
                                       reward_bounds_path=args.reward_bounds, resume=args.resume,
                                       tau_s=args.tau_s, save_group_trace=args.save_group_trace,
                                       diagnostics=args.diagnostics)
        except ArtifactReadError as exc:
            build_parser().error(str(exc))
    elif args.command == "diagnose-results":
        from .diagnostics import DiagnosticDataError, generate_diagnostics
        try:
            result = generate_diagnostics(config, args.outputs, seed=args.seed, slots=args.slots)
        except DiagnosticDataError as exc:
            parser = build_parser()
            parser.error(str(exc))
    elif args.command == "smoke":
        validate_fixed_study_config(config)
        result = run_simulation(config, channel_seed=args.seed, policy="fixed_grouping",
                                fixed_action=4, max_slots=args.slots)["summary"]
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
