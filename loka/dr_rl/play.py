"""Roll out the frozen DR-RL policy on the clean Walker plant (no faults).

    python -m loka.dr_rl.play
    python -m loka.dr_rl.play --checkpoint results/dr_rl/best
    python -m loka.dr_rl.play --checkpoint results/dr_rl/checkpoints/ppo_walker_500000_steps.zip
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from loka.dr_rl.config import LOKA_ROOT
from loka.walker_suite.config import EpisodeDefaults, SuiteConfig, TestCase
from loka.walker_suite.episode import run_episode


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a nominal (unperturbed) DR-RL Walker rollout.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "DR-RL YAML. Use loka/dr_rl/config_gym.yaml for the Gym-style "
            "harness (23-D obs, no gait clock). Default: config.yaml."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Policy zip or directory. Gait harness: results/dr_rl/. "
            "Gym harness: results/dr_rl_gym/. VecNormalize stats are loaded "
            "from the same folder."
        ),
    )
    parser.add_argument("--goal-distance", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--speed-goal", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for video/timeseries (default: results/dr_rl/play/<stamp>).",
    )
    parser.add_argument("--no-record", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import os

    from loka.dr_rl.config import checkpoint_dir, load_dr_config

    if args.config:
        os.environ["LOKA_DR_RL_CONFIG"] = str(Path(args.config).resolve())
    cfg = load_dr_config(args.config)
    if args.checkpoint:
        ckpt = Path(args.checkpoint).resolve()
        os.environ["LOKA_DR_RL_CHECKPOINT"] = str(ckpt)
        print(f"[dr_rl] checkpoint {ckpt}")
    print(f"[dr_rl] config {cfg.get('_config_path')}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_play = checkpoint_dir(cfg) / "play" / stamp
    out_root = Path(args.output_dir) if args.output_dir else default_play
    out_root = out_root if out_root.is_absolute() else LOKA_ROOT / out_root
    episode_dir = out_root / "nominal__dr_rl"

    suite = SuiteConfig(
        output_dir=out_root,
        record=not args.no_record,
        defaults=EpisodeDefaults(
            goal_distance_m=float(args.goal_distance),
            perturbation_time_s=1e9,
            timeout_s=float(args.timeout),
            speed_goal=float(args.speed_goal),
        ),
        baselines=["dr_rl"],
        tests=[],
        init_noise=0.0,
    )
    test = TestCase(name="nominal", perturbation={"kind": "none"})
    print(f"[dr_rl] nominal rollout → {episode_dir}")
    metadata = run_episode(test, "dr_rl", suite, episode_dir)
    print(
        f"[dr_rl] {metadata['outcome']} t={metadata['t_end']:.2f}s "
        f"x={metadata['pos_x_final']:.2f}m"
    )
    video = episode_dir / "episode.mp4"
    if video.is_file():
        print(f"[dr_rl] video {video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
