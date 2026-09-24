#!/usr/bin/env python3
"""Run the config-driven Walker perturbation suite.

    python -m loka.run_walker_suite
    python -m loka.run_walker_suite --config loka/config/walker_suite.yaml \\
        --tests ice,backpack --baselines loka,fixed_mpc,dr_rl --num-trials 5
"""

from __future__ import annotations

import argparse
import sys

from loka.walker_suite.config import DEFAULT_CONFIG_PATH, load_suite_config
from loka.walker_suite.pipeline import run_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Walker LOKA vs fixed-MJPC vs DR-RL perturbation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"YAML suite config (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--tests",
        default=None,
        help="Comma-separated test names to run (default: all in the config).",
    )
    parser.add_argument(
        "--baselines",
        default=None,
        help="Comma-separated baselines: loka, fixed_mpc, dr_rl.",
    )
    parser.add_argument("--goal-distance", type=float, default=None)
    parser.add_argument("--perturbation-time", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--speed-goal", type=float, default=None)
    parser.add_argument(
        "--num-trials",
        type=int,
        default=None,
        help="Trials per test and baseline (default: num_trials in the config, usually 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base RNG seed. Trial k uses seed+k for every baseline.",
    )
    parser.add_argument(
        "--init-noise",
        type=float,
        default=None,
        help="Uniform half-width on qpos and qvel at reset. 0 disables it.",
    )
    parser.add_argument(
        "--dr-rl-checkpoint",
        default=None,
        help="DR-RL policy zip or directory (overrides dr_rl_checkpoint in the suite YAML).",
    )
    parser.add_argument(
        "--dr-rl-config",
        default=None,
        help="DR-RL YAML for that checkpoint (overrides dr_rl_config in the suite YAML).",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Skip episode videos (time series are still written).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_suite_config(args)
    except Exception as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2
    run_dir = run_pipeline(config)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
