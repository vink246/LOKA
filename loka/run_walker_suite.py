#!/usr/bin/env python3
"""Run the config-driven Walker perturbation suite.

    python -m loka.run_walker_suite
    python -m loka.run_walker_suite --config loka/config/walker_suite.yaml \\
        --tests ice,backpack --baselines fixed_mpc --timeout 12
"""

from __future__ import annotations

import argparse
import sys

from loka.walker_suite.config import DEFAULT_CONFIG_PATH, load_suite_config
from loka.walker_suite.pipeline import run_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Walker LOKA vs fixed-MJPC perturbation pipeline.",
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
        help="Comma-separated baselines: loka, fixed_mpc.",
    )
    parser.add_argument("--goal-distance", type=float, default=None)
    parser.add_argument("--perturbation-time", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--speed-goal", type=float, default=None)
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
