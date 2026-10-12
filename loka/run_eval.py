#!/usr/bin/env python3
"""Run the G1 evaluation suite, or recompute metrics from a finished run.

    python -m loka.run_eval
    python -m loka.run_eval --tasks goto_goal --baselines low_level --num-trials 3
    python -m loka.run_eval metrics results/g1_eval/<stamp> --metrics success,fell
    python -m loka.run_eval render results/g1_eval/<stamp> --episode <unit dir>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from loka.harness.cli import add_suite_arguments
from loka.harness.config import load_suite_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="G1 evaluation harness (low-level controller vs LOKA).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Roll out the suite.")
    add_suite_arguments(run)
    metrics = sub.add_parser("metrics", help="Recompute metrics, statistics and plots.")
    metrics.add_argument("run_dir")
    metrics.add_argument(
        "--metrics",
        default=None,
        help="Comma-separated metric names. Default: each task's own list.",
    )
    render = sub.add_parser("render", help="Encode one stored episode to mp4.")
    render.add_argument("run_dir")
    render.add_argument("--episode", required=True, help="Episode directory with render.npz.")
    render.add_argument(
        "--mujoco-gl",
        default=None,
        help="MUJOCO_GL backend for the render process (egl or osmesa).",
    )
    return parser


def _normalize(argv: list[str]) -> list[str]:
    if not argv or argv[0] not in {"run", "metrics", "render"}:
        return ["run", *argv]
    return argv


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(_normalize(raw))
    if args.command == "metrics":
        from loka.harness.pipeline import recompute

        names = None
        if args.metrics:
            names = [part.strip() for part in args.metrics.split(",") if part.strip()]
        missing = recompute(Path(args.run_dir), names)
        return 1 if missing else 0
    if args.command == "render":
        return _render(args)
    try:
        suite = load_suite_config(args)
    except Exception as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2
    from loka.harness.pipeline import run_suite

    run_dir = run_suite(suite)
    print(run_dir)
    return 0


def _render(args) -> int:
    from loka.harness.baselines import locomotion_config
    from loka.harness.config import load_suite_yaml
    from loka.harness.render import render_episode_dir

    if args.mujoco_gl:
        os.environ["MUJOCO_GL"] = args.mujoco_gl
    suite = load_suite_yaml(Path(args.run_dir) / "config.resolved.yaml")
    path = render_episode_dir(
        args.episode,
        recording=suite.recording,
        model_path=str(locomotion_config(suite).model_path),
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
