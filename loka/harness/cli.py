"""CLI flags shared by ``run_eval`` and the PACE-ICE submitter."""

from __future__ import annotations

import argparse

from loka.harness.config import DEFAULT_CONFIG_PATH


def add_suite_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"YAML suite config (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument("--tasks", default=None, help="Comma-separated task names.")
    parser.add_argument(
        "--baselines",
        default=None,
        help="Comma-separated baseline names (low_level, loka, or a registered type).",
    )
    parser.add_argument("--num-trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--init-noise", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None, help="Local process pool size.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Write into this directory instead of a new timestamp folder.",
    )
    parser.add_argument(
        "--trial",
        type=int,
        default=None,
        help="Run only this 0-based trial (seed + trial).",
    )
    parser.add_argument(
        "--unit",
        default=None,
        help="Run only this unit id (task__perturbation__baseline__trial_XX).",
    )
    parser.add_argument("--record", action="store_true", help="Encode episode videos.")
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Do not encode videos. render.npz is still written.",
    )
