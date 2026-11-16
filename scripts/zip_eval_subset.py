#!/usr/bin/env python3
"""Zip a filtered subset of a finished (or in-progress) eval run for download.

Unit directories are named ``task__perturbation__baseline__trial_XX``. Pass
``--tasks``, ``--baselines`` and ``--perturbations`` as comma-separated lists
to keep only matching units. Always includes the run's top-level summary
files (``config.resolved.yaml``, ``summary.*``, ``statistics.*``) when present.
By default, matching plot files under ``plots/`` whose names contain a
selected task are included too.

    # Stairs, curb and navigation only (all baselines and trials):
    python scripts/zip_eval_subset.py results/g1_master/pace_YYYYMMDD_HHMMSS \\
        --tasks curb,stairs_regular,nav_object_goal,nav_route \\
        --out ~/scratch/nav_terrain.zip

    # All LOKA baselines (loka, loka_vision, loka_claude_*, …) of those tasks,
    # skipping heavy camera frames / render buffers for a smaller download:
    python scripts/zip_eval_subset.py results/g1_master/pace_YYYYMMDD_HHMMSS \\
        --tasks curb,stairs_regular,nav_object_goal,nav_route \\
        --baselines 'loka*' --exclude-frames --exclude-render \\
        --out ~/scratch/loka_nav_terrain.zip

    # Dry run: list what would be packed.
    python scripts/zip_eval_subset.py results/g1_master/pace_YYYYMMDD_HHMMSS \\
        --tasks curb,stairs_regular --dry-run
"""

from __future__ import annotations

import argparse
import fnmatch
import zipfile
from pathlib import Path

UNIT_SEP = "__"
TRIAL_PREFIX = "trial_"
SUMMARY_NAMES = (
    "config.resolved.yaml",
    "summary.json",
    "summary.csv",
    "statistics.json",
    "statistics.csv",
)
# Skip Windows Zone.Identifier sidecars and Python caches by default.
SKIP_GLOBS = ("*:Zone.Identifier", "*/__pycache__/*", "*.pyc")


def _split_csv(text: str | None) -> list[str] | None:
    if text is None:
        return None
    names = [part.strip() for part in str(text).split(",") if part.strip()]
    return names or None


def parse_unit_id(name: str) -> tuple[str, str, str, str] | None:
    """``task, perturbation, baseline, trial`` from a unit directory name."""
    if UNIT_SEP not in name or TRIAL_PREFIX not in name:
        return None
    head, trial = name.rsplit(UNIT_SEP, 1)
    if not trial.startswith(TRIAL_PREFIX):
        return None
    parts = head.split(UNIT_SEP)
    if len(parts) < 3:
        return None
    task = parts[0]
    perturbation = parts[1]
    baseline = UNIT_SEP.join(parts[2:])
    return task, perturbation, baseline, trial


def _matches(value: str, allowed: list[str] | None) -> bool:
    """Exact match, or shell-style glob when a filter contains ``*`` / ``?`` / ``[``."""
    if allowed is None:
        return True
    for pattern in allowed:
        if any(ch in pattern for ch in "*?["):
            if fnmatch.fnmatch(value, pattern):
                return True
        elif value == pattern:
            return True
    return False


def select_units(
    run_dir: Path,
    *,
    tasks: list[str] | None,
    baselines: list[str] | None,
    perturbations: list[str] | None,
) -> list[Path]:
    selected = []
    for path in sorted(run_dir.iterdir()):
        if not path.is_dir():
            continue
        parsed = parse_unit_id(path.name)
        if parsed is None:
            continue
        task, perturbation, baseline, _trial = parsed
        if (
            _matches(task, tasks)
            and _matches(baseline, baselines)
            and _matches(perturbation, perturbations)
        ):
            selected.append(path)
    return selected


def select_plots(run_dir: Path, tasks: list[str] | None) -> list[Path]:
    plots = run_dir / "plots"
    if not plots.is_dir():
        return []
    chosen = []
    for path in plots.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(run_dir).as_posix()
        if any(fnmatch.fnmatch(rel, pat) for pat in SKIP_GLOBS):
            continue
        if tasks is None or any(task in path.name for task in tasks):
            chosen.append(path)
    return chosen


def iter_files(
    root: Path,
    *,
    exclude_frames: bool,
    exclude_render: bool,
    exclude_videos: bool,
) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(rel, pat) for pat in SKIP_GLOBS):
            continue
        if exclude_frames and ("/frames/" in f"/{rel}" or rel.startswith("frames/")):
            continue
        if exclude_render and path.name == "render.npz":
            continue
        if exclude_videos and path.suffix.lower() in {".mp4", ".webm", ".avi"}:
            continue
        files.append(path)
    return files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Zip a filtered subset of an eval run for download from PACE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Eval run directory (e.g. results/g1_master/pace_YYYYMMDD_HHMMSS).",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task names (e.g. curb,stairs_regular,nav_object_goal,nav_route).",
    )
    parser.add_argument(
        "--baselines",
        default=None,
        help="Comma-separated baseline names (e.g. loka_vision).",
    )
    parser.add_argument(
        "--perturbations",
        default=None,
        help="Comma-separated perturbation names (e.g. nominal).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output zip path (default: <run_dir_name>_<filters>.zip next to the run).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not include matching files under plots/.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not include top-level summary / config files.",
    )
    parser.add_argument(
        "--exclude-frames",
        action="store_true",
        help="Skip camera frame PNGs (frames/ and navila/frames/).",
    )
    parser.add_argument(
        "--exclude-render",
        action="store_true",
        help="Skip render.npz buffers (large; needed only to re-encode videos).",
    )
    parser.add_argument(
        "--exclude-videos",
        action="store_true",
        help="Skip .mp4 / .webm / .avi files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the matching units and file count; write nothing.",
    )
    return parser


def default_out(run_dir: Path, tasks, baselines, perturbations) -> Path:
    parts = [run_dir.name]
    for label, values in (
        ("tasks", tasks),
        ("baselines", baselines),
        ("pert", perturbations),
    ):
        if values:
            parts.append(f"{label}-" + "+".join(values))
    return run_dir.parent / ("_".join(parts) + ".zip")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"run directory not found: {run_dir}")
        return 2
    tasks = _split_csv(args.tasks)
    baselines = _split_csv(args.baselines)
    perturbations = _split_csv(args.perturbations)
    units = select_units(
        run_dir, tasks=tasks, baselines=baselines, perturbations=perturbations
    )
    if not units:
        print(
            f"no units matched in {run_dir} "
            f"(tasks={tasks}, baselines={baselines}, perturbations={perturbations})"
        )
        return 1

    members: list[tuple[Path, str]] = []
    if not args.no_summary:
        for name in SUMMARY_NAMES:
            path = run_dir / name
            if path.is_file():
                members.append((path, name))
    if not args.no_plots:
        for path in select_plots(run_dir, tasks):
            members.append((path, path.relative_to(run_dir).as_posix()))

    unit_files = 0
    for unit in units:
        for path in iter_files(
            unit,
            exclude_frames=args.exclude_frames,
            exclude_render=args.exclude_render,
            exclude_videos=args.exclude_videos,
        ):
            members.append((path, path.relative_to(run_dir).as_posix()))
            unit_files += 1

    out = (
        args.out.expanduser().resolve()
        if args.out is not None
        else default_out(run_dir, tasks, baselines, perturbations)
    )
    print(f"run:          {run_dir}")
    print(f"units:        {len(units)}")
    for unit in units:
        print(f"  {unit.name}")
    print(f"files:        {len(members)} ({unit_files} inside units)")
    print(f"zip:          {out}")
    if args.dry_run:
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, arcname in members:
            archive.write(path, arcname)
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"wrote {out} ({size_mb:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
