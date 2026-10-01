"""Run every configured Walker test against each baseline."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path

import yaml

from loka.walker_suite.config import SuiteConfig, load_suite_yaml
from loka.walker_suite.logging import _json_ready
from loka.walker_suite.plots import plot_run
from loka.walker_suite.statistics import (
    format_condition_line,
    summarize_run,
    write_statistics,
)
from loka.walker_suite.stochastic import trial_seed

SUMMARY_FIELDS = (
    "test",
    "baseline",
    "trial",
    "seed",
    "init_noise",
    "outcome",
    "t_end",
    "pos_x_final",
    "fell",
    "fallen_at_end",
    "t_first_fall",
    "time_to_recovery_s",
    "time_not_fallen_s",
    "rms_height_m",
    "rms_pitch_rad",
    "rms_speed_mps",
    "rms_tracking",
    "avg_grf_mag_n",
    "avg_grf_horizontal_n",
    "nonfoot_contact_fraction",
    "nonfoot_geoms",
    "llm_turn_count",
    "wall_s",
    "fault_injected",
    "episode_dir",
)


def trial_indices(suite: SuiteConfig) -> list[int]:
    """Trials this process should run. ``suite.trial`` selects one of them."""
    if suite.num_trials < 1:
        raise ValueError("num_trials must be >= 1")
    if suite.trial is None:
        return list(range(suite.num_trials))
    if suite.trial < 0 or suite.trial >= suite.num_trials:
        raise ValueError(f"trial must be in 0..{suite.num_trials - 1}")
    return [suite.trial]


def _write_resolved_config(run_dir: Path, suite: SuiteConfig) -> None:
    path = run_dir / "config.resolved.yaml"
    text = yaml.safe_dump(_json_ready(suite.to_dict()), sort_keys=False)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


def _write_summary_files(run_dir: Path, results: list[dict]) -> None:
    summary_json = run_dir / "summary.json"
    summary_json.write_text(
        json.dumps(_json_ready(results), indent=2), encoding="utf-8"
    )
    summary_csv = run_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_FIELDS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def _ordered_results(run_dir: Path, results: list[dict]) -> list[dict]:
    config_path = run_dir / "config.resolved.yaml"
    if not config_path.is_file():
        return results
    suite = load_suite_yaml(config_path)
    test_order = {test.name: index for index, test in enumerate(suite.tests)}
    baseline_order = {name: index for index, name in enumerate(suite.baselines)}
    return sorted(
        results,
        key=lambda row: (
            test_order.get(str(row.get("test")), 10**6),
            baseline_order.get(str(row.get("baseline")), 10**6),
            int(row.get("trial", 0)),
        ),
    )


def merge_partials(run_dir: Path) -> list[str]:
    """Combine per-job partials into one summary, then write statistics and plots.

    Returns the baselines and trials that were queued but have no partial yet.
    """
    run_dir = Path(run_dir)
    partial_dir = run_dir / "partials"
    results: list[dict] = []
    if partial_dir.is_dir():
        for path in sorted(partial_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                results.extend(payload)
    results = _ordered_results(run_dir, results)
    _write_summary_files(run_dir, results)
    stats = summarize_run(results)
    write_statistics(run_dir, stats)
    plot_paths = plot_run(run_dir, results, stats)
    for path in plot_paths:
        print(f"[walker_suite] plot {path}")

    missing: list[str] = []
    jobs_path = run_dir / "slurm" / "jobs.json"
    if jobs_path.is_file():
        jobs = json.loads(jobs_path.read_text(encoding="utf-8")).get("jobs") or []
        have = {(row.get("baseline"), int(row.get("trial", -1))) for row in results}
        for job in jobs:
            key = (job.get("baseline"), int(job.get("trial", -1)))
            if key not in have:
                missing.append(f"{key[0]} trial {key[1]}")
    if missing:
        print("[walker_suite] missing partials: " + ", ".join(missing))
    else:
        print(f"[walker_suite] merged {len(results)} episodes into {run_dir}")
    return missing


def run_pipeline(suite: SuiteConfig) -> Path:
    from loka.walker_suite.episode import run_episode

    if suite.run_dir is not None:
        run_dir = Path(suite.run_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(suite.output_dir) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(run_dir, suite)

    trials = trial_indices(suite)
    shared_slice = suite.run_dir is not None and suite.trial is not None
    results: list[dict] = []
    for test in suite.tests:
        for baseline in suite.baselines:
            label = f"{test.name}__{baseline}"
            condition: list[dict] = []
            for trial in trials:
                seed = trial_seed(suite.seed, trial)
                print(
                    f"\n=== {label} trial {trial + 1}/{suite.num_trials} "
                    f"seed={seed} ==="
                )
                episode_dir = run_dir / label / f"trial_{trial:02d}"
                metadata = run_episode(
                    test,
                    baseline,
                    suite,
                    episode_dir,
                    trial=trial,
                    seed=seed,
                )
                results.append(metadata)
                condition.append(metadata)
            print(f"[walker_suite] {format_condition_line(summarize_run(condition)['conditions'][0])}")

    if shared_slice:
        partial_dir = run_dir / "partials"
        partial_dir.mkdir(parents=True, exist_ok=True)
        by_baseline: dict[str, list[dict]] = {}
        for row in results:
            by_baseline.setdefault(str(row["baseline"]), []).append(row)
        for baseline, rows in by_baseline.items():
            path = partial_dir / f"{baseline}__trial_{suite.trial:02d}.json"
            path.write_text(json.dumps(_json_ready(rows), indent=2), encoding="utf-8")
            print(f"[walker_suite] partial {path}")
        print(f"\n[walker_suite] wrote {run_dir}")
        return run_dir

    _write_summary_files(run_dir, results)

    stats = summarize_run(results)
    write_statistics(run_dir, stats)
    plot_paths = plot_run(run_dir, results, stats)
    for path in plot_paths:
        print(f"[walker_suite] plot {path}")

    print(f"\n[walker_suite] wrote {run_dir}")
    return run_dir
