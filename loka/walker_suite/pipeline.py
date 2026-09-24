"""Run every configured Walker test against each baseline."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import yaml

from loka.walker_suite.config import SuiteConfig
from loka.walker_suite.logging import _json_ready
from loka.walker_suite.plots import plot_run
from loka.walker_suite.statistics import (
    format_condition_line,
    summarize_run,
    write_statistics,
)
from loka.walker_suite.stochastic import trial_seed


def run_pipeline(suite: SuiteConfig) -> Path:
    from loka.walker_suite.episode import run_episode

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(suite.output_dir) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_path = run_dir / "config.resolved.yaml"
    resolved_path.write_text(
        yaml.safe_dump(_json_ready(suite.to_dict()), sort_keys=False),
        encoding="utf-8",
    )

    results: list[dict] = []
    for test in suite.tests:
        for baseline in suite.baselines:
            label = f"{test.name}__{baseline}"
            condition: list[dict] = []
            for trial in range(suite.num_trials):
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

    summary_json = run_dir / "summary.json"
    summary_json.write_text(
        json.dumps(_json_ready(results), indent=2), encoding="utf-8"
    )

    summary_csv = run_dir / "summary.csv"
    if results:
        fieldnames = [
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
            "rms_height_m",
            "rms_pitch_rad",
            "rms_speed_mps",
            "rms_tracking",
            "llm_turn_count",
            "wall_s",
            "fault_injected",
            "episode_dir",
        ]
        with summary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

    stats = summarize_run(results)
    write_statistics(run_dir, stats)
    plot_paths = plot_run(run_dir, results, stats)
    for path in plot_paths:
        print(f"[walker_suite] plot {path}")

    print(f"\n[walker_suite] wrote {run_dir}")
    return run_dir
