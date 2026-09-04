"""Run every configured Walker test against each baseline."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import yaml

from loka.walker_suite.config import SuiteConfig
from loka.walker_suite.logging import _json_ready


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
            print(f"\n=== {label} ===")
            episode_dir = run_dir / label
            metadata = run_episode(test, baseline, suite, episode_dir)
            results.append(metadata)

    summary_json = run_dir / "summary.json"
    summary_json.write_text(
        json.dumps(_json_ready(results), indent=2), encoding="utf-8"
    )

    summary_csv = run_dir / "summary.csv"
    if results:
        fieldnames = [
            "test",
            "baseline",
            "outcome",
            "t_end",
            "pos_x_final",
            "fell",
            "fallen_at_end",
            "t_first_fall",
            "llm_turn_count",
            "wall_s",
            "fault_injected",
        ]
        with summary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

    print(f"\n[walker_suite] wrote {run_dir}")
    return run_dir
