"""Run every configured condition, and rebuild metrics from a finished run."""

from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
from datetime import datetime
from pathlib import Path

import yaml

from loka.harness.config import SuiteConfig, load_suite_yaml
from loka.harness.registry import ensure_registered, get_metric, get_task
from loka.harness.stats import format_condition_line, summarize_rows, write_statistics
from loka.harness.storage import json_ready, load_unit_episodes, primary_episode
from loka.harness.units import iter_units


def write_resolved_config(run_dir: Path, suite: SuiteConfig) -> None:
    """Write ``config.resolved.yaml`` once. A second writer leaves it alone."""
    path = Path(run_dir) / "config.resolved.yaml"
    text = yaml.safe_dump(json_ready(suite.to_dict()), sort_keys=False)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


def run_unit(suite: SuiteConfig, unit, run_dir: Path) -> dict:
    """Run one trial unless ``done.json`` is already there."""
    ensure_registered()
    unit_dir = Path(run_dir) / unit.id
    done_path = unit_dir / "done.json"
    if done_path.is_file():
        print(f"[eval] skip {unit.id} (already done)")
        return json.loads(done_path.read_text(encoding="utf-8"))
    unit_dir.mkdir(parents=True, exist_ok=True)
    task_spec = suite.task(unit.task)
    print(f"\n=== {unit.id} seed={unit.seed} ===")
    row = get_task(task_spec.type)(task_spec).run(suite, unit, unit_dir)
    row.update(unit.to_dict())
    _apply_metrics(row, task_spec.metrics, unit_dir)
    done_path.write_text(json.dumps(json_ready(row), indent=2), encoding="utf-8")
    return row


def _apply_metrics(row: dict, names: list[str], unit_dir: Path) -> None:
    episodes = load_unit_episodes(unit_dir)
    for name in names:
        metric = get_metric(name)
        if not episodes:
            row[name] = None
            continue
        if metric.scope == "episode":
            row[name] = metric.fn(primary_episode(episodes))
        else:
            row[name] = metric.fn(episodes)


def _run_unit_star(suite: SuiteConfig, unit, run_dir: str) -> dict:
    return run_unit(suite, unit, Path(run_dir))


def run_suite(suite: SuiteConfig) -> Path:
    ensure_registered()
    if suite.run_dir is not None:
        run_dir = Path(suite.run_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(suite.output_dir) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    # The on-disk config describes the whole suite. A single-unit worker
    # selects its unit in memory and must not publish that filter.
    published = suite
    if suite.unit is not None or suite.trial is not None:
        from dataclasses import replace

        published = replace(suite, unit=None, trial=None)
    write_resolved_config(run_dir, published)

    units = iter_units(suite)
    if suite.unit is not None:
        units = [unit for unit in units if unit.id == suite.unit]
        if not units:
            raise ValueError(f"Unknown unit {suite.unit!r}")
    single = suite.unit is not None

    if not single and suite.workers > 1 and len(units) > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(suite.workers) as pool:
            rows = pool.starmap(
                _run_unit_star, [(suite, unit, str(run_dir)) for unit in units]
            )
    else:
        rows = [run_unit(suite, unit, run_dir) for unit in units]

    if single:
        print(f"[eval] wrote {run_dir / units[0].id}")
        return run_dir

    _finalize(run_dir, rows)
    print(f"[eval] wrote {run_dir}")
    return run_dir


def recompute(run_dir: Path, metric_names: list[str] | None = None) -> list[str]:
    """Rebuild summary, statistics and plots from stored episodes.

    Returns the unit ids that have no ``done.json`` yet.
    """
    ensure_registered()
    run_dir = Path(run_dir)
    suite = load_suite_yaml(run_dir / "config.resolved.yaml")
    # Score the whole run, even if this process was a one-trial slice.
    from dataclasses import replace

    suite = replace(suite, trial=None, unit=None, run_dir=run_dir)
    missing: list[str] = []
    rows: list[dict] = []
    for unit in iter_units(suite):
        unit_dir = run_dir / unit.id
        done_path = unit_dir / "done.json"
        if not done_path.is_file():
            missing.append(unit.id)
            continue
        row = json.loads(done_path.read_text(encoding="utf-8"))
        names = list(metric_names) if metric_names else list(suite.task(unit.task).metrics)
        for name in names:
            get_metric(name)
        _apply_metrics(row, names, unit_dir)
        done_path.write_text(json.dumps(json_ready(row), indent=2), encoding="utf-8")
        rows.append(row)
    _finalize(run_dir, rows)
    if missing:
        print("[eval] missing units: " + ", ".join(missing))
    else:
        print(f"[eval] recomputed {len(rows)} units in {run_dir}")
    return missing


def _finalize(run_dir: Path, rows: list[dict]) -> None:
    _write_summary(run_dir, rows)
    stats = summarize_rows(rows)
    write_statistics(run_dir, stats)
    from loka.harness.plots import plot_run

    for path in plot_run(run_dir, rows, stats):
        print(f"[eval] plot {path}")
    for condition in stats.get("conditions") or []:
        print(f"[eval] {format_condition_line(condition)}")


def _write_summary(run_dir: Path, rows: list[dict]) -> None:
    (run_dir / "summary.json").write_text(
        json.dumps(json_ready(rows), indent=2), encoding="utf-8"
    )
    fieldnames: list[str] = []
    flat_rows = []
    for row in rows:
        flat = {
            key: value
            for key, value in row.items()
            if not isinstance(value, (dict, list))
        }
        flat_rows.append(flat)
        for key in flat:
            if key not in fieldnames:
                fieldnames.append(key)
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(json_ready(flat_rows))
