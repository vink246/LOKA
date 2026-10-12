"""Per-condition means, standard deviations and 95% intervals."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

from loka.harness.storage import json_ready

_SKIP = {
    "task",
    "perturbation",
    "baseline",
    "trial",
    "seed",
    "outcome",
    "episode_dir",
    "probes",
    "goal_xy",
    "goal_via",
}


def _values(rows: list[dict], key: str) -> np.ndarray:
    numbers = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            numbers.append(number)
    return np.asarray(numbers, dtype=float)


def t_interval(values: np.ndarray, alpha: float = 0.05):
    """Mean, sample std, and the 95% t interval. One sample has no interval."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None, None, None
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, 0.0, None, None
    std = float(arr.std(ddof=1))
    half = float(scipy_stats.t.ppf(1.0 - alpha / 2.0, arr.size - 1)) * std / np.sqrt(arr.size)
    return mean, std, mean - half, mean + half


def wilson_interval(successes: int, n: int, z: float = 1.96):
    """Wilson score interval for a success rate. Returns (low, high) or Nones."""
    if n <= 0:
        return None, None
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    margin = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return float(center - margin), float(center + margin)


def metric_keys(rows: list[dict]) -> list[str]:
    keys: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key in _SKIP or key in keys:
                continue
            if isinstance(value, (int, float)) or value is None:
                keys.append(key)
    return keys


def summarize_rows(rows: list[dict]) -> dict[str, Any]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row.get("task"), row.get("perturbation"), row.get("baseline"))].append(row)
    conditions = []
    for (task, perturbation, baseline), trials in groups.items():
        condition: dict[str, Any] = {
            "task": task,
            "perturbation": perturbation,
            "baseline": baseline,
            "n": len(trials),
        }
        for key in metric_keys(trials):
            values = _values(trials, key)
            mean, std, low, high = t_interval(values)
            condition[f"{key}_mean"] = mean
            condition[f"{key}_std"] = std
            condition[f"{key}_ci95_low"] = low
            condition[f"{key}_ci95_high"] = high
            condition[f"{key}_n"] = int(values.size)
        if "success" in trials[0] or any("success" in row for row in trials):
            flags = _values(trials, "success")
            if flags.size:
                low, high = wilson_interval(int(np.sum(flags > 0.5)), int(flags.size))
                condition["success_wilson95_low"] = low
                condition["success_wilson95_high"] = high
        conditions.append(condition)
    return {"conditions": conditions}


def write_statistics(run_dir: Path, stats: dict[str, Any]) -> None:
    run_dir = Path(run_dir)
    (run_dir / "statistics.json").write_text(
        json.dumps(json_ready(stats), indent=2), encoding="utf-8"
    )
    rows = stats.get("conditions") or []
    csv_path = run_dir / "statistics.csv"
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(json_ready(rows))


def format_condition_line(condition: dict[str, Any]) -> str:
    parts = [
        f"{condition.get('task')}/{condition.get('perturbation')}/{condition.get('baseline')}",
        f"n={condition.get('n')}",
    ]
    for key, value in condition.items():
        if not key.endswith("_mean") or value is None:
            continue
        metric = key[: -len("_mean")]
        std = condition.get(f"{metric}_std")
        if std is None:
            parts.append(f"{metric}={value:.4g}")
        else:
            parts.append(f"{metric}={value:.4g}±{std:.3g}")
    return " ".join(parts)
