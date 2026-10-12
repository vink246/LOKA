"""Bar charts with 95% intervals, and goal-walk trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")


def plot_run(run_dir: Path, rows: list[dict], stats: dict[str, Any]) -> list[Path]:
    if not rows:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"[eval] plots skipped ({exc})")
        return []

    plot_dir = Path(run_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    conditions = stats.get("conditions") or []
    by_task: dict[str, list[dict]] = {}
    for condition in conditions:
        by_task.setdefault(str(condition.get("task")), []).append(condition)
    for task, group in by_task.items():
        for metric in _metrics_for(group):
            path = plot_dir / f"{task}_{metric}.png"
            _bars(plt, path, task, metric, group)
            written.append(path)
    written.extend(_trajectories(plt, plot_dir, rows))
    return written


def _metrics_for(conditions: list[dict]) -> list[str]:
    metrics = []
    for condition in conditions:
        for key in condition:
            if key.endswith("_mean") and condition.get(key) is not None:
                name = key[: -len("_mean")]
                if name not in ("n",) and name not in metrics:
                    metrics.append(name)
    return metrics


def _bars(plt, path: Path, task: str, metric: str, conditions: list[dict]) -> None:
    perturbations = []
    for condition in conditions:
        name = str(condition.get("perturbation"))
        if name not in perturbations:
            perturbations.append(name)
    baselines = []
    for condition in conditions:
        name = str(condition.get("baseline"))
        if name not in baselines:
            baselines.append(name)
    lookup = {
        (str(c.get("perturbation")), str(c.get("baseline"))): c for c in conditions
    }
    fig, ax = plt.subplots(figsize=(max(6.0, 1.4 * len(perturbations)), 4.2))
    width = 0.8 / max(len(baselines), 1)
    x = np.arange(len(perturbations))
    for index, baseline in enumerate(baselines):
        means = []
        errors = []
        for perturbation in perturbations:
            condition = lookup.get((perturbation, baseline))
            mean = None if condition is None else condition.get(f"{metric}_mean")
            low = None if condition is None else condition.get(f"{metric}_ci95_low")
            high = None if condition is None else condition.get(f"{metric}_ci95_high")
            means.append(0.0 if mean is None else float(mean))
            if mean is None or low is None or high is None:
                errors.append(0.0)
            else:
                errors.append(max(float(mean) - float(low), float(high) - float(mean)))
        offset = (index - (len(baselines) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            means,
            width=width,
            yerr=errors,
            label=baseline,
            color=_COLORS[index % len(_COLORS)],
            capsize=3,
        )
    ax.set_xticks(x, perturbations, rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(task)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _trajectories(plt, plot_dir: Path, rows: list[dict]) -> list[Path]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if not row.get("goal_xy"):
            continue
        grouped.setdefault((str(row.get("task")), str(row.get("perturbation"))), []).append(row)
    written = []
    for (task, perturbation), group in grouped.items():
        path = plot_dir / f"{task}_{perturbation}_paths.png"
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        goal = np.asarray(group[0]["goal_xy"], dtype=float)
        radius = float(group[0].get("goal_radius") or 0.25)
        baselines = []
        for row in group:
            name = str(row.get("baseline"))
            if name not in baselines:
                baselines.append(name)
        for row in group:
            pelvis = _pelvis(row.get("episode_dir"))
            if pelvis is None:
                continue
            color = _COLORS[baselines.index(str(row.get("baseline"))) % len(_COLORS)]
            ax.plot(pelvis[:, 0], pelvis[:, 1], color=color, alpha=0.45, linewidth=1.0)
        for index, baseline in enumerate(baselines):
            ax.plot([], [], color=_COLORS[index % len(_COLORS)], label=baseline)
        circle = plt.Circle(goal, radius, fill=False, linestyle="--", color="black")
        ax.add_patch(circle)
        ax.scatter([goal[0]], [goal[1]], c="black", s=20, zorder=3)
        ax.scatter([0.0], [0.0], c="gray", s=20, zorder=3)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"{task} / {perturbation}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    return written


def _pelvis(episode_dir) -> np.ndarray | None:
    if not episode_dir:
        return None
    path = Path(episode_dir) / "timeseries.npz"
    if not path.is_file():
        return None
    with np.load(path) as blob:
        if "pelvis_xyz" not in blob.files:
            return None
        return np.asarray(blob["pelvis_xyz"])
