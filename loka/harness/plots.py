"""Charts for a finished run.

Standing push resistance stays one bar chart with a 95% interval. Go-to-goal
is one figure per perturbation: successes, falls, sim time, wall time, and
path efficiency. Counts have no interval. Time and path efficiency use only
the trials that arrived, and a baseline with no arrivals is left off those panels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


# Go-to-goal panels. ``count`` is a sum across trials. ``mean`` averages the
# trials that produced a value; failures store None and drop out.
_GOTO_PANELS = (
    ("Successes", "num_successes", "count"),
    ("Falls", "num_falls", "count"),
    ("Time to goal (sim s)", "time_to_goal_s", "mean"),
    ("Time to goal (wall s)", "wall_time_to_goal_s", "mean"),
    ("Path efficiency", "path_efficiency", "mean"),
)


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
    stand = [c for c in conditions if str(c.get("task")) == "stand_push"]
    if any(c.get("max_push_force_n_mean") is not None for c in stand):
        path = plot_dir / "stand_push_max_push_force_n.png"
        _bars(plt, path, "stand_push", "max_push_force_n", stand)
        written.append(path)
    written.extend(_goto_figures(plt, plot_dir, rows))
    written.extend(_trajectories(plt, plot_dir, rows))
    return written


def _goto_figures(plt, plot_dir: Path, rows: list[dict]) -> list[Path]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if str(row.get("task")) != "goto_goal":
            continue
        grouped.setdefault(str(row.get("perturbation")), []).append(row)
    written = []
    for perturbation, group in grouped.items():
        path = plot_dir / f"goto_goal_{perturbation}.png"
        _goto_figure(plt, path, perturbation, group)
        written.append(path)
    return written


def _goto_figure(plt, path: Path, perturbation: str, rows: list[dict]) -> None:
    baselines = []
    for row in rows:
        name = str(row.get("baseline"))
        if name not in baselines:
            baselines.append(name)
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 11.0), constrained_layout=True)
    flat = list(axes.ravel())
    x = np.arange(len(baselines))
    colors = [_COLORS[i % len(_COLORS)] for i in range(len(baselines))]
    for ax, (title, key, how) in zip(flat, _GOTO_PANELS):
        means, errors = _panel_values(rows, baselines, key, how)
        for index, (height, err) in enumerate(zip(means, errors)):
            if not np.isfinite(height):
                continue
            ax.bar(
                x[index],
                height,
                color=colors[index],
                yerr=None if how == "count" else err,
                capsize=0 if how == "count" else 3,
            )
        ax.set_xticks(x, baselines)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        finite = means[np.isfinite(means)]
        top = float(np.max(finite)) if finite.size else 1.0
        if how == "mean":
            err = errors[np.isfinite(means)] if finite.size else np.array([0.0])
            top = float(np.max(finite + err[: finite.size])) if finite.size else 1.0
        ax.set_ylim(0.0, max(top * 1.15, 1e-3))
    flat[-1].axis("off")
    fig.suptitle(f"goto_goal / {perturbation}")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _panel_values(rows: list[dict], baselines: list[str], key: str, how: str):
    from loka.harness.stats import t_interval

    means = []
    errors = []
    for baseline in baselines:
        group = [row for row in rows if str(row.get("baseline")) == baseline]
        if how == "count":
            total = 0.0
            for row in group:
                value = row.get(key)
                if value is None:
                    continue
                total += float(value)
            means.append(total)
            errors.append(0.0)
            continue
        numbers = []
        for row in group:
            value = row.get(key)
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                numbers.append(number)
        if not numbers:
            means.append(np.nan)
            errors.append(0.0)
            continue
        arr = np.asarray(numbers, dtype=float)
        mean, _std, low, high = t_interval(arr)
        means.append(float(mean))
        if low is None or high is None:
            errors.append(0.0)
        else:
            errors.append(max(float(mean) - float(low), float(high) - float(mean)))
    return np.asarray(means, dtype=float), np.asarray(errors, dtype=float)


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
