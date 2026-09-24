"""Bar and line charts for a finished Walker suite run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from loka.walker_suite.statistics import HEIGHT_GOAL_M

ROLLING_WINDOW_S = 0.5
_GRID_DT = 0.01

BASELINE_COLORS = {
    "loka": "#1b6ca8",
    "fixed_mpc": "#c47b17",
    "dr_rl": "#2f7d4a",
}
_FALLBACK_COLORS = ("#6b4c9a", "#b23a48", "#3d7a7a", "#8a6a2f")


def baseline_color(name: str, index: int) -> str:
    if name in BASELINE_COLORS:
        return BASELINE_COLORS[name]
    return _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)]


def rolling_rms(values, times, window_s: float = ROLLING_WINDOW_S) -> np.ndarray:
    """RMS of ``values`` over the trailing ``window_s`` seconds at each sample."""
    samples = np.asarray(values, dtype=float)
    stamps = np.asarray(times, dtype=float)
    if samples.shape != stamps.shape:
        raise ValueError("values and times must have the same length")
    out = np.empty(samples.size, dtype=float)
    if samples.size == 0:
        return out
    squared = np.square(samples)
    cumulative = np.cumsum(squared)
    start = 0
    window = float(window_s)
    for i, stamp in enumerate(stamps):
        while stamps[start] < stamp - window - 1e-12:
            start += 1
        total = cumulative[i] - (cumulative[start - 1] if start else 0.0)
        count = i - start + 1
        out[i] = np.sqrt(total / count) if count else np.nan
    return out


def _load_curve(episode_dir: Path, speed_goal: float, height_goal: float):
    path = Path(episode_dir) / "timeseries.npz"
    if not path.is_file():
        return None
    with np.load(path) as blob:
        times = np.asarray(blob["t"], dtype=float)
        height = np.asarray(blob["height"], dtype=float)
        speed = np.asarray(blob["vel_x"], dtype=float)
        if "pitch" in blob.files:
            pitch = np.asarray(blob["pitch"], dtype=float)
        else:
            pitch = np.zeros_like(times)
    if times.size == 0:
        return None
    height_err = height - float(height_goal)
    pitch_err = pitch
    speed_err = speed - float(speed_goal)
    combined = np.sqrt(height_err**2 + pitch_err**2 + speed_err**2)
    return {
        "t": times,
        "rms_height": rolling_rms(height_err, times),
        "rms_pitch": rolling_rms(pitch_err, times),
        "rms_speed": rolling_rms(speed_err, times),
        "rms_tracking": rolling_rms(combined, times),
    }


def _mean_on_grid(curves: list[dict[str, np.ndarray]], key: str, grid: np.ndarray):
    if not curves or grid.size == 0:
        return np.array([]), np.array([])
    stacked = []
    for curve in curves:
        stacked.append(
            np.interp(grid, curve["t"], curve[key], left=np.nan, right=np.nan)
        )
    arr = np.vstack(stacked)
    count = np.isfinite(arr).sum(axis=0)
    total = np.nansum(arr, axis=0)
    mean = np.full(grid.shape, np.nan, dtype=float)
    np.divide(total, count, out=mean, where=count > 0)
    if arr.shape[0] > 1:
        centered = np.where(np.isfinite(arr), arr - mean, 0.0)
        var = np.zeros(grid.shape, dtype=float)
        np.divide(
            np.sum(np.square(centered), axis=0),
            count - 1,
            out=var,
            where=count > 1,
        )
        std = np.sqrt(var)
    else:
        std = np.zeros(grid.shape, dtype=float)
    std = np.where(np.isfinite(std), std, 0.0)
    return mean, std


def _condition_map(stats: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (row["test"], row["baseline"]): row for row in stats.get("conditions") or []
    }


def _bar_values(condition: dict[str, Any] | None, mean_key: str, std_key: str):
    if condition is None:
        return np.nan, 0.0
    mean = condition.get(mean_key)
    std = condition.get(std_key)
    if mean is None:
        return np.nan, 0.0
    return float(mean), 0.0 if std is None else float(std)


def plot_run(run_dir: Path, results: list[dict[str, Any]], stats: dict[str, Any]) -> list[Path]:
    """Write one metrics bar chart and one RMS line chart per perturbation."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    conditions = _condition_map(stats)
    tests = list(stats.get("tests") or [])
    baselines = list(stats.get("baselines") or [])
    height_goal = float(stats.get("height_goal_m", HEIGHT_GOAL_M))

    for test in tests:
        written.append(
            _plot_bars(plt, plot_dir, test, baselines, conditions)
        )
        written.append(
            _plot_rms(plt, plot_dir, test, baselines, results, height_goal)
        )
    return written


def _plot_bars(plt, plot_dir: Path, test: str, baselines: list[str], conditions) -> Path:
    panels = (
        ("Success rate", "success_rate", "success_rate_std", (0.0, 1.05)),
        ("Task completion time", "avg_completion_time_s", "avg_completion_time_std_s", None),
        ("Time to recovery", "avg_time_to_recovery_s", "avg_time_to_recovery_std_s", None),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    flat = axes.ravel()
    x = np.arange(len(baselines))
    colors = [baseline_color(name, i) for i, name in enumerate(baselines)]

    for ax, (title, mean_key, std_key, ylim) in zip(flat[:3], panels):
        means = []
        stds = []
        for name in baselines:
            mean, std = _bar_values(conditions.get((test, name)), mean_key, std_key)
            means.append(mean)
            stds.append(std)
        means_arr = np.asarray(means, dtype=float)
        stds_arr = np.asarray(stds, dtype=float)
        stds_arr = np.where(np.isfinite(means_arr), stds_arr, 0.0)
        ax.bar(x, means_arr, color=colors, yerr=stds_arr, capsize=3, ecolor="#333333")
        ax.set_xticks(x)
        ax.set_xticklabels(baselines)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        finite = means_arr[np.isfinite(means_arr)]
        if ylim is not None:
            ax.set_ylim(*ylim)
        elif finite.size == 0:
            ax.set_ylim(0.0, 1.0)
        else:
            top = float(np.max(finite + stds_arr[np.isfinite(means_arr)]))
            ax.set_ylim(0.0, max(top * 1.15, 1e-3))

    ax = flat[3]
    width = 0.18
    series = (
        ("Height", "avg_rms_height_m", "avg_rms_height_std_m", "#4c78a8"),
        ("Pitch", "avg_rms_pitch_rad", "avg_rms_pitch_std_rad", "#e45756"),
        ("Speed", "avg_rms_speed_mps", "avg_rms_speed_std_mps", "#f58518"),
        ("Combined", "avg_rms_tracking", "avg_rms_tracking_std", "#54a24b"),
    )
    for i, (label, mean_key, std_key, color) in enumerate(series):
        means = []
        stds = []
        for name in baselines:
            mean, std = _bar_values(conditions.get((test, name)), mean_key, std_key)
            means.append(mean)
            stds.append(std)
        offset = (i - 1.5) * width
        means_arr = np.asarray(means, dtype=float)
        stds_arr = np.asarray(stds, dtype=float)
        stds_arr = np.where(np.isfinite(means_arr), stds_arr, 0.0)
        ax.bar(
            x + offset,
            means_arr,
            width=width,
            label=label,
            color=color,
            yerr=stds_arr,
            capsize=3,
            ecolor="#333333",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(baselines)
    ax.set_ylabel("RMS error")
    ax.set_title("RMS error")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0.0)

    n_trials = 0
    for name in baselines:
        row = conditions.get((test, name))
        if row:
            n_trials = max(n_trials, int(row["n_trials"]))
    fig.suptitle(f"{test}  ({n_trials} trial{'s' if n_trials != 1 else ''})")
    path = plot_dir / f"{test}_metrics.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _plot_rms(
    plt,
    plot_dir: Path,
    test: str,
    baselines: list[str],
    results: list[dict[str, Any]],
    height_goal: float,
) -> Path:
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 10.5), sharex=True, constrained_layout=True)
    keys = (
        ("rms_height", "Height RMS (m)"),
        ("rms_pitch", "Pitch RMS (rad)"),
        ("rms_speed", "Speed RMS (m/s)"),
        ("rms_tracking", "Combined RMS"),
    )
    grouped: dict[str, list[dict[str, np.ndarray]]] = {name: [] for name in baselines}
    for row in results:
        if row.get("test") != test or row.get("baseline") not in grouped:
            continue
        episode_dir = row.get("episode_dir")
        if not episode_dir:
            continue
        curve = _load_curve(episode_dir, float(row.get("speed_goal", 1.0)), height_goal)
        if curve is not None:
            grouped[row["baseline"]].append(curve)

    t_end = 0.0
    for curves in grouped.values():
        for curve in curves:
            if curve["t"].size:
                t_end = max(t_end, float(curve["t"][-1]))
    grid = np.arange(0.0, t_end + 1e-9, _GRID_DT) if t_end > 0 else np.array([])

    for name_i, name in enumerate(baselines):
        color = baseline_color(name, name_i)
        mean_std = {
            key: _mean_on_grid(grouped[name], key, grid) for key, _label in keys
        }
        for ax, (key, ylabel) in zip(axes, keys):
            mean, std = mean_std[key]
            if mean.size == 0:
                continue
            ax.plot(grid, mean, color=color, label=name, linewidth=1.6)
            if grouped[name] and len(grouped[name]) > 1:
                ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.18)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

    for ax in axes:
        if ax.get_lines():
            ax.legend(frameon=False, loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{test}  ({ROLLING_WINDOW_S:g} s rolling RMS)")
    path = plot_dir / f"{test}_rms_over_time.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
