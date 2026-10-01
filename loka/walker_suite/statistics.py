"""Per-trial and per-condition metrics for the Walker perturbation suite."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

# MJPC residual "Height Goal" setpoint in models/walker/task.xml.
HEIGHT_GOAL_M = 1.2
# Spawn pose is off the speed goal. Fails in this window are not a recovery.
STARTUP_GRACE_S = 1.0
# A single sample under the tracking deadband is not nominal. The walker has
# to hold not-failed (and not fallen) for this long before the bout ends.
NOMINAL_HOLD_S = 0.5

# A logged sample is a fail when the walker has fallen or the tracking-error
# excess is above the error-spec trigger. Nominal is neither.
def _is_failed(row: dict[str, Any]) -> bool:
    return bool(row.get("in_failure") or row.get("fallen"))


def _rms(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return float("nan")
    return float(np.sqrt(np.mean(np.square(arr))))


def trajectory_rms(
    height,
    vel_x,
    pitch,
    speed_goal: float,
    *,
    height_goal: float = HEIGHT_GOAL_M,
) -> dict[str, float]:
    """RMS deviation from the upright height, pitch, and speed setpoints.

    Pitch is measured from upright (0 rad). ``rms_tracking`` is
    ``sqrt(mean(e_height^2 + e_pitch^2 + e_speed^2))``. Height is in meters,
    pitch in radians, and speed in m/s, so the combined value mixes those units.
    """
    height_err = np.asarray(height, dtype=float) - float(height_goal)
    pitch_err = np.asarray(pitch, dtype=float)
    speed_err = np.asarray(vel_x, dtype=float) - float(speed_goal)
    combined = np.sqrt(height_err**2 + pitch_err**2 + speed_err**2)
    return {
        "rms_height_m": _rms(height_err),
        "rms_pitch_rad": _rms(pitch_err),
        "rms_speed_mps": _rms(speed_err),
        "rms_tracking": _rms(combined),
    }


def time_to_recovery(
    times,
    failed,
    *,
    grace_s: float = STARTUP_GRACE_S,
    nominal_hold_s: float = NOMINAL_HOLD_S,
) -> float:
    """Longest fail-to-nominal interval in one episode, in seconds.

    Samples before ``grace_s`` (the first second after spawn) are not fails.
    A sample is failed when the walker has fallen or the tracking error is
    outside the nominal band. Standing back up is not enough: the bout ends
    only once the nominal state has held for ``nominal_hold_s``. A shorter
    touch of that band, including a knee-kick that flickers under the
    deadband, stays inside the same bout. If the episode ends before that
    hold, the open interval runs to the end of the log. A trial that never
    fails after the grace period returns 0.
    """
    t = np.asarray(times, dtype=float)
    flags = np.asarray(failed, dtype=bool)
    if t.size == 0:
        return 0.0
    if flags.shape != t.shape:
        raise ValueError("times and failed must have the same length")
    flags = flags & (t >= float(grace_s))
    hold = float(nominal_hold_s)
    longest = 0.0
    start = None
    i = 0
    n = int(t.size)
    while i < n:
        if flags[i] and start is None:
            start = float(t[i])
            i += 1
            continue
        if start is not None and not flags[i]:
            j = i
            while j < n and not flags[j]:
                j += 1
            span = float(t[j - 1] - t[i])
            if span + 1e-9 >= hold:
                longest = max(longest, float(t[i]) - start)
                start = None
            i = j
            continue
        i += 1
    if start is not None:
        longest = max(longest, float(t[-1]) - start)
    return float(longest)


def time_not_fallen(
    times: list[float] | np.ndarray,
    fallen: list[float] | np.ndarray,
    t_end: float | None = None,
) -> float:
    """Seconds the walker is upright, integrated from the log.

    A sample counts as fallen when ``fallen`` is above 0.5. Each gap takes the
    state of its left sample. The span before the first sample and the span
    from the last sample to ``t_end`` use the nearest sample.
    """
    t = np.asarray(times, dtype=float)
    down = np.asarray(fallen, dtype=float) > 0.5
    if t.size == 0:
        return 0.0
    dt = np.diff(t)
    upright = float(np.sum(dt[~down[:-1]])) if dt.size else 0.0
    if t[0] > 0.0 and not bool(down[0]):
        upright += float(t[0])
    end = float(t[-1] if t_end is None else t_end)
    tail = max(end - float(t[-1]), 0.0)
    if not bool(down[-1]):
        upright += tail
    return upright


def metrics_from_log(
    rows: list[dict[str, Any]],
    *,
    speed_goal: float,
    height_goal: float = HEIGHT_GOAL_M,
    t_end: float | None = None,
) -> dict[str, float | None]:
    """Scalar tracking and recovery metrics for one logged episode."""
    empty = {
        "rms_height_m": None,
        "rms_pitch_rad": None,
        "rms_speed_mps": None,
        "rms_tracking": None,
        "time_to_recovery_s": None,
        "time_not_fallen_s": None,
        "avg_grf_mag_n": None,
        "avg_grf_horizontal_n": None,
        "nonfoot_contact_fraction": None,
    }
    if not rows:
        return empty
    rms = trajectory_rms(
        [row["height"] for row in rows],
        [row["vel_x"] for row in rows],
        [row.get("pitch", 0.0) for row in rows],
        speed_goal,
        height_goal=height_goal,
    )
    recovery = time_to_recovery(
        [row["t"] for row in rows],
        [_is_failed(row) for row in rows],
    )
    upright = (
        time_not_fallen(
            [row["t"] for row in rows],
            [row.get("fallen", 0.0) for row in rows],
            t_end,
        )
        if "fallen" in rows[0]
        else None
    )
    return {
        "rms_height_m": rms["rms_height_m"],
        "rms_pitch_rad": rms["rms_pitch_rad"],
        "rms_speed_mps": rms["rms_speed_mps"],
        "rms_tracking": rms["rms_tracking"],
        "time_to_recovery_s": recovery,
        "time_not_fallen_s": upright,
        "avg_grf_mag_n": _column_mean(rows, "grf_mag"),
        "avg_grf_horizontal_n": _column_mean(rows, "grf_horizontal"),
        "nonfoot_contact_fraction": _column_mean(rows, "nonfoot_floor"),
    }


def touched_nonfoot_geoms(rows: list[dict[str, Any]]) -> list[str]:
    """Non-foot geoms that touched the floor, in first-seen order.

    Reads the ``floor_<geom>`` flags written at log time.
    """
    touched: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if not str(key).startswith("floor_"):
                continue
            try:
                flagged = float(value) > 0.0
            except (TypeError, ValueError):
                flagged = False
            if not flagged:
                continue
            name = str(key)[len("floor_") :]
            if name and name not in seen:
                seen.add(name)
                touched.append(name)
    return touched


def _column_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows or key not in rows[0]:
        return None
    arr = np.asarray([row.get(key, np.nan) for row in rows], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.mean())


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def aggregate_condition(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Means across the trials of one test × baseline."""
    successes = [row for row in trials if row.get("outcome") == "success"]
    success_flags = [1.0 if row.get("outcome") == "success" else 0.0 for row in trials]
    rate, rate_std = _mean_std(success_flags)
    completion, completion_std = _mean_std(
        [_finite_or_none(row.get("t_end")) for row in successes]
    )
    recovery, recovery_std = _mean_std(
        [_finite_or_none(row.get("time_to_recovery_s")) for row in trials]
    )
    rms_h, rms_h_std = _mean_std(
        [_finite_or_none(row.get("rms_height_m")) for row in trials]
    )
    rms_p, rms_p_std = _mean_std(
        [_finite_or_none(row.get("rms_pitch_rad")) for row in trials]
    )
    rms_v, rms_v_std = _mean_std(
        [_finite_or_none(row.get("rms_speed_mps")) for row in trials]
    )
    rms_c, rms_c_std = _mean_std(
        [_finite_or_none(row.get("rms_tracking")) for row in trials]
    )
    upright, upright_std = _mean_std(
        [_finite_or_none(row.get("time_not_fallen_s")) for row in trials]
    )
    grf, grf_std = _mean_std(
        [_finite_or_none(row.get("avg_grf_mag_n")) for row in trials]
    )
    grf_h, grf_h_std = _mean_std(
        [_finite_or_none(row.get("avg_grf_horizontal_n")) for row in trials]
    )
    nonfoot, nonfoot_std = _mean_std(
        [_finite_or_none(row.get("nonfoot_contact_fraction")) for row in trials]
    )
    return {
        "test": trials[0]["test"] if trials else None,
        "baseline": trials[0]["baseline"] if trials else None,
        "n_trials": len(trials),
        "n_success": len(successes),
        "success_rate": rate,
        "success_rate_std": rate_std,
        "avg_completion_time_s": completion,
        "avg_completion_time_std_s": completion_std,
        "avg_time_to_recovery_s": recovery,
        "avg_time_to_recovery_std_s": recovery_std,
        "avg_rms_height_m": rms_h,
        "avg_rms_height_std_m": rms_h_std,
        "avg_rms_pitch_rad": rms_p,
        "avg_rms_pitch_std_rad": rms_p_std,
        "avg_rms_speed_mps": rms_v,
        "avg_rms_speed_std_mps": rms_v_std,
        "avg_rms_tracking": rms_c,
        "avg_rms_tracking_std": rms_c_std,
        "avg_time_not_fallen_s": upright,
        "avg_time_not_fallen_std_s": upright_std,
        "avg_grf_mag_n": grf,
        "avg_grf_mag_std_n": grf_std,
        "avg_grf_horizontal_n": grf_h,
        "avg_grf_horizontal_std_n": grf_h_std,
        "avg_nonfoot_contact_fraction": nonfoot,
        "avg_nonfoot_contact_fraction_std": nonfoot_std,
    }


def _ordered_unique(rows: list[dict[str, Any]], key: str) -> list[str]:
    seen: list[str] = []
    for row in rows:
        value = str(row[key])
        if value not in seen:
            seen.append(value)
    return seen


def summarize_run(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Group trial rows into one record per test × baseline."""
    tests = _ordered_unique(results, "test") if results else []
    baselines = _ordered_unique(results, "baseline") if results else []
    conditions: list[dict[str, Any]] = []
    for test in tests:
        for baseline in baselines:
            trials = [
                row
                for row in results
                if row.get("test") == test and row.get("baseline") == baseline
            ]
            if trials:
                conditions.append(aggregate_condition(trials))
    return {
        "height_goal_m": HEIGHT_GOAL_M,
        "tests": tests,
        "baselines": baselines,
        "conditions": conditions,
    }


def format_condition_line(condition: dict[str, Any]) -> str:
    def _fmt(value: Any, digits: int = 3, unit: str = "") -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.{digits}f}{unit}"

    return (
        f"{condition['test']} / {condition['baseline']}: "
        f"success {_fmt(condition['success_rate'], 2)} "
        f"({condition['n_success']}/{condition['n_trials']}), "
        f"completion {_fmt(condition['avg_completion_time_s'], unit='s')}, "
        f"recovery {_fmt(condition['avg_time_to_recovery_s'], unit='s')}, "
        f"upright {_fmt(condition.get('avg_time_not_fallen_s'), unit='s')}, "
        f"rms h/p/v/track "
        f"{_fmt(condition['avg_rms_height_m'])}/"
        f"{_fmt(condition['avg_rms_pitch_rad'])}/"
        f"{_fmt(condition['avg_rms_speed_mps'])}/"
        f"{_fmt(condition['avg_rms_tracking'])}, "
        f"grf {_fmt(condition.get('avg_grf_mag_n'), unit='N')} "
        f"(horiz {_fmt(condition.get('avg_grf_horizontal_n'), unit='N')}), "
        f"nonfoot {_fmt(condition.get('avg_nonfoot_contact_fraction'), 2)}"
    )


_STAT_FIELDS = (
    "test",
    "baseline",
    "n_trials",
    "n_success",
    "success_rate",
    "success_rate_std",
    "avg_completion_time_s",
    "avg_completion_time_std_s",
    "avg_time_to_recovery_s",
    "avg_time_to_recovery_std_s",
    "avg_rms_height_m",
    "avg_rms_height_std_m",
    "avg_rms_pitch_rad",
    "avg_rms_pitch_std_rad",
    "avg_rms_speed_mps",
    "avg_rms_speed_std_mps",
    "avg_rms_tracking",
    "avg_rms_tracking_std",
    "avg_time_not_fallen_s",
    "avg_time_not_fallen_std_s",
    "avg_grf_mag_n",
    "avg_grf_mag_std_n",
    "avg_grf_horizontal_n",
    "avg_grf_horizontal_std_n",
    "avg_nonfoot_contact_fraction",
    "avg_nonfoot_contact_fraction_std",
)


def write_statistics(run_dir: Path, stats: dict[str, Any]) -> None:
    run_dir = Path(run_dir)
    payload = json.loads(json.dumps(stats, allow_nan=False))
    (run_dir / "statistics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    conditions = stats.get("conditions") or []
    with (run_dir / "statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_STAT_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for condition in conditions:
            row = {key: condition.get(key) for key in _STAT_FIELDS}
            writer.writerow(row)
