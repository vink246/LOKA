"""Raw-telemetry ablation: the recent per-tick stream instead of the compressor.

The orchestrator gets a CSV table of single-tick values over the window, with a
short header. Nothing is averaged, ranked, derived or labelled. The table never
costs more tokens than the compressor report it replaces: the row count is
lowered until it fits, then channel groups are dropped, then precision.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import numpy as np

from loka.agent.ablations import COST_CHANNEL_NAMES, RawTelemetrySpec

#: Characters per token when tiktoken is unavailable.
FALLBACK_CHARS_PER_TOKEN = 4
MIN_PRECISION = 2

_BASE_COLUMNS = ("t", "x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz", "wx", "wy", "wz")
_COMMAND_COLUMNS = ("mode", "spd", "hdg", "gx", "gy", "goal", "walk")
_CONTACT_COLUMNS = tuple(f"fz_{side}{i}" for side in ("l", "r") for i in range(4))

_GROUP_DESCRIPTIONS = {
    "base": (
        "t [s]; x y z pelvis position in the world [m]; roll pitch yaw [deg]; "
        "vx vy vz pelvis velocity in the world [m/s]; wx wy wz angular velocity [rad/s]"
    ),
    "command": (
        "mode gait mode (0 stand, 1 walk, 2 tread, 3 limp); spd commanded speed [m/s]; "
        "hdg commanded heading [rad]; gx gy goal [m]; goal 1 while a goal is active; "
        "walk 1 while stepping"
    ),
    "effort_dev": (
        "e_<actuator> effort deviation [Nm], measured force minus the belief model's "
        "inverse dynamics resolved into that actuator"
    ),
    "motor_dev": "m_<actuator> motor deficit [Nm], commanded minus applied torque",
    "slip": "slip fastest planted sole speed in the ground plane [m/s]",
    "mpc_cost": (
        "c_<channel> MPC weighted state residual for "
        + " ".join(COST_CHANNEL_NAMES)
    ),
    "contact_fz": "fz_l0..fz_l3 and fz_r0..fz_r3 normal force at the left and right sole corners [N]",
    "prop_force": "prop_n total contact force from objects [N]",
}


class TokenCounter:
    """Token count with a tiktoken encoding, or characters / 4 without it."""

    def __init__(self, encoding: str = "o200k_base") -> None:
        self.name = encoding
        self._encoder = None
        try:
            import tiktoken

            self._encoder = tiktoken.get_encoding(encoding)
        except Exception as exc:  # missing package or no network for the BPE file
            warnings.warn(
                f"tiktoken encoding {encoding!r} unavailable ({exc}); counting tokens as "
                f"characters / {FALLBACK_CHARS_PER_TOKEN}",
                stacklevel=2,
            )
            self.name = f"chars/{FALLBACK_CHARS_PER_TOKEN}"

    @property
    def exact(self) -> bool:
        return self._encoder is not None

    def count(self, text: str) -> int:
        if self._encoder is not None:
            return len(self._encoder.encode(text, disallowed_special=()))
        return math.ceil(len(text) / FALLBACK_CHARS_PER_TOKEN)


_COUNTERS: dict[str, TokenCounter] = {}


def token_counter(encoding: str = "o200k_base") -> TokenCounter:
    if encoding not in _COUNTERS:
        _COUNTERS[encoding] = TokenCounter(encoding)
    return _COUNTERS[encoding]


def _short(name: str) -> str:
    return name[: -len("_joint")] if name.endswith("_joint") else name


def group_columns(group: str, actuator_names: Sequence[str]) -> tuple[str, ...]:
    names = [_short(str(n)) for n in actuator_names]
    if group == "base":
        return _BASE_COLUMNS
    if group == "command":
        return _COMMAND_COLUMNS
    if group == "effort_dev":
        return tuple(f"e_{n}" for n in names)
    if group == "motor_dev":
        return tuple(f"m_{n}" for n in names)
    if group == "slip":
        return ("slip",)
    if group == "mpc_cost":
        return tuple(f"c_{n}" for n in COST_CHANNEL_NAMES)
    if group == "contact_fz":
        return _CONTACT_COLUMNS
    if group == "prop_force":
        return ("prop_n",)
    raise ValueError(f"unknown raw channel group {group!r}")


def _vector(frame: dict, key: str, size: int) -> np.ndarray:
    value = frame.get(key)
    if value is None:
        return np.zeros(size)
    out = np.asarray(value, dtype=float).reshape(-1)
    if out.size < size:
        out = np.concatenate([out, np.zeros(size - out.size)])
    return out[:size]


def group_values(frame: dict, group: str, n_actuators: int) -> list[float]:
    if group == "base":
        qpos = _vector(frame, "qpos", 3)
        qvel = _vector(frame, "qvel", 6)
        rpy = np.degrees(_vector(frame, "rpy", 3))
        return [float(frame.get("time", 0.0)), *qpos[:3], *rpy, *qvel[:6]]
    if group == "command":
        return [
            _num(frame.get("cmd_mode")),
            _num(frame.get("cmd_speed")),
            _num(frame.get("cmd_heading")),
            _num(frame.get("goal_x")),
            _num(frame.get("goal_y")),
            1.0 if frame.get("goal_active") else 0.0,
            1.0 if frame.get("walking") else 0.0,
        ]
    if group == "effort_dev":
        return list(_vector(frame, "effort_dev", n_actuators))
    if group == "motor_dev":
        return list(_vector(frame, "motor_dev", n_actuators))
    if group == "slip":
        return [_num(frame.get("slip"))]
    if group == "mpc_cost":
        return list(_vector(frame, "mpc_cost", len(COST_CHANNEL_NAMES)))
    if group == "contact_fz":
        forces = np.asarray(frame.get("contact_forces", np.zeros((8, 3))), dtype=float).reshape(-1, 3)
        fz = forces[:, 2] if forces.size else np.zeros(0)
        return list(np.concatenate([fz, np.zeros(max(0, 8 - fz.size))])[:8])
    if group == "prop_force":
        return [float(sum(float(e.get("force_n", 0.0)) for e in frame.get("prop_contacts") or ()))]
    raise ValueError(f"unknown raw channel group {group!r}")


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fmt(value: float, precision: int) -> str:
    if not np.isfinite(value):
        return "nan"
    text = f"{value:.{precision}g}"
    if text in ("-0", "0.0", "-0.0"):
        return "0"
    return text


def describe_columns(channels: Sequence[str]) -> str:
    """The column legend the prompt shows for these channel groups."""
    return "; ".join(_GROUP_DESCRIPTIONS[group] for group in channels) + "."


def window_frames(frames: Sequence[dict], window_s: float) -> list[dict]:
    frames = list(frames)
    if not frames:
        return []
    end = float(frames[-1].get("time", 0.0))
    return [f for f in frames if float(f.get("time", 0.0)) >= end - window_s - 1e-9]


def sample_indices(length: int, n: int) -> np.ndarray:
    """``n`` evenly spaced indices into ``length`` frames; the last is always kept."""
    if length < 1:
        return np.zeros(0, dtype=int)
    n = max(1, min(int(n), length))
    if n == 1:
        return np.array([length - 1])
    return np.unique(np.round(np.linspace(0, length - 1, n)).astype(int))


def render_raw_table(
    frames: Sequence[dict],
    *,
    channels: Sequence[str],
    n: int,
    precision: int,
    window_s: float,
    actuator_names: Sequence[str],
) -> tuple[str, int]:
    """The RAW TELEMETRY section and the row count it holds."""
    idx = sample_indices(len(frames), n)
    rows = [frames[i] for i in idx]
    columns: list[str] = []
    for group in channels:
        columns.extend(group_columns(group, actuator_names))
    n_act = len(actuator_names)
    if len(rows) > 1:
        span = float(rows[-1].get("time", 0.0)) - float(rows[0].get("time", 0.0))
        spacing = f"about {span / (len(rows) - 1):.3f} s apart"
    else:
        spacing = "the tick at this call"
    lines = [
        "RAW TELEMETRY",
        f"  Last {window_s:.1f} s of the per-tick stream: {len(rows)} rows, {spacing}, oldest first; "
        "the last row is this call. Every value is a single tick: nothing is averaged, ranked or derived.",
    ]
    lines.append(",".join(columns))
    for frame in rows:
        values: list[float] = []
        for group in channels:
            values.extend(group_values(frame, group, n_act))
        lines.append(",".join(_fmt(v, precision) for v in values))
    return "\n".join(lines) + "\n", len(rows)


def _drop_sequence(channels: Sequence[str], order: Sequence[str]) -> list[tuple[str, ...]]:
    """``channels``, then with each group of ``order`` removed in turn (never base)."""
    current = list(channels)
    sets = [tuple(current)]
    for group in order:
        if group == "base" or group not in current:
            continue
        current = [c for c in current if c != group]
        sets.append(tuple(current))
    return sets


def fit_raw_telemetry(
    frames: Sequence[dict],
    spec: RawTelemetrySpec,
    *,
    reference: str | None,
    channels: Sequence[str],
    actuator_names: Sequence[str],
) -> tuple[str, dict]:
    """The largest raw table that fits the token budget, and how it was fit.

    ``reference`` is the compressor report for this turn; its token count is
    the budget when ``spec.budget`` is ``match_compressor``.
    """
    counter = token_counter(spec.tokenizer)
    reference_tokens = counter.count(reference) if reference is not None else None
    if spec.budget == "match_compressor":
        if reference_tokens is None:
            raise ValueError("raw telemetry with budget match_compressor needs the compressor report")
        budget = int(math.floor(reference_tokens * spec.budget_fraction))
    else:
        budget = int(math.floor(int(spec.budget) * spec.budget_fraction))
    window = window_frames(frames, spec.window_s)
    stats: dict = {
        "mode": "raw",
        "budget_tokens": budget,
        "compressor_tokens": reference_tokens,
        "tokenizer": counter.name,
        "tokenizer_exact": counter.exact,
    }
    if not window:
        text = "RAW TELEMETRY\n  (no frames)\n"
        stats.update(telemetry_tokens=counter.count(text), rows=0, precision=None, channels=[])
        stats["within_budget"] = stats["telemetry_tokens"] <= budget
        return text, stats
    max_n = min(len(window), int(round(spec.window_s * spec.max_hz)) + 1)
    min_n = min(spec.min_samples, max_n)

    def render(groups, n, precision):
        text, rows = render_raw_table(
            window,
            channels=groups,
            n=n,
            precision=precision,
            window_s=spec.window_s,
            actuator_names=actuator_names,
        )
        return text, rows, counter.count(text)

    def largest_fit(groups, precision, lo):
        text, rows, tokens = render(groups, lo, precision)
        if tokens > budget:
            return None
        best = (text, rows, tokens)
        hi = max_n
        while lo < hi:
            mid = (lo + hi + 1) // 2
            candidate = render(groups, mid, precision)
            if candidate[2] <= budget:
                best, lo = candidate, mid
            else:
                hi = mid - 1
        return best

    # 1. drop_order groups at full precision; 2. lower precision; 3. drop the
    # remaining groups, last column group first; 4. one row of base.
    configured = _drop_sequence(channels, spec.drop_order)
    reduced = configured[-1]
    remaining = _drop_sequence(reduced, list(reversed(reduced)))[1:]
    attempts = [(groups, spec.precision, min_n) for groups in configured]
    attempts += [(reduced, p, min_n) for p in range(spec.precision - 1, MIN_PRECISION - 1, -1)]
    attempts += [(groups, MIN_PRECISION, min_n) for groups in remaining]
    attempts.append((("base",), MIN_PRECISION, 1))
    for groups, precision, lo in attempts:
        found = largest_fit(groups, precision, lo)
        if found is not None:
            text, rows, tokens = found
            stats.update(
                telemetry_tokens=tokens,
                rows=rows,
                precision=precision,
                channels=list(groups),
                within_budget=True,
            )
            return text, stats
    # Even one base row is over a tiny budget: cut the text to the budget.
    text, rows, tokens = render(("base",), 1, MIN_PRECISION)
    while tokens > budget and len(text) > 1:
        text = text[: max(1, int(len(text) * budget / max(tokens, 1)) - 1)]
        tokens = counter.count(text)
    stats.update(
        telemetry_tokens=tokens,
        rows=rows,
        precision=MIN_PRECISION,
        channels=["base"],
        within_budget=tokens <= budget,
        truncated=True,
    )
    return text, stats
