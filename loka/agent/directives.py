"""Locked operator missions and the conditions that wake the orchestrator.

A mission directive is text the operator set. LOKA reads it on every turn and
drives Task_Targets / Error_Tracking / gait toward it. Nothing in a scratchpad
can edit or drop it.

A listener is LOKA's own wake-up: a condition plus the message to show when
it becomes true. The periodic heartbeat is separate and lives in the runtime.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

LISTENER_KINDS = frozenset(
    {
        "near_xy",
        "heading_error_below",
        "speed_below",
        "pelvis_z_below",
        "pelvis_z_above",
        "after_s",
    }
)

_GOAL_HINT = re.compile(
    r"\b(go to|goto|navigate|coordinate|waypoint|destination)\b",
    re.IGNORECASE,
)
_PAIR = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)"
)


@dataclass
class MissionDirective:
    """Operator text that stays for the session."""

    text: str
    set_at: float
    goal_xy: tuple[float, float] | None = None

    def format_line(self) -> str:
        goal = ""
        if self.goal_xy is not None:
            goal = f"  parsed world xy=({self.goal_xy[0]:.3g}, {self.goal_xy[1]:.3g})"
        return f"- t={self.set_at:.2f}s: {self.text.strip()}{goal}"


@dataclass
class Listener:
    """One-shot wake condition armed by the orchestrator."""

    message: str
    kind: str
    params: dict = field(default_factory=dict)
    armed_at: float = 0.0

    def format_line(self) -> str:
        bits = " ".join(f"{k}={v}" for k, v in self.params.items())
        return f"- [{self.kind} {bits}] armed t={self.armed_at:.2f}s — {self.message}"


def parse_goal_xy(text: str) -> tuple[float, float] | None:
    """Pull a world (x, y) out of a navigation sentence, if one is present."""
    if not text or _GOAL_HINT.search(text) is None:
        return None
    match = _PAIR.search(text)
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2))


def register_directive(loka_state: dict, text: str, sim_time: float) -> MissionDirective:
    """Append a locked directive. Existing ones stay."""
    directive = MissionDirective(
        text=text.strip(),
        set_at=float(sim_time),
        goal_xy=parse_goal_xy(text),
    )
    loka_state.setdefault("mission_directives", []).append(directive)
    return directive


def format_mission_directives(loka_state: dict) -> str:
    directives = loka_state.get("mission_directives") or []
    if not directives:
        return (
            "No operator mission directives. The standing charter applies: "
            "hold balance, and wait for a goal."
        )
    lines = [
        "LOCKED. You cannot edit, drop, complete, or replace these.",
        "Pursue them with Task_Targets, gait.*, Error_Tracking, and Listeners.",
    ]
    lines.extend(d.format_line() for d in directives)
    return "\n".join(lines)


def parse_listeners(raw, *, armed_at: float) -> list[Listener]:
    """Parse a Listeners YAML list. Unknown kinds are skipped by the caller."""
    if not raw:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    listeners: list[Listener] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or item.get("when") or "").strip()
        message = str(item.get("message") or "").strip()
        if kind not in LISTENER_KINDS or not message:
            continue
        params = {
            k: item[k]
            for k in item
            if k not in {"kind", "when", "message"}
        }
        listeners.append(
            Listener(message=message, kind=kind, params=params, armed_at=float(armed_at))
        )
    return listeners


def format_listeners(loka_state: dict) -> str:
    listeners = loka_state.get("listeners") or []
    if not listeners:
        return "None armed. A periodic check still runs every 30 s."
    return "\n".join(listener.format_line() for listener in listeners)


def _f(params: dict, key: str, default: float) -> float:
    try:
        return float(params.get(key, default))
    except (TypeError, ValueError):
        return default


def listener_fired(listener: Listener, frame: dict) -> bool:
    """True when this listener's condition holds on the latest frame."""
    qpos = np.asarray(frame.get("qpos", np.zeros(3)), dtype=float)
    kind = listener.kind
    p = listener.params
    if kind == "near_xy":
        radius = _f(p, "radius", 0.4)
        dx = float(qpos[0]) - _f(p, "x", 0.0)
        dy = float(qpos[1]) - _f(p, "y", 0.0)
        return dx * dx + dy * dy <= radius * radius
    if kind == "heading_error_below":
        err = frame.get("heading_error_raw", frame.get("heading_error", 0.0))
        return abs(float(err)) <= _f(p, "tol", 0.15)
    if kind == "speed_below":
        return float(frame.get("planar_speed", 0.0)) <= _f(p, "speed", 0.05)
    if kind == "pelvis_z_below":
        return float(qpos[2]) <= _f(p, "z", 0.6)
    if kind == "pelvis_z_above":
        return float(qpos[2]) >= _f(p, "z", 0.75)
    if kind == "after_s":
        return float(frame.get("time", 0.0)) - listener.armed_at >= _f(p, "delay", 5.0)
    return False


def take_fired(listeners: list[Listener], frame: dict) -> tuple[list[Listener], list[Listener]]:
    """Split into (still armed, just fired)."""
    stay: list[Listener] = []
    fired: list[Listener] = []
    for listener in listeners:
        if listener_fired(listener, frame):
            fired.append(listener)
        else:
            stay.append(listener)
    return stay, fired


def heartbeat_due(now: float, last_invoke_t: float, period_s: float) -> bool:
    return float(now) - float(last_invoke_t) >= float(period_s)
