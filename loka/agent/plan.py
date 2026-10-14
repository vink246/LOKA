"""LOKA's self-assigned plan: its own editable route and reasoning.

A mission directive is the operator's and is locked. The plan is LOKA's: how
it intends to satisfy the directives (a detour, a waypoint sequence) and why.
It lives in ``loka_state["plan"]`` and is shown on every turn, so the reasoning
behind a far waypoint survives after the turns that chose it drop out of the
conversation window.

Scratchpad semantics for the ``Plan`` section:

- omitted: the plan is unchanged;
- ``null``, ``{}``, ``""``, ``clear`` or ``none``: the plan is cleared;
- a mapping: its fields replace those fields of the current plan, so
  ``Plan: {Current_Step: 3}`` only advances the step;
- a list of steps or a plain string: replaces the steps / the notes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_FIELDS = {
    "goal": "goal",
    "objective": "goal",
    "rationale": "rationale",
    "reasoning": "rationale",
    "why": "rationale",
    "steps": "steps",
    "route": "steps",
    "waypoints": "steps",
    "current_step": "current_step",
    "step": "current_step",
    "notes": "notes",
    "note": "notes",
}
_CLEAR = {"", "clear", "none", "null", "drop", "done"}
MAX_STEPS = 12
MAX_TEXT = 600


@dataclass
class SelfPlan:
    goal: str = ""
    rationale: str = ""
    steps: list[str] = field(default_factory=list)
    current_step: int = 1
    notes: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    revision: int = 1

    def format_lines(self) -> list[str]:
        lines = [
            f"Revision {self.revision}, written t={self.created_at:.2f}s, "
            f"last edited t={self.updated_at:.2f}s.",
        ]
        if self.goal:
            lines.append(f"Goal: {self.goal}")
        if self.rationale:
            lines.append(f"Rationale: {self.rationale}")
        if self.steps:
            lines.append("Steps:")
            for index, step in enumerate(self.steps, start=1):
                if index < self.current_step:
                    mark = "done"
                elif index == self.current_step:
                    mark = "NOW "
                else:
                    mark = "next"
                lines.append(f"  [{mark}] {index}. {step}")
            if self.current_step > len(self.steps):
                lines.append("  All steps are marked done.")
        if self.notes:
            lines.append(f"Notes: {self.notes}")
        return lines


def _text(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())[:MAX_TEXT]


def _steps(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        value = [value]
    steps = []
    for item in value:
        if isinstance(item, dict):
            item = ", ".join(f"{k}: {v}" for k, v in item.items())
        text = _text(item)
        if text:
            steps.append(text)
    return steps[:MAX_STEPS]


def _is_clear(raw) -> bool:
    if raw is None:
        return True
    if isinstance(raw, str):
        return raw.strip().lower() in _CLEAR
    return isinstance(raw, (dict, list)) and not raw


def apply_plan(loka_state: dict, raw, sim_time: float) -> str:
    """Apply one scratchpad ``Plan`` section; return a one-line summary."""
    current: SelfPlan | None = loka_state.get("plan")
    if _is_clear(raw):
        loka_state["plan"] = None
        return "plan cleared" if current is not None else "no plan to clear"
    if isinstance(raw, list):
        raw = {"steps": raw}
    elif not isinstance(raw, dict):
        raw = {"notes": raw}
    updates = {}
    for key, value in raw.items():
        name = _FIELDS.get(str(key).strip().lower().replace(" ", "_"))
        if name is not None:
            updates[name] = value
    if not updates:
        return "plan unchanged (no known fields: Goal, Rationale, Steps, Current_Step, Notes)"
    if current is None:
        plan = SelfPlan(created_at=float(sim_time), updated_at=float(sim_time), revision=1)
    else:
        plan = current
        plan.updated_at = float(sim_time)
        plan.revision += 1
    if "goal" in updates:
        plan.goal = _text(updates["goal"])
    if "rationale" in updates:
        plan.rationale = _text(updates["rationale"])
    if "notes" in updates:
        plan.notes = _text(updates["notes"])
    if "steps" in updates:
        plan.steps = _steps(updates["steps"])
        if "current_step" not in updates:
            plan.current_step = 1
    if "current_step" in updates:
        try:
            plan.current_step = max(1, int(float(updates["current_step"])))
        except (TypeError, ValueError):
            pass
    loka_state["plan"] = plan
    total = len(plan.steps)
    return f"plan revision {plan.revision}: step {min(plan.current_step, total) if total else 0} of {total}"


def format_plan(loka_state: dict) -> str:
    plan: SelfPlan | None = loka_state.get("plan")
    if plan is None:
        return (
            "None. When a directive needs more than one move (a detour, a waypoint "
            "route, a staged recovery), write one under Plan so later turns keep "
            "the reasoning."
        )
    return "\n".join(
        [
            "Your own plan, not an operator directive. Follow it, advance "
            "Current_Step as steps finish, and rewrite it when the scene or "
            "telemetry says it no longer fits.",
            *plan.format_lines(),
        ]
    )
