"""Keep small local models from copying the reply template instead of answering.

Two parts. :data:`REPLY_REMINDER` is appended as the last thing in the turn, so
the request to answer for this scene is the most recent text the model reads
(Llama 3.2 Vision folds the system prompt into the user message, ahead of the
turn). :func:`check_reply` flags a reply that copied the template or did not
act on the scene; the worker then re-asks once with :func:`feedback_message`.
"""

from __future__ import annotations

import math
import re

from loka.agent.reply_parse import TOP_LEVEL_KEYS

#: ``<metres>``, ``<walk or stand>``: the compact prompt's placeholders.
_ANGLE_PLACEHOLDER = re.compile(r"<[^<>\n]{1,80}>")
#: ``[The one search or target change this response makes]``: the full prompt's.
_BRACKET_PLACEHOLDER = re.compile(r"\[[A-Z][^\[\]\n]{12,}\]")

REPLY_REMINDER = (
    "## YOUR REPLY\n"
    "Reply now for THIS scene and THIS turn. Plain YAML only, starting with Semantic_State:, "
    "in the reply format from the instructions. Replace every <...> with a value you worked out "
    "from the telemetry and the image above; never copy the format's placeholder text. If the "
    "task is not done, include Task_Targets that move the robot toward the target, not to where "
    "it already stands."
)

#: Smallest distance between a walk goal and the pelvis that counts as moving [m].
MIN_GOAL_OFFSET_M = 0.2


def check_reply(
    raw_text: str,
    scratchpad: dict | None,
    *,
    pelvis_xy: tuple[float, float] | None = None,
    goal_tolerance: float = 0.15,
) -> list[str]:
    """Reasons the reply should be asked again; empty when it is acceptable."""
    if scratchpad is None:
        return ["it is not YAML in the reply format"]
    problems: list[str] = []
    text = raw_text or ""
    if _ANGLE_PLACEHOLDER.search(text) or _BRACKET_PLACEHOLDER.search(text):
        problems.append("it still contains the format's placeholder text (<...>)")
    echoed = [str(k) for k in scratchpad if k not in TOP_LEVEL_KEYS]
    if echoed:
        problems.append(f"it repeats prompt headings ({', '.join(echoed[:4])}) instead of answering")
    sections = [k for k in scratchpad if k in TOP_LEVEL_KEYS and k != "Semantic_State"]
    if not sections and not (scratchpad.get("Semantic_State") or {}):
        problems.append("it is empty")
    targets = scratchpad.get("Task_Targets") or {}
    mode = str(targets.get("gait.mode", "")).strip().lower()
    if pelvis_xy is not None and mode in ("walk", "1", "1.0") and "gait.goal_x" in targets and "gait.goal_y" in targets:
        try:
            gx, gy = float(targets["gait.goal_x"]), float(targets["gait.goal_y"])
        except (TypeError, ValueError):
            gx = gy = None
        if gx is not None:
            px, py = (float(v) for v in pelvis_xy)
            if math.hypot(gx - px, gy - py) <= max(float(goal_tolerance), MIN_GOAL_OFFSET_M):
                problems.append(
                    f"its walk goal ({gx:.2f}, {gy:.2f}) is where the robot already stands "
                    f"({px:.2f}, {py:.2f}), so it does not move the robot"
                )
    return problems


def feedback_message(problems: list[str]) -> str:
    """The user turn that asks again after a rejected reply."""
    return (
        "Your reply was not used because " + "; ".join(problems) + ". "
        "Look at the telemetry and the image again and reply for this scene: plain YAML only, "
        "starting with Semantic_State:, with every value worked out from this turn."
    )
