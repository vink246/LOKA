"""Orchestrator reply recovery: markdown, fences, prose and unquoted values."""

from __future__ import annotations

import pytest

from loka.agent.reply_parse import parse_orchestrator_reply

GOOD = """Semantic_State:
  Hypothesis: "walk to the chair"
  Analysis: "clear path ahead"
Task_Targets:
  gait.mode: walk
  gait.speed: 0.15
  gait.goal_x: 3.5
  gait.goal_y: -0.6
"""


def _assert_walks(pad: dict) -> None:
    assert pad["Task_Targets"]["gait.mode"] == "walk"
    assert pad["Task_Targets"]["gait.goal_x"] == 3.5
    assert "Semantic_State" in pad


def test_clean_yaml_passes_untouched():
    pad, cleaned, notes = parse_orchestrator_reply(GOOD)
    _assert_walks(pad)
    assert notes == [] and cleaned == GOOD.strip()


def test_bold_section_names_from_the_pace_run():
    reply = """**Semantic_State**
  Hypothesis: "walk to the chair"
  Analysis: "clear path ahead"
**Task_Targets:**
  gait.mode: walk
  gait.speed: 0.15
  gait.goal_x: 3.5
  gait.goal_y: -0.6
"""
    pad, cleaned, notes = parse_orchestrator_reply(reply)
    _assert_walks(pad)
    assert "**" not in cleaned and "markdown removed from keys" in notes


def test_headings_bold_fields_and_surrounding_prose():
    reply = """Sure! Here is my response.

### Semantic_State
  - **Hypothesis**: walk to the chair
  **Analysis**: The chair is ahead: walk to it.
## Task_Targets
  gait.mode: walk
  gait.speed: 0.15
  gait.goal_x: 3.5
  gait.goal_y: -0.6

Let me know if you need anything else.
"""
    pad, _cleaned, notes = parse_orchestrator_reply(reply)
    assert pad["Task_Targets"]["gait.goal_x"] == 3.5
    assert "prose outside the YAML dropped" in notes


def test_fenced_block_inside_prose_and_unclosed_fence():
    pad, _, notes = parse_orchestrator_reply(f"I will walk.\n```yaml\n{GOOD}```\nDone.")
    _assert_walks(pad)
    assert "code fence removed" in notes
    pad, _, _ = parse_orchestrator_reply(f"```yaml\n{GOOD}")
    _assert_walks(pad)


def test_unquoted_values_with_colons_and_leading_asterisks_get_quoted():
    reply = """Semantic_State:
  Hypothesis: *walk* to the chair
  Analysis: Goal: the chair at (3.5, -0.6). Path: clear # no obstacles
Task_Targets:
  gait.mode: walk
  gait.speed: 0.15
  gait.goal_x: 3.5
  gait.goal_y: -0.6
"""
    pad, _, notes = parse_orchestrator_reply(reply)
    _assert_walks(pad)
    assert pad["Semantic_State"]["Analysis"].startswith("Goal: the chair")
    assert "unsafe values quoted" in notes


def test_lowercase_and_spaced_section_names_are_canonicalized():
    reply = GOOD.replace("Semantic_State:", "semantic_state:").replace("Task_Targets:", "Task targets:")
    pad, _, notes = parse_orchestrator_reply(reply)
    _assert_walks(pad)
    assert "section names normalized" in notes


def test_one_broken_section_does_not_lose_the_others():
    reply = GOOD + "Listeners:\n  - kind: near_xy\n    x: [3.0\n    message: \"close\"\n"
    pad, _, notes = parse_orchestrator_reply(reply)
    _assert_walks(pad)
    assert "Listeners" not in pad
    assert any("section by section" in n and "Listeners" in n for n in notes)


def test_lists_and_listeners_survive():
    reply = GOOD + """Listeners:
  - kind: near_xy
    x: 3.2
    y: -0.6
    radius: 0.4
    message: "Close to the chair. Stand."
"""
    pad, _, _ = parse_orchestrator_reply(reply)
    assert pad["Listeners"][0]["kind"] == "near_xy"
    assert pad["Listeners"][0]["message"] == "Close to the chair. Stand."


#: Turn 4 of logs/loka/local_llm (Llama 3.2 Vision on PACE): every field lost its indent.
FLATTENED = """Semantic_State:

Hypothesis: "Walk to the chair, passing the box on its right."
Analysis: "The box at (1.2, -0.4) is on the straight line, so I go around it first."

Task_Targets:

gait.mode: walk
gait.speed: 0.15
gait.goal_x: 1.2
gait.goal_y: 0.4

Listeners:

- kind: near_xy
  x: 1.0
  y: 0.4
  radius: 0.3
  message: "Past the box. Head to the chair."

Plan:

Goal: "Stand next to the chair."
Rationale: "The box blocks the direct line."
Steps:
  - "Walk to (1.2, 0.4), beside the box"
  - "Walk to (3.0, -0.6), in front of the chair, and stand"
Current_Step: 1

Belief_Searches:

- kind: mass
  name: torso_link

CAMERA:
"""


def test_flattened_sections_from_the_pace_run_are_re_nested():
    pad, _, notes = parse_orchestrator_reply(FLATTENED)
    assert pad["Semantic_State"]["Hypothesis"].startswith("Walk to the chair")
    assert pad["Task_Targets"] == {
        "gait.mode": "walk",
        "gait.speed": 0.15,
        "gait.goal_x": 1.2,
        "gait.goal_y": 0.4,
    }
    assert pad["Listeners"][0]["message"] == "Past the box. Head to the chair."
    assert pad["Plan"]["Current_Step"] == 1 and len(pad["Plan"]["Steps"]) == 2
    assert pad["Belief_Searches"] == [{"kind": "mass", "name": "torso_link"}]
    assert "Hypothesis" not in pad and "gait.mode" not in pad and "Goal" not in pad
    assert "flattened sections re-indented" in notes


def test_flattened_list_item_fields_stay_in_their_item():
    reply = "Semantic_State:\nHypothesis: \"x\"\nListeners:\n- kind: after_s\ndelay: 2.0\nmessage: \"look again\"\n"
    pad, _, _ = parse_orchestrator_reply(reply)
    assert pad["Listeners"] == [{"kind": "after_s", "delay": 2.0, "message": "look again"}]


def test_stray_and_empty_sections_are_salvaged():
    reply = "Semantic_State:\nTask_Targets:\nHypothesis: \"walk\"\ngait.mode: walk\ngait.goal_x: 2.0\nPlan:\n"
    pad, _, notes = parse_orchestrator_reply(reply)
    assert pad["Semantic_State"] == {"Hypothesis": "walk"}
    assert pad["Task_Targets"] == {"gait.mode": "walk", "gait.goal_x": 2.0}
    assert "Plan" not in pad
    pad, _, notes = parse_orchestrator_reply("Task_Targets:\n  gait.mode: stand\n")
    assert pad["Semantic_State"] == {} and "missing Semantic_State added" in notes
    pad, _, _ = parse_orchestrator_reply('Semantic_State: "just a sentence"\n')
    assert pad["Semantic_State"] == {"Hypothesis": "just a sentence"}


#: The shape behind "empty sections dropped: Semantic_State, Task_Targets" on PACE.
BULLETED = """Here is my response:

**Semantic_State:**
* Hypothesis: "Walk toward the chair, keeping the box on the right."
* Analysis: "The path ahead is clear apart from the box."

**Task_Targets:**
* gait.mode: walk
* gait.speed: 0.15
* gait.goal_x: 1.0
* gait.goal_y: 0.35

**Listeners:**
* kind: near_xy
  x: 0.9
  y: 0.35
  radius: 0.3
  message: "Beside the box."

**Plan:**
* Goal: "Stand by the chair."
* Steps:
  1. "Walk to (1.0, 0.35)"
  2. "Walk to the chair"
* Current_Step: 1
"""


def test_markdown_bullets_are_read_as_fields():
    pad, _, notes = parse_orchestrator_reply(BULLETED)
    assert pad["Semantic_State"]["Hypothesis"].startswith("Walk toward the chair")
    assert pad["Task_Targets"] == {
        "gait.mode": "walk",
        "gait.speed": 0.15,
        "gait.goal_x": 1.0,
        "gait.goal_y": 0.35,
    }
    assert pad["Listeners"] == [
        {"kind": "near_xy", "x": 0.9, "y": 0.35, "radius": 0.3, "message": "Beside the box."}
    ]
    assert pad["Plan"]["Steps"] == ["Walk to (1.0, 0.35)", "Walk to the chair"]
    assert pad["Plan"]["Current_Step"] == 1
    assert "markdown bullets turned into YAML items" in notes


def test_listeners_with_no_kind_do_not_clear_the_armed_ones():
    reply = 'Semantic_State:\n  Hypothesis: "x"\nListeners:\n  - message: "no kind here"\n'
    pad, _, notes = parse_orchestrator_reply(reply)
    assert "Listeners" not in pad
    assert "Listeners dropped: no entry has a kind" in notes
    pad, _, _ = parse_orchestrator_reply('Semantic_State:\n  Hypothesis: "x"\nListeners: []\n')
    assert pad["Listeners"] == []


def test_apply_survives_a_null_semantic_state():
    from loka.agent.apply import apply_stand_scratchpad
    from loka.control.locomotion import LocomotionConfig
    from loka.control.stacks import make

    controller = make(LocomotionConfig())
    summary = apply_stand_scratchpad(
        controller, {"Semantic_State": None, "Task_Targets": {"gait.mode": "stand"}}, {"mutations": []}
    )
    assert summary["ok"]


@pytest.mark.parametrize("reply", ["", "   ", "I cannot help with that.", "- just\n- a list"])
def test_replies_with_no_scratchpad_raise(reply):
    with pytest.raises(ValueError):
        parse_orchestrator_reply(reply)
