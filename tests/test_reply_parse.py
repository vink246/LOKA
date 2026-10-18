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


@pytest.mark.parametrize("reply", ["", "   ", "I cannot help with that.", "- just\n- a list"])
def test_replies_with_no_scratchpad_raise(reply):
    with pytest.raises(ValueError):
        parse_orchestrator_reply(reply)
