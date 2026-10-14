"""LOKA's self-assigned plan: parse, edit, clear, and show on every turn."""

from __future__ import annotations

import yaml

from loka.agent.apply import canonicalize_scratchpad
from loka.agent.plan import apply_plan, format_plan
from loka.agent.session import FailureEpisode, OperatorSession


def _state():
    return {"mission_directives": [], "listeners": [], "plan": None, "mutations": []}


def _route(state, t=9.3):
    apply_plan(
        state,
        {
            "Goal": "Reach (3,0)",
            "Rationale": "Chair at (1.5,0); the right side is clear.",
            "Steps": ["Walk to (1.0,-0.9)", "Walk to (2.0,-0.9)", "Walk to (3,0)"],
        },
        t,
    )


def test_no_plan_invites_one():
    assert format_plan(_state()).startswith("None.")


def test_plan_keeps_rationale_and_marks_progress():
    state = _state()
    _route(state)
    text = format_plan(state)
    assert "Rationale: Chair at (1.5,0)" in text
    assert "[NOW ] 1. Walk to (1.0,-0.9)" in text
    assert "[next] 3. Walk to (3,0)" in text


def test_partial_edit_only_changes_given_fields():
    state = _state()
    _route(state)
    apply_plan(state, {"Current_Step": 2, "Notes": "first waypoint reached"}, 20.0)
    plan = state["plan"]
    assert plan.revision == 2
    assert plan.created_at == 9.3 and plan.updated_at == 20.0
    assert plan.goal == "Reach (3,0)" and len(plan.steps) == 3
    text = format_plan(state)
    assert "[done] 1." in text and "[NOW ] 2." in text
    assert "Notes: first waypoint reached" in text


def test_new_steps_restart_at_step_one():
    state = _state()
    _route(state)
    apply_plan(state, {"Current_Step": 3}, 15.0)
    apply_plan(state, {"Steps": ["Walk to (1.5,1.0)", "Walk to (3,0)"]}, 16.0)
    assert state["plan"].current_step == 1
    assert state["plan"].rationale.startswith("Chair")


def test_clear_forms():
    for raw in (None, "", "clear", "none", {}, []):
        state = _state()
        _route(state)
        apply_plan(state, raw, 30.0)
        assert state["plan"] is None, raw


def test_list_and_string_shorthands():
    state = _state()
    apply_plan(state, ["(1,-0.9)", "(3,0)"], 1.0)
    assert state["plan"].steps == ["(1,-0.9)", "(3,0)"]
    apply_plan(state, "watch the box on the left", 2.0)
    assert state["plan"].notes == "watch the box on the left"
    assert state["plan"].steps == ["(1,-0.9)", "(3,0)"]


def test_scratchpad_key_is_canonicalised():
    pad = canonicalize_scratchpad(
        yaml.safe_load("semantic_state: {hypothesis: x}\nself_assigned_plan: {goal: g}\n")
    )
    assert "Plan" in pad and "Semantic_State" in pad


def test_every_turn_kind_shows_the_plan():
    state = _state()
    _route(state)
    session = OperatorSession()
    episode = FailureEpisode(started_at=1.0)
    turns = [
        session.build_request_turn("go to 3,0", "TEL", state, 3.0, "MPC"),
        session.build_listener_turn(["at waypoint"], "TEL", state, 12.0, "MPC"),
        session.build_heartbeat_turn("TEL", state, 30.0, "MPC", period_s=15.0),
        episode.build_initial_user_turn("TEL", state, 5.0, "MPC"),
        episode.build_user_turn("TEL", state, 6.0, "MPC"),
    ]
    for turn in turns:
        assert "## SELF-ASSIGNED PLAN" in turn
        assert "Walk to (2.0,-0.9)" in turn
        assert turn.index("## LOCKED MISSION DIRECTIVES") < turn.index("## SELF-ASSIGNED PLAN")
