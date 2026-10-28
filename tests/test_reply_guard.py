"""Reply guard: copy detection, the single retry, and placeholder-safe Task_Targets."""

from __future__ import annotations

import queue

import pytest

from loka.agent import llm
from loka.agent.reply_guard import REPLY_REMINDER, check_reply, feedback_message
from loka.agent.reply_parse import parse_orchestrator_reply

GOOD = 'Semantic_State:\n  Hypothesis: "go"\nTask_Targets:\n  gait.mode: walk\n  gait.goal_x: 2.5\n  gait.goal_y: -0.4\n'


def _check(text, **kw):
    pad, _, _ = parse_orchestrator_reply(text)
    return check_reply(text, pad, **kw)


def test_a_real_reply_passes():
    assert _check(GOOD, pelvis_xy=(0.0, 0.0)) == []


@pytest.mark.parametrize(
    "reply, reason",
    [
        ('Semantic_State:\n  Hypothesis: "<the one thing you do this turn>"\n', "placeholder"),
        ('Semantic_State:\n  Hypothesis: "[The one search or target change this response makes]"\n', "placeholder"),
        ('Semantic_State: {}\nCAMERA: null\nGAIT_TARGETS: null\n', "prompt headings"),
        ("Semantic_State: {}\n", "empty"),
        (GOOD.replace("2.5", "0.05").replace("-0.4", "0.0"), "already stands"),
    ],
)
def test_copied_or_inactive_replies_are_flagged(reply, reason):
    problems = _check(reply, pelvis_xy=(0.0, 0.0))
    assert any(reason in p for p in problems), problems


def test_a_stand_at_the_goal_is_not_flagged_and_no_pose_skips_the_goal_check():
    standing = 'Semantic_State:\n  Hypothesis: "done"\nTask_Targets:\n  gait.mode: stand\n'
    assert _check(standing, pelvis_xy=(0.0, 0.0)) == []
    at_self = GOOD.replace("2.5", "0.0").replace("-0.4", "0.0")
    assert _check(at_self) == []


def test_unparseable_reply_is_flagged():
    assert check_reply("no yaml here", None) == ["it is not YAML in the reply format"]


def test_placeholder_task_targets_never_reach_the_controller():
    pad, _, notes = parse_orchestrator_reply(
        'Semantic_State:\n  Hypothesis: "x"\nTask_Targets:\n  gait.mode: <walk or stand>\n'
        "  gait.speed: 0.15\n  gait.goal_x: <world x in metres>\n  height: True\n"
    )
    assert pad["Task_Targets"] == {"gait.speed": 0.15}
    assert any("non-numeric Task_Targets dropped" in n for n in notes)
    pad, _, _ = parse_orchestrator_reply('Semantic_State:\n  Hypothesis: "x"\nTask_Targets:\n  gait.goal_x: <x>\n')
    assert "Task_Targets" not in pad


class _Stub:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, prefill=None, max_new_tokens=None):
        self.calls.append(messages)
        return self.replies.pop(0)


def _run(stub, guard):
    llm.set_chat_backend(stub)
    try:
        out: queue.Queue = queue.Queue()
        llm.stand_llm_worker([{"role": "user", "content": "turn"}], out, guard=guard)
        return out.get_nowait()
    finally:
        llm.set_chat_backend(None)


def test_worker_appends_the_reminder_last_and_retries_a_copied_reply():
    copied = 'Semantic_State:\n  Hypothesis: "<the one thing you do this turn>"\n'
    stub = _Stub([copied, GOOD])
    result = _run(stub, {"reminder": REPLY_REMINDER, "check": True, "pelvis_xy": (0.0, 0.0)})
    assert len(stub.calls) == 2
    assert stub.calls[0][-1]["content"].endswith(REPLY_REMINDER)
    retry = stub.calls[1]
    assert retry[-2] == {"role": "assistant", "content": copied}
    assert retry[-1]["role"] == "user" and "placeholder" in retry[-1]["content"]
    assert result["scratchpad"]["Task_Targets"]["gait.goal_x"] == 2.5
    assert result["reply_check"]["problems"] and result["reply_check"]["retry_problems"] == []


def test_worker_keeps_a_good_first_reply_and_falls_back_when_the_retry_is_garbage():
    stub = _Stub([GOOD])
    result = _run(stub, {"check": True, "pelvis_xy": (0.0, 0.0)})
    assert len(stub.calls) == 1 and "reply_check" not in result
    at_self = GOOD.replace("2.5", "0.0").replace("-0.4", "0.0")
    stub = _Stub([at_self, "sorry, I cannot"])
    result = _run(stub, {"check": True, "pelvis_xy": (0.0, 0.0)})
    assert len(stub.calls) == 2
    assert result["scratchpad"]["Task_Targets"]["gait.goal_x"] == 0.0
    assert result["reply_check"]["retry_problems"] == ["it is not YAML in the reply format"]


def test_feedback_names_the_problems():
    msg = feedback_message(["it is empty", "its walk goal is where the robot already stands"])
    assert "it is empty" in msg and "already stands" in msg and msg.endswith("from this turn.")


def test_cli_reply_check_defaults():
    from loka.run_loka import parse_args

    assert parse_args(["--orchestrator-mllama", "/tmp/m"]).reply_check is True
    assert parse_args([]).reply_check is False
    assert parse_args(["--reply-check"]).reply_check is True
    assert parse_args(["--orchestrator-mllama", "/tmp/m", "--no-reply-check"]).reply_check is False
