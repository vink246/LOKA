"""Describe-first orchestration: description prompt, verbatim section, worker flow."""

from __future__ import annotations

import math
import queue

import numpy as np
import pytest

from loka.agent import llm
from loka.agent.scene_describe import DESCRIBE_PREFILL, build_describe_prompt, format_scene_section
from loka.vision import CameraPose, EgoCameraSpec, encode_png

DESCRIPTION = """SCENE: A room with a blue checkered floor.
OBJECT: box | right, about 15 degrees | about 1.3 m away | blocks path: yes
OBJECT: chair | right, about 10 degrees | about 3.5 m away | blocks path: no
TARGET: chair, ahead and slightly right
"""


def _pose() -> CameraPose:
    return CameraPose(time=3.0, xyz=(0.0, 0.0, 1.25), yaw=0.0, pitch_down=math.radians(30.0), roll=0.0)


def test_section_is_the_description_verbatim_with_no_computed_positions():
    section = format_scene_section(DESCRIPTION)
    assert section.startswith("## WHAT THE CAMERA SHOWS")
    assert DESCRIPTION.strip() in section
    # Nothing is derived from the description: no coordinates, no approach point.
    assert "(" not in section.replace("(its yaw in STATE)", "")
    assert "approach" not in section.lower()
    assert "(no description)" in format_scene_section("")


def test_describe_prompt_carries_task_and_camera_facts_only():
    prompt = build_describe_prompt("walk to the chair", EgoCameraSpec(), _pose())
    assert '"walk to the chair"' in prompt
    assert "<left, ahead, or right>, about <degrees> degrees | about <metres> m away" in prompt
    assert prompt.count(" m ahead") >= 3
    for prop in ("box", "table", "shelf", "chair"):
        assert prop not in prompt.replace("walk to the chair", "")


def test_worker_describes_first_then_replies_with_the_description():
    calls = []

    class Stub:
        def chat(self, messages, prefill=None, max_new_tokens=None):
            calls.append({"messages": messages, "prefill": prefill})
            if prefill == DESCRIBE_PREFILL:
                return DESCRIPTION
            return 'Semantic_State:\n  Hypothesis: "go"\nTask_Targets:\n  gait.mode: walk\n  gait.heading: -0.17\n'

    png = encode_png(np.zeros((8, 8, 3), dtype=np.uint8))
    vision = {
        "render": lambda snapshot: png,
        "snapshot": None,
        "detail": "high",
        "frame_path": None,
        "describe": {"prompt": "DESCRIBE THIS"},
    }
    llm.set_chat_backend(Stub())
    try:
        out: queue.Queue = queue.Queue()
        llm.stand_llm_worker(
            [{"role": "system", "content": "S"}, {"role": "user", "content": "turn text"}], out, vision=vision
        )
        result = out.get_nowait()
    finally:
        llm.set_chat_backend(None)
    assert len(calls) == 2
    first = calls[0]["messages"]
    assert len(first) == 1 and first[0]["content"][0]["type"] == "image_url"
    assert first[0]["content"][1]["text"] == "DESCRIBE THIS"
    reply_turn = calls[1]["messages"][-1]["content"]
    text = next(p["text"] for p in reply_turn if p["type"] == "text")
    assert "## WHAT THE CAMERA SHOWS" in text and "about 3.5 m away" in text
    assert any(p["type"] == "image_url" for p in reply_turn)
    assert result["scratchpad"]["Task_Targets"]["gait.heading"] == -0.17
    assert result["scene_description"] == DESCRIPTION.strip()
    assert "_png" not in result


def test_cli_defaults_for_local_and_openai():
    from loka.run_loka import parse_args

    local = parse_args(["--orchestrator-mllama", "/tmp/m", "--vision"])
    assert local.describe_first is True and local.max_memory == 0 and local.prompt_style == "compact"
    gpt = parse_args(["--vision"])
    assert gpt.describe_first is False and gpt.max_memory is None and gpt.prompt_style == "full"
    gpt_compact = parse_args(["--vision", "--prompt-style", "compact", "--describe-first", "--max-memory", "1"])
    assert gpt_compact.describe_first is True and gpt_compact.max_memory == 1
    no_describe = parse_args(["--orchestrator-mllama", "/tmp/m", "--no-describe-first", "--max-memory", "2"])
    assert no_describe.describe_first is False and no_describe.max_memory == 2
    with pytest.raises(SystemExit):
        parse_args(["--max-memory", "-1"])
