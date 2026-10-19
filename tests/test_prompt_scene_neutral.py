"""Prompts give camera facts only: no prop names, no hint that objects can be pushed.

Both orchestrators (GPT with the full prompt, local models with the compact one)
must find what is in the scene from the image themselves.
"""

from __future__ import annotations

import math
import re

import pytest

from loka.agent.compress import format_prop_contact_section
from loka.agent.context import format_compact_vision_block, format_vision_block
from loka.agent.llm import load_stand_system_prompt
from loka.agent.reply_guard import REPLY_REMINDER
from loka.agent.scene_describe import build_describe_prompt
from loka.scene_props import PROP_KINDS
from loka.vision import CameraPose, EgoCameraSpec

_POSE = CameraPose(time=0.0, xyz=(0.0, 0.0, 1.25), yaw=0.0, pitch_down=math.radians(30.0), roll=0.0)
_PROP_WORDS = re.compile(r"\b(" + "|".join(PROP_KINDS) + r"|bookshelf|cardboard)s?\b", re.IGNORECASE)
_PUSH_WORDS = re.compile(r"\b(push|pushes|pushing|pushable|shove|kick|kicked|knock)\b", re.IGNORECASE)


def _prompts() -> dict[str, str]:
    spec = EgoCameraSpec()
    full_cam = format_vision_block(spec, _POSE, 0.2)
    compact_cam = format_compact_vision_block(spec, _POSE, 0.2)
    return {
        "full system": load_stand_system_prompt(style="full", vision=full_cam),
        "compact system": load_stand_system_prompt(style="compact", vision=compact_cam),
        "describe": build_describe_prompt("go to the marked spot", spec, _POSE),
        "reminder": REPLY_REMINDER,
    }


@pytest.mark.parametrize("name", ["full system", "compact system", "describe", "reminder"])
def test_prompts_name_no_props_and_never_suggest_pushing(name):
    text = _prompts()[name]
    assert not _PROP_WORDS.findall(text), f"{name} names props: {sorted(set(_PROP_WORDS.findall(text)))}"
    assert not _PUSH_WORDS.findall(text), f"{name} suggests pushing: {sorted(set(_PUSH_WORDS.findall(text)))}"


def test_camera_blocks_keep_the_camera_facts():
    prompts = _prompts()
    for name in ("full system", "compact system"):
        text = prompts[name]
        assert "ahead" in text and re.search(r"row \d+", text) and "horizon" in text.lower()
        assert "FOV" in text or "fov" in text.lower()


def test_contact_telemetry_is_numbered_not_named():
    frames = [
        {"prop_contacts": [{"prop": "chair_0", "kind": "chair", "force_n": 40.0, "links": ["left_wrist_yaw_link"]}]},
        {"prop_contacts": [{"prop": "box_1", "kind": "box", "force_n": 12.0, "links": ["right_ankle_roll_link"]}]},
    ]
    text = format_prop_contact_section(frames)
    assert text.startswith("OBJECT CONTACT")
    assert "object 1: peak normal 40.0 N" in text and "object 2: peak normal 12.0 N" in text
    assert "left_wrist_yaw_link" in text
    assert not _PROP_WORDS.findall(text) and "chair_0" not in text and "box_1" not in text
    assert format_prop_contact_section([{"prop_contacts": []}]) == ""
