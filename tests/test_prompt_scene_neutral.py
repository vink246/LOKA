"""Prompts give camera facts only: no prop names, no hint that objects can be pushed.

Both orchestrators (GPT with the full prompt, local models with the compact one)
must find what is in the scene from the image themselves.
"""

from __future__ import annotations

import math
import re

import pytest

from loka.agent.context import format_compact_vision_block, format_vision_block
from loka.agent.llm import load_stand_system_prompt
from loka.agent.reply_guard import REPLY_REMINDER
from loka.agent.scene_describe import build_describe_prompt
from loka.scene_props import PROP_KINDS
from loka.vision import CameraPose, EgoCameraSpec

_POSE = CameraPose(time=0.0, xyz=(0.0, 0.0, 1.25), yaw=0.0, pitch_down=math.radians(30.0), roll=0.0)
_PROP_WORDS = re.compile(r"\b(" + "|".join(PROP_KINDS) + r"|bookshelf|cardboard)s?\b", re.IGNORECASE)
_PUSH_WORDS = re.compile(r"\b(push|pushes|pushing|pushable|shove|kick|kicked|knock)\b", re.IGNORECASE)


_FLOOR_METRIC_CLUE = re.compile(
    r"checkerboard|plank runs|hardwood|texture tile|metric pattern|floor tile|"
    r"\d+(?:\.\d+)?\s*m\s+squares?|repeating texture every|"
    r"count (?:plank|squares|tile)",
    re.IGNORECASE,
)


def _prompts(*, floor_cell: float | None = None) -> dict[str, str]:
    spec = EgoCameraSpec()
    full_cam = format_vision_block(spec, _POSE, floor_cell)
    compact_cam = format_compact_vision_block(spec, _POSE, floor_cell)
    return {
        "full system": load_stand_system_prompt(style="full", vision=full_cam),
        "compact system": load_stand_system_prompt(style="compact", vision=compact_cam),
        "describe": build_describe_prompt("go to the marked spot", spec, _POSE),
        "reminder": REPLY_REMINDER,
    }


@pytest.mark.parametrize("name", ["full system", "compact system", "describe", "reminder"])
def test_prompts_name_no_props_and_never_suggest_pushing(name):
    text = _prompts(floor_cell=0.2)[name]
    assert not _PROP_WORDS.findall(text), f"{name} names props: {sorted(set(_PROP_WORDS.findall(text)))}"
    assert not _PUSH_WORDS.findall(text), f"{name} suggests pushing: {sorted(set(_PUSH_WORDS.findall(text)))}"


@pytest.mark.parametrize("name", ["full system", "compact system"])
def test_prompts_omit_floor_metric_when_disabled(name):
    text = _prompts(floor_cell=None)[name]
    hits = _FLOOR_METRIC_CLUE.findall(text)
    assert not hits, f"{name} leaks floor metric clues: {hits}"


@pytest.mark.parametrize("name", ["full system", "compact system"])
def test_prompts_include_floor_metric_when_enabled(name):
    text = _prompts(floor_cell=0.2)[name]
    assert "checkerboard" in text.lower()
    assert "0.20 m" in text


@pytest.mark.parametrize("name", ["full system", "compact system"])
def test_prompts_include_wood_floor_metric_when_enabled(name):
    text = _prompts(floor_cell=2.0)[name]
    assert "hardwood" in text.lower()
    assert "2.00 m" in text


def test_camera_blocks_keep_the_camera_facts():
    prompts = _prompts(floor_cell=0.2)
    for name in ("full system", "compact system"):
        text = prompts[name]
        assert "ahead" in text and re.search(r"row \d+", text) and "horizon" in text.lower()
        assert "FOV" in text or "fov" in text.lower()


@pytest.mark.parametrize("name", ["full system", "compact system"])
def test_prompts_promise_no_contact_sensing(name):
    text = _prompts(floor_cell=0.2)[name]
    assert "OBJECT CONTACT" not in text and "prop_n" not in text


@pytest.mark.parametrize("style", ["full", "compact"])
def test_text_only_prompt_has_no_camera_mission_rules(style):
    """Language-only runs must not be told to use a camera they do not have."""
    text = load_stand_system_prompt(style=style, vision=None)
    assert "camera" not in text.lower()
    assert "round the riser" not in text.lower()
    assert "round heights" not in text.lower()
    assert "parsed world xy" in text


@pytest.mark.parametrize("style", ["full", "compact"])
def test_vision_prompt_keeps_image_completion_in_the_camera_block(style):
    spec = EgoCameraSpec()
    cam = (
        format_compact_vision_block(spec, _POSE, 0.2)
        if style == "compact"
        else format_vision_block(spec, _POSE, 0.2)
    )
    text = load_stand_system_prompt(style=style, vision=cam)
    assert "round the riser" not in text.lower()
    assert "no parsed world xy is unfinished until the image shows" in text
