"""Compact orchestrator prompt: size, required content, and CLI defaults."""

from __future__ import annotations

import re

import pytest

from loka.agent.llm import load_stand_system_prompt
from loka.agent.reply_parse import parse_orchestrator_reply


def _example_yaml(prompt: str) -> str:
    start = prompt.index("Semantic_State:\n")
    lines = []
    for line in prompt[start:].splitlines():
        if not line.strip() or line.startswith("[IMPORTANT]") or line.startswith("##"):
            break
        lines.append(line)
    return "\n".join(lines)


@pytest.mark.parametrize("style", ["full", "compact"])
def test_the_prompt_example_is_a_valid_scratchpad(style):
    prompt = load_stand_system_prompt(style=style)
    pad, _, notes = parse_orchestrator_reply(_example_yaml(prompt))
    assert {"Semantic_State", "Task_Targets", "Listeners", "Plan", "Belief_Searches"} <= set(pad)
    assert notes == []


def test_compact_prompt_keeps_the_rules_and_is_short():
    compact = load_stand_system_prompt(style="compact", vision="## CAMERA\nx")
    full = load_stand_system_prompt(style="full")
    assert "{{" not in compact
    assert len(compact) < len(full) / 2
    for needed in (
        "gait.goal_x",
        "gait.speed: 0",
        "Belief_Searches",
        "bias_under_threshold",
        "near_xy",
        "Plan",
        "no markdown",
        "head-camera image",
    ):
        assert needed in compact
    with pytest.raises(ValueError):
        load_stand_system_prompt(style="tiny")


@pytest.mark.slow
def test_compact_runtime_turn_is_a_fraction_of_the_full_one():
    from loka.agent.runtime import LokaConfig, LokaRuntime
    from loka.control.locomotion import LocomotionConfig
    from loka.sim import Simulation

    sizes = {}
    for style in ("full", "compact"):
        sim = Simulation(LocomotionConfig(), props=None, camera=None)
        rt = LokaRuntime(
            sim,
            LokaConfig(enable_llm=False, enable_session_log=False, print_compressor=False, prompt_style=style),
        )
        for _ in range(300):
            rt.step()
        user = rt.operator_session.build_request_turn(
            "walk to the chair",
            rt._telemetry_for(rt.recent_frames[-1]),
            rt.loka_state,
            float(sim.data.time),
            rt._config_text(),
        )
        sizes[style] = (len(rt.system_prompt), len(user))
        if style == "compact":
            assert "MJCF kinematics" not in rt.system_prompt
            assert "torso_link" in rt.system_prompt and "left_knee" in rt.system_prompt
            assert "STATE COST" not in user and "CURRENT MPC TUNABLES" not in user
            assert re.search(r"\|bias\|=\S+ Nm \(threshold", user)
            assert "gait.mode=stand" in user
    assert sizes["compact"][0] < sizes["full"][0] / 8
    assert sizes["compact"][1] < sizes["full"][1] / 2


def test_prompt_style_defaults_follow_the_orchestrator():
    from loka.run_loka import parse_args

    assert parse_args([]).prompt_style == "full"
    assert parse_args(["--orchestrator-mllama", "/tmp/m"]).prompt_style == "compact"
    assert parse_args(["--orchestrator-mllama", "/tmp/m", "--prompt-style", "full"]).prompt_style == "full"
    assert parse_args(["--prompt-style", "compact"]).prompt_style == "compact"
