"""LOKA ablations: config, prompt patches, raw telemetry, cost-only detector."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from loka.agent.ablations import AblationSet, CostDetectorSpec, RawTelemetrySpec, resolve_variant
from loka.agent.anomaly import CostTrackingObserver, EffortDeviationObserver
from loka.agent.directives import Listener, listener_fired
from loka.agent.llm import load_stand_system_prompt
from loka.agent.model_registry import load_model_registry, resolve_model
from loka.agent.prompt_ablations import apply_prompt_ablations
from loka.agent.raw_telemetry import (
    TokenCounter,
    describe_columns,
    fit_raw_telemetry,
    sample_indices,
    token_counter,
)
from loka.agent.session import _RESUME_GOAL, _STOOD_FOR_FAULT, FailureEpisode, OperatorSession
from loka.control.locomotion import REPO_ROOT
from loka.harness.config import parse_suite
from loka.harness.metrics import llm_input_tokens, llm_output_tokens, telemetry_tokens_mean
from loka.harness.storage import EpisodeData

FILLS = {"RAW_TELEMETRY_COLUMNS": "<columns>", "GAIT_SAFE_BAND": "<band>"}
VARIANTS = ("raw_telemetry", "cost_detector", "raw_and_cost")


def _ablations(variant: str, **cost_overrides) -> AblationSet:
    resolved = resolve_variant(variant)
    if cost_overrides and "cost_detector" in resolved["params"]:
        resolved["params"]["cost_detector"].update(cost_overrides)
    return AblationSet.from_dict(resolved)


def _suite(baselines, **extra) -> dict:
    return {
        "num_trials": 1,
        "baselines": baselines,
        "tasks": [
            {
                "name": "goto_goal",
                "type": "goto_goal",
                "goal": {"xy": [1.0, 0.0]},
                "timeout_s": 5.0,
                "perturbations": [{"name": "nominal", "kind": "none"}],
                "metrics": ["success"],
            }
        ],
        **extra,
    }


# -- config ---------------------------------------------------------------------


def test_variant_resolves_and_the_resolved_config_stands_alone(tmp_path: Path):
    suite = parse_suite(
        _suite(
            [
                "loka",
                {"name": "raw", "type": "loka_ablated", "options": {"variant": "raw_and_cost"}},
            ]
        )
    )
    raw = suite.baseline("raw")
    assert raw.options["ablations"]["apply"] == ["cost_detector", "raw_telemetry"]
    assert raw.options["ablations"]["params"]["raw_telemetry"]["budget"] == "match_compressor"
    # Plain loka stays a bare name in the resolved config.
    assert suite.to_dict()["baselines"][0] == "loka"
    # A resolved config parses without the ablation file.
    payload = suite.to_dict()
    payload["ablation_config"] = str(tmp_path / "missing.yaml")
    again = parse_suite(payload)
    assert again.baseline("raw").options["ablations"] == raw.options["ablations"]


def test_loka_ablated_needs_a_known_variant():
    with pytest.raises(ValueError, match="variant"):
        parse_suite(_suite([{"name": "x", "type": "loka_ablated"}]))
    with pytest.raises(ValueError, match="Unknown ablation variant"):
        parse_suite(_suite([{"name": "x", "type": "loka_ablated", "options": {"variant": "nope"}}]))
    with pytest.raises(ValueError, match="loka_ablated"):
        parse_suite(_suite([{"name": "x", "type": "loka", "options": {"variant": "raw_telemetry"}}]))
    with pytest.raises(ValueError, match="no orchestrator"):
        parse_suite(_suite([{"name": "x", "type": "low_level", "options": {"orchestrator": {}}}]))


def test_orchestrator_override_precedence(tmp_path: Path):
    path = tmp_path / "ablations.yaml"
    data = yaml.safe_load((REPO_ROOT / "loka" / "config" / "loka_ablations.yaml").read_text())
    data["variants"]["raw_claude"] = {
        "apply": ["raw_telemetry"],
        "orchestrator": {"model": "claude-sonnet-4-5", "latency": {"mode": "pause"}},
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    suite = parse_suite(
        _suite(
            [
                {"name": "a", "type": "loka_ablated", "options": {"variant": "raw_claude"}},
                {
                    "name": "b",
                    "type": "loka_ablated",
                    "options": {"variant": "raw_claude", "orchestrator": {"model": "gpt-x"}},
                },
            ],
            ablation_config=str(path),
            orchestrator={"model": "gpt-base", "latency": {"mode": "wall", "seconds": 3.0}},
        )
    )
    a = suite.orchestrator_for("a")
    assert (a.model, a.latency_mode, a.latency_s) == ("claude-sonnet-4-5", "pause", 3.0)
    b = suite.orchestrator_for("b")
    assert (b.model, b.latency_mode) == ("gpt-x", "pause")


def test_local_model_entry_is_inlined():
    suite = parse_suite(
        _suite([{"name": "g", "type": "loka", "options": {"orchestrator": {"model": "gemma-3-27b-it"}}}])
    )
    entry = suite.baseline("g").options["model_entry"]
    assert entry["provider"] == "local" and entry["engine"] == "vllm"
    assert entry["slurm"]["gres"].startswith("gpu:")


def test_runtime_config_carries_provider_and_ablations():
    from loka.harness.baselines import loka_runtime_config

    suite = parse_suite(
        _suite(
            [
                {
                    "name": "c",
                    "type": "loka_ablated",
                    "options": {
                        "variant": "raw_telemetry",
                        "orchestrator": {"model": "claude-sonnet-4-5", "max_tokens": 777},
                    },
                }
            ]
        )
    )
    config = loka_runtime_config(suite.baseline("c"), suite)
    assert config.llm_provider == "anthropic"
    assert config.llm_options["max_tokens"] == 777
    assert config.ablations.names == ("raw_telemetry",)


def test_origin_guard(tmp_path: Path):
    for name in ("Qwen/Qwen2.5-7B-Instruct", "deepseek-r1", "glm-4-9b"):
        with pytest.raises(ValueError, match="Chinese origin"):
            resolve_model(name)
    registry = yaml.safe_load((REPO_ROOT / "loka" / "config" / "orchestrator_models.yaml").read_text())
    registry["models"]["sneaky"] = {"provider": "local", "hf_repo": "nvidia/OpenReasoning", "base_family": "qwen"}
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    with pytest.raises(ValueError, match="base family"):
        load_model_registry(path)
    # Every shipped entry passes and routes by prefix when unlisted.
    load_model_registry()
    assert resolve_model("claude-haiku-4-5")["provider"] == "anthropic"
    assert resolve_model("gpt-6-luna")["provider"] == "openai"


# -- prompt patches ----------------------------------------------------------------


@pytest.mark.parametrize("style", ["full", "compact"])
def test_default_prompt_is_unchanged(style):
    path = REPO_ROOT / ("system_prompt_compact.txt" if style == "compact" else "system_prompt.txt")
    base = load_stand_system_prompt(style=style)
    assert base == load_stand_system_prompt(style=style, ablations=AblationSet())
    assert path.read_text(encoding="utf-8").split("{{PLANT}}")[0] in base


@pytest.mark.parametrize("style", ["full", "compact"])
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("hide", [True, False])
def test_every_patch_combination_applies(style, variant, hide):
    ablations = _ablations(variant, hide_residual_telemetry=hide)
    prompt = load_stand_system_prompt(style=style, ablations=ablations, fills=FILLS)
    assert "{{" not in prompt
    if ablations.raw_telemetry is not None:
        assert "RAW TELEMETRY" in prompt and "<columns>" in prompt
    if ablations.cost_detector is not None:
        assert "tracking cost" in prompt
        assert "|bias|, |motor|, or stance slip" not in prompt
        if hide:
            assert "EFFORT DEVIATION" not in prompt


def test_a_moved_anchor_fails_loudly():
    text = load_stand_system_prompt(style="full").replace("STATE COST, the MPC residual", "STATE COST: MPC")
    with pytest.raises(ValueError, match="anchor occurs 0 times"):
        apply_prompt_ablations(text, "full", _ablations("raw_telemetry"), FILLS)
    with pytest.raises(ValueError, match="placeholder"):
        load_stand_system_prompt(style="full", ablations=_ablations("raw_telemetry"), fills={})


# -- raw telemetry ----------------------------------------------------------------


def _frames(n: int = 1000, dt: float = 0.002) -> list[dict]:
    rng = np.random.default_rng(0)
    frames = []
    for i in range(n):
        frames.append(
            {
                "time": i * dt,
                "qpos": np.r_[0.01 * i, 0.0, 0.78, 1.0, 0.0, 0.0, 0.0, rng.normal(size=29)],
                "qvel": rng.normal(size=35),
                "rpy": rng.normal(scale=0.01, size=3),
                "cmd_mode": 1.0,
                "cmd_speed": 0.15,
                "cmd_heading": 0.0,
                "goal_x": 3.0,
                "goal_y": 0.0,
                "goal_active": True,
                "walking": True,
                "effort_dev": rng.normal(size=29),
                "motor_dev": rng.normal(size=29),
                "slip": 0.05,
                "mpc_cost": np.abs(rng.normal(size=12)),
                "contact_forces": np.abs(rng.normal(scale=50, size=(8, 3))),
                "prop_contacts": [],
            }
        )
    return frames


ACTUATORS = [f"joint_{i}" for i in range(29)]


def test_raw_table_fits_the_budget_and_drops_in_order():
    spec = RawTelemetrySpec.from_dict({})
    frames = _frames()
    reference = "x " * 1200
    text, stats = fit_raw_telemetry(
        frames, spec, reference=reference, channels=spec.channels, actuator_names=ACTUATORS
    )
    counter = token_counter(spec.tokenizer)
    assert stats["within_budget"]
    assert counter.count(text) == stats["telemetry_tokens"] <= stats["budget_tokens"]
    assert stats["budget_tokens"] == counter.count(reference)
    assert stats["rows"] >= spec.min_samples
    # The last row is the newest frame.
    assert text.strip().splitlines()[-1].startswith(f"{frames[-1]['time']:.3g},")
    # A tiny budget keeps base only, in one row, and still fits.
    small, small_stats = fit_raw_telemetry(
        frames, spec, reference="x " * 60, channels=spec.channels, actuator_names=ACTUATORS
    )
    assert small_stats["channels"] == ["base"]
    assert counter.count(small) <= small_stats["budget_tokens"]


def test_hidden_residual_channels_and_legend():
    hidden = _ablations("raw_and_cost")
    assert hidden.raw_channels() == ("base", "command", "mpc_cost", "contact_fz", "prop_force")
    shown = _ablations("raw_and_cost", hide_residual_telemetry=False)
    assert "effort_dev" in shown.raw_channels()
    legend = describe_columns(hidden.raw_channels())
    assert "e_<actuator>" not in legend and "c_<channel>" in legend


def test_sample_indices_keep_the_newest_frame():
    assert list(sample_indices(10, 1)) == [9]
    picked = sample_indices(1000, 4)
    assert picked[0] == 0 and picked[-1] == 999 and len(picked) == 4
    counter = TokenCounter.__new__(TokenCounter)
    counter._encoder = None
    counter.name = "chars/4"
    assert counter.count("a" * 9) == 3


def test_raw_runtime_turn_stays_within_the_compressor_budget():
    from loka.agent.runtime import LokaConfig, LokaRuntime
    from loka.sim import Simulation

    runtime = LokaRuntime(
        Simulation(),
        LokaConfig(enable_llm=False, enable_session_log=False, ablations=_ablations("raw_telemetry")),
    )
    runtime.effort_observer.printout = False
    frame = None
    for _ in range(600):
        frame = runtime.step()
    text = runtime._telemetry_for(frame)
    stats = runtime._telemetry_stats
    assert text.startswith("RAW TELEMETRY")
    assert stats["telemetry_tokens"] <= stats["compressor_tokens"]
    assert "effort_dev" in frame and np.asarray(frame["effort_dev"]).shape == (29,)
    assert "RAW TELEMETRY" in runtime.system_prompt


# -- cost detector ----------------------------------------------------------------


def _cost_spec(**overrides) -> CostDetectorSpec:
    data = {"stand_threshold": 1.0, "walk_threshold": 5.0, "consecutive_steps": 5}
    data.update(overrides)
    return CostDetectorSpec.from_dict(data)


def test_cost_observer_trips_on_cost_not_on_residual():
    obs = CostTrackingObserver(_cost_spec(), window_size_steps=10)
    big_residual = np.full(29, 10.0)
    for _ in range(30):
        obs.update(big_residual, np.full(12, 0.01))
    assert obs.trigger_value == pytest.approx(0.12)
    assert not obs.out_of_bounds and obs.consecutive_out_of_bounds == 0
    # The residual is still averaged for the belief search.
    assert float(np.sum(np.abs(obs.joint_bias))) == pytest.approx(290.0)
    for _ in range(30):
        obs.update(np.zeros(29), np.full(12, 1.0))
    assert obs.out_of_bounds and obs.consecutive_out_of_bounds >= 5
    assert obs.trigger_threshold == 1.0
    obs.is_walking = True
    assert obs.trigger_threshold == 5.0


def test_cost_channels_select_the_sum():
    obs = CostTrackingObserver(_cost_spec(cost_channels=["roll", "pitch"]), window_size_steps=2)
    for _ in range(3):
        obs.update(np.zeros(29), np.arange(12, dtype=float))
    assert obs.trigger_value == pytest.approx(1.0)


def test_residual_observer_trigger_is_bias():
    obs = EffortDeviationObserver(window_size_steps=2)
    assert obs.trigger_value is None
    for _ in range(3):
        obs.update(np.full(29, 1.0), np.zeros(12))
    assert obs.kind == "residual"
    assert obs.trigger_value == pytest.approx(29.0)


def test_missing_thresholds_are_reported():
    with pytest.raises(ValueError, match="calibrate_cost_detector"):
        CostTrackingObserver(CostDetectorSpec.from_dict({}))


def test_listener_follows_the_trigger():
    listener = Listener(kind="bias_under_threshold", params={}, message="go", armed_at=0.0)
    frame = {"bias_nm": 50.0, "bias_threshold_nm": 5.0, "trigger_value": 0.5, "trigger_threshold": 1.0}
    assert listener_fired(listener, frame)
    frame["trigger_value"] = 2.0
    assert not listener_fired(listener, frame)
    assert not listener_fired(listener, {"bias_nm": 50.0, "bias_threshold_nm": 5.0})


def test_session_wording_defaults_are_unchanged():
    episode = FailureEpisode(started_at=0.0)
    assert episode.text["stood"] == _STOOD_FOR_FAULT and episode.text["resume"] == _RESUME_GOAL
    cost = FailureEpisode(started_at=0.0, detector="cost")
    turn = cost.build_initial_user_turn("T", {"primary_objective": "o"}, 1.0, "C", just_stood=True, max_turns=1)
    assert "|bias|" not in turn and "tracking cost" in turn
    beat = OperatorSession(raw_telemetry=True).build_heartbeat_turn("T", {}, 1.0, "C", period_s=15)
    assert "closing speed" not in beat and "RAW TELEMETRY" in beat


def test_cost_runtime_hides_the_residual():
    from loka.agent.runtime import LokaConfig, LokaRuntime
    from loka.sim import Simulation

    runtime = LokaRuntime(
        Simulation(),
        LokaConfig(enable_llm=False, enable_session_log=False, ablations=_ablations("cost_detector")),
    )
    runtime.effort_observer.printout = False
    assert isinstance(runtime.effort_observer, CostTrackingObserver)
    frame = None
    for _ in range(150):
        frame = runtime.step()
    text = runtime._telemetry_for(frame)
    assert "TRACKING COST" in text and "STATE COST" in text
    assert "EFFORT DEVIATION" not in text and "stance slip" not in text
    assert frame["trigger_value"] == runtime.effort_observer.trigger_value
    assert "tracking cost" in runtime.system_prompt


# -- metrics ------------------------------------------------------------------------


def test_token_metrics():
    episode = EpisodeData(
        directory=Path("."),
        metadata={},
        turns=[
            {"usage": {"input_tokens": 100, "output_tokens": 10}, "telemetry": {"telemetry_tokens": 50}},
            {"usage": {"input_tokens": 200, "output_tokens": 30}, "telemetry": {"telemetry_tokens": 70}},
            {"usage": None, "telemetry": None},
        ],
    )
    assert llm_input_tokens(episode) == 300.0
    assert llm_output_tokens(episode) == 40.0
    assert telemetry_tokens_mean(episode) == 60.0
    assert llm_input_tokens(EpisodeData(directory=Path("."), metadata={})) is None


def test_shipped_ablation_suite_parses():
    from loka.harness.config import load_suite_yaml

    suite = load_suite_yaml(REPO_ROOT / "loka" / "config" / "eval_ablations.yaml")
    kinds = {spec.name: spec.type for spec in suite.baselines}
    assert kinds["loka_raw_telemetry"] == "loka_ablated"
    assert copy.deepcopy(suite.to_dict()) == parse_suite(suite.to_dict()).to_dict()
