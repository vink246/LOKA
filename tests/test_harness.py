"""Harness config, search, metrics, friction and a short offscreen episode."""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from loka.agent.faults import FaultSpec, resolve_actuator
from loka.agent.runtime import LokaConfig, LokaRuntime
from loka.control.locomotion import REPO_ROOT
from loka.harness.config import (
    ControllerSpec,
    SuiteConfig,
    load_suite_yaml,
    parse_suite,
)
from loka.harness.metrics import (
    fell,
    final_goal_error_m,
    max_push_force_n,
    path_efficiency,
    path_length_m,
    success,
    time_to_goal_s,
)
from loka.harness.registry import get_baseline, register_baseline
from loka.harness.search import FellBeforePush, bisect_force, run_force_search
from loka.harness.stats import t_interval, wilson_interval
from loka.harness.stochastic import apply_seeded_state_noise
from loka.harness.storage import EpisodeData
from loka.harness.units import iter_units
from loka.sim import Simulation

SCENE = REPO_ROOT / "models" / "g1" / "scene.xml"
SUITE = REPO_ROOT / "loka" / "config" / "eval_suite.yaml"


def _episode(**metadata) -> EpisodeData:
    pelvis = np.array([[0.0, 0.0, 0.78], [1.0, 0.0, 0.78]], dtype=float)
    rpy = np.zeros((2, 3))
    return EpisodeData(
        directory=Path("."),
        metadata=metadata,
        timeseries={"pelvis_xyz": pelvis, "rpy": rpy, "goal_distance": np.array([1.0, 0.1])},
    )


def test_registry_extension():
    @register_baseline("dummy_harness_baseline")
    class Dummy:
        pass

    assert get_baseline("dummy_harness_baseline") is Dummy
    with pytest.raises(KeyError, match="baseline"):
        get_baseline("not_a_baseline")


def test_shipped_suite_and_round_trip():
    suite = load_suite_yaml(SUITE)
    units = iter_units(suite)
    # stand_push plus six goto perturbations, two baselines, five trials.
    assert len(units) == 7 * 2 * 5
    assert units[0].id == "stand_push__shoulder_mass__low_level__trial_00"
    again = parse_suite(suite.to_dict())
    assert [unit.id for unit in iter_units(again)] == [unit.id for unit in units]


def test_unknown_names(tmp_path: Path):
    raw = {
        "baselines": ["low_level"],
        "tasks": [
            {
                "name": "goto_goal",
                "type": "goto_goal",
                "goal": {"xy": [1.0, 0.0]},
                "timeout_s": 5.0,
                "perturbations": [{"name": "nominal", "kind": "none"}],
                "metrics": ["not_a_metric"],
            }
        ],
    }
    with pytest.raises(ValueError, match="metric"):
        parse_suite(raw)
    raw["tasks"][0]["metrics"] = ["success"]
    raw["tasks"][0]["perturbations"] = [{"name": "nominal", "kind": "nope"}]
    with pytest.raises(ValueError, match="perturbation"):
        parse_suite(raw)
    raw["tasks"][0]["type"] = "not_a_task"
    raw["tasks"][0]["perturbations"] = [{"name": "nominal", "kind": "none"}]
    with pytest.raises(ValueError, match="type"):
        parse_suite(raw)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"baselines": ["nope"], "tasks": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_suite_yaml(path)


def test_bisection_doubles_then_brackets():
    seen: list[float] = []

    def probe(force: float) -> bool:
        seen.append(force)
        return force >= 5.0

    result = bisect_force(probe, lo=0.0, hi=1.0, resolution=0.2, max_hi=100.0)
    assert seen[:5] == [0.0, 1.0, 2.0, 4.0, 8.0]
    assert 5.0 - 0.2 <= result < 5.0


def test_bisection_fall_at_low_end_and_before_push():
    assert bisect_force(lambda _force: True, lo=0.0, hi=1.0, resolution=0.2, max_hi=10.0) == 0.0

    def probe(_force: float) -> bool:
        raise FellBeforePush()

    assert run_force_search(probe, lo=0.0, hi=1.0, resolution=0.2, max_hi=10.0) == 0.0
    assert bisect_force(lambda _force: False, lo=0.0, hi=1.0, resolution=0.2, max_hi=8.0) == 8.0


def test_metrics_from_stored_episode():
    episode = _episode(
        outcome="success",
        t_end=4.0,
        time_to_goal_s=4.0,
        fell=False,
        goal_xy=[1.0, 0.0],
        final_goal_error_m=0.1,
    )
    assert success(episode) == 1.0
    assert time_to_goal_s(episode) == 4.0
    assert final_goal_error_m(episode) == pytest.approx(0.1)
    assert fell(episode) == 0.0
    assert path_length_m(episode) == pytest.approx(1.0)
    assert path_efficiency(episode) == pytest.approx(1.0)
    missed = _episode(outcome="timeout", fell=True, t_first_fall=1.5, goal_xy=[1.0, 0.0])
    assert success(missed) == 0.0
    assert time_to_goal_s(missed) is None
    survived = _episode(fell=False, fell_before_push=False, push_force_n=4.0)
    toppled = _episode(fell=True, fell_before_push=False, push_force_n=6.0)
    assert max_push_force_n([survived, toppled]) == 4.0
    early = _episode(fell=True, fell_before_push=True, push_force_n=2.0)
    assert max_push_force_n([early]) == 0.0


def test_intervals():
    mean, std, low, high = t_interval(np.array([1.0, 2.0, 3.0]))
    assert mean == pytest.approx(2.0)
    assert std == pytest.approx(1.0)
    assert low < mean < high
    band = wilson_interval(1, 2)
    assert band[0] is not None and 0.0 <= band[0] < band[1] <= 1.0


def test_init_noise_keeps_the_base_pose():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    base = data.qpos[:7].copy()
    joints0 = data.qpos[7:].copy()
    apply_seeded_state_noise(model, data, seed=3, scale=0.01)
    assert np.allclose(data.qpos[:7], base)
    assert float(np.max(np.abs(data.qpos[7:] - joints0))) > 0.0
    joints = data.qpos[7:].copy()
    assert float(np.max(np.abs(data.qvel[6:]))) > 0.0
    again = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, again, 0)
    apply_seeded_state_noise(model, again, seed=3, scale=0.01)
    assert np.allclose(again.qpos, data.qpos)
    other = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, other, 0)
    apply_seeded_state_noise(model, other, seed=4, scale=0.01)
    assert not np.allclose(other.qpos[7:], joints)


def test_floor_priority_sets_contact_friction():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert int(model.geom_priority[floor]) == 1
    nominal = _foot_friction(model)
    assert nominal == pytest.approx(1.0, abs=1e-3)
    model.geom_friction[floor, 0] = 0.05
    lowered = _foot_friction(model)
    assert lowered == pytest.approx(0.05, abs=1e-3)


def _foot_friction(model) -> float:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for _ in range(30):
        mujoco.mj_step(model, data)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    mus = []
    for index in range(data.ncon):
        geoms = {int(data.contact.geom[index, 0]), int(data.contact.geom[index, 1])}
        if floor in geoms:
            mus.append(float(data.contact.friction[index, 0]))
    assert mus, "expected a foot-floor contact"
    return float(np.median(mus))


def test_gear_fault_stays_off_the_controller_model():
    suite = SuiteConfig(
        controller=ControllerSpec(config="loka/config/g1.yaml", stack="legacy_dcm")
    )
    from loka.harness.baselines import LowLevelBaseline

    agent = LowLevelBaseline(None, suite)
    agent.reset(0, 0.0)
    plant = agent.sim.model
    belief = agent.sim.controller.robot.model
    assert plant is not belief
    act_id, _name = resolve_actuator(plant, "right_knee")
    before = float(plant.actuator_gear[act_id, 0])
    belief_before = float(belief.actuator_gear[act_id, 0])
    agent.inject(FaultSpec("actuator_scale", 0.0, {"actuator": "right_knee", "scale": 0.5}))
    assert float(plant.actuator_gear[act_id, 0]) == pytest.approx(before * 0.5)
    assert float(belief.actuator_gear[act_id, 0]) == pytest.approx(belief_before)


def test_fixed_sim_latency_gate():
    sim = Simulation()
    runtime = LokaRuntime(
        sim,
        LokaConfig(
            enable_llm=False,
            enable_session_log=False,
            llm_latency_mode="fixed_sim",
            llm_latency_s=1.0,
            llm_request_timeout_s=0.05,
        ),
    )
    runtime.llm_busy = True
    runtime._dispatch_sim_t = 0.0
    runtime._dispatch_wall_t = time.perf_counter()
    runtime._pending_dispatch_reason = "test"
    runtime._pending_user_content = "hello"
    runtime.llm_queue.put(
        {
            "scratchpad": {"Semantic_State": {"Hypothesis": "mass", "Analysis": "test"}},
            "raw_yaml": "Semantic_State:\n  Hypothesis: mass\n",
        }
    )
    runtime._drain_llm()
    assert runtime.llm_busy
    assert runtime.turns == []
    sim.data.time = 1.0
    runtime._drain_llm()
    assert not runtime.llm_busy
    assert len(runtime.turns) == 1
    assert runtime.turns[0]["timed_out"] is False
    assert runtime.turns[0]["reason"] == "test"

    runtime.llm_busy = True
    runtime.llm_queue = __import__("queue").Queue()
    runtime._dispatch_sim_t = 0.0
    runtime._dispatch_wall_t = time.perf_counter()
    sim.data.time = 2.0
    started = time.perf_counter()
    runtime._drain_llm()
    assert time.perf_counter() - started >= 0.04
    assert runtime.turns[-1]["timed_out"] is True
    assert not runtime.llm_busy


def test_smoke_episode_round_trip(tmp_path: Path):
    from loka.harness.pipeline import recompute, run_suite

    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        yaml.safe_dump(
            {
                "output_dir": str(tmp_path / "out"),
                "seed": 1,
                "num_trials": 1,
                "init_noise": 0.0,
                "log_hz": 20,
                "workers": 1,
                "controller": {"config": "loka/config/g1.yaml", "stack": "legacy_dcm"},
                "baselines": ["low_level"],
                "recording": {"enabled": False, "fps": 5, "width": 64, "height": 48},
                "tasks": [
                    {
                        "name": "goto_goal",
                        "type": "goto_goal",
                        "goal": {"xy": [3.0, 0.0], "speed": 0.2, "radius": 0.25},
                        "command_time_s": 0.0,
                        "perturbation_time_s": 0.05,
                        "timeout_s": 0.3,
                        "perturbations": [{"name": "nominal", "kind": "none"}],
                        "metrics": ["success", "fell", "final_goal_error_m"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    suite = load_suite_yaml(suite_path)
    suite.run_dir = tmp_path / "run"
    run_dir = run_suite(suite)
    unit = run_dir / "goto_goal__nominal__low_level__trial_00"
    assert (unit / "metadata.json").is_file()
    assert (unit / "timeseries.npz").is_file()
    assert (unit / "render.npz").is_file()
    assert (unit / "done.json").is_file()
    assert not (unit / "episode.mp4").exists()
    from loka.harness.storage import load_episode

    loaded = load_episode(unit)
    assert loaded.timeseries["qpos"].ndim == 2
    assert loaded.metadata["outcome"] in {"timeout", "fell", "success"}
    missing = recompute(run_dir, ["success", "fell"])
    assert missing == []
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "statistics.json").is_file()
