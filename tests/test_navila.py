"""NaVILA mid-level navigation baseline: parser, history, executor, async runtime."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from loka.control.locomotion import LocomotionConfig
from loka.control.unitree_rl import yaw_of
from loka.sim import Simulation, UnitreeRLSimulation
from loka.vla.navila import (
    NavAction,
    NavActionExecutor,
    NavilaConfig,
    NavilaRuntime,
    apply_scratch_cache_defaults,
    build_prompt,
    parse_action,
    parse_script,
    require_navila_checkpoint,
    require_navila_gpu,
    sample_history,
)


def test_parse_action_variants():
    assert parse_action("The next action is move forward 75 cm.") == NavAction("forward", 75.0)
    assert parse_action("move forward 100 cm") == NavAction("forward", 100.0)
    assert parse_action("turn left 30 degrees") == NavAction("turn_left", 30.0)
    assert parse_action("turn right 15 degree") == NavAction("turn_right", 15.0)
    assert parse_action("stop") == NavAction("stop")
    assert parse_action("The next action is: stop.") == NavAction("stop")
    assert parse_action("I see a chair but cannot decide", fallback="stop") == NavAction("stop")


def test_parse_script():
    actions = parse_script("move forward 50 cm; turn left 30 degrees; stop")
    assert [a.kind for a in actions] == ["forward", "turn_left", "stop"]
    assert actions[0].value == 50.0
    with pytest.raises(ValueError):
        parse_script("   ;  ")


def test_sample_history_keeps_first_and_current():
    frames = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(12)]
    window = sample_history(frames, 8)
    assert len(window) == 8
    assert window[0][0, 0, 0] == 0  # first history frame
    assert window[-1][0, 0, 0] == 11  # current
    short = sample_history(frames[:2], 8)
    assert len(short) == 8
    assert short[-1][0, 0, 0] == 1
    assert short[0][0, 0, 0] == 0
    single = sample_history([frames[3]], 8)
    assert len(single) == 8
    assert all(f[0, 0, 0] == 3 for f in single)


def test_build_prompt_image_tokens():
    prompt = build_prompt("walk to the chair", num_frames=8)
    assert prompt.count("<image>") == 8
    assert "walk to the chair" in prompt
    assert "historical observations" in prompt
    assert "current observation" in prompt


def test_require_checkpoint_refuses_hub_and_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Hugging Face|Refusing"):
        require_navila_checkpoint("a8cheng/navila-llama3-8b-8f")
    with pytest.raises(FileNotFoundError, match="empty"):
        require_navila_checkpoint("")
    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError, match="not found"):
        require_navila_checkpoint(missing)
    incomplete = tmp_path / "ckpt"
    incomplete.mkdir()
    with pytest.raises(FileNotFoundError, match="config.json"):
        require_navila_checkpoint(incomplete)
    (incomplete / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="weight"):
        require_navila_checkpoint(incomplete)
    (incomplete / "model.safetensors").write_bytes(b"x")
    assert require_navila_checkpoint(incomplete) == incomplete.resolve()


def test_scratch_cache_defaults_fill_only_unset(monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", "/custom/cache")
    resolved = apply_scratch_cache_defaults()
    assert resolved["HF_HOME"] == str(Path("~/scratch/huggingface").expanduser())
    assert resolved["XDG_CACHE_HOME"] == "/custom/cache"


def test_require_navila_gpu_messages():
    with mock.patch.dict(sys.modules, {"torch": None}):
        # Force ImportError path by patching the import inside the function.
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(RuntimeError, match="torch is not installed"):
                require_navila_gpu()

    torch_mod = mock.MagicMock()
    torch_mod.cuda.is_available.return_value = False
    llava_mod = mock.MagicMock()
    with mock.patch.dict(sys.modules, {"torch": torch_mod, "llava": llava_mod}):
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            require_navila_gpu()


def test_policy_load_passes_model_name_not_base(tmp_path):
    from loka.vla.navila import NavilaPolicy

    ckpt = tmp_path / "navila-llama3-8b-8f"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}", encoding="utf-8")
    (ckpt / "model.safetensors").write_bytes(b"x")

    model = mock.MagicMock()
    model.to.return_value = model
    builder = mock.MagicMock()
    builder.load_pretrained_model.return_value = ("tok", model, "proc", 4096)
    mm_utils = mock.MagicMock()
    mm_utils.get_model_name_from_path.side_effect = lambda p: Path(p).name
    modules = {
        "torch": mock.MagicMock(),
        "llava": mock.MagicMock(),
        "llava.mm_utils": mm_utils,
        "llava.model": mock.MagicMock(),
        "llava.model.builder": builder,
    }
    with mock.patch.dict(sys.modules, modules):
        policy = NavilaPolicy(ckpt, device="cuda:0")
    builder.load_pretrained_model.assert_called_once_with(
        str(ckpt.resolve()), "navila-llama3-8b-8f", None, device="cuda:0"
    )
    model.to.assert_called_once_with("cuda:0")
    assert policy.tokenizer == "tok" and policy.image_processor == "proc"


def test_script_mode_never_imports_torch():
    """Building a script runtime must not pull torch into sys.modules."""
    before = "torch" in sys.modules
    config = LocomotionConfig()
    sim = Simulation(config, props=None, camera=None)
    runtime = NavilaRuntime(
        sim,
        NavilaConfig(forward_speed=0.25, max_actions=5),
        plant="mpc",
        script=[NavAction("stop")],
        enable_camera=False,
    )
    runtime.submit_instruction("script")
    # One step should dispatch stop and finish.
    for _ in range(5):
        runtime.step()
    assert runtime._finished
    runtime.close()
    if not before:
        assert "torch" not in sys.modules


def _pelvis_yaw(sim) -> float:
    return yaw_of(sim.robot_qpos[3:7])


@pytest.mark.slow
def test_executor_forward_mpc():
    sim = Simulation(LocomotionConfig(), props=None, camera=None)
    sim.reset()
    ex = NavActionExecutor(sim.controller, plant="mpc", forward_speed=0.25, goal_tolerance=0.15)
    start = sim.robot_qpos[:2].copy()
    ex.start(NavAction("forward", 50.0), sim.robot_qpos, float(sim.data.time))
    done = None
    for _ in range(int(20.0 / sim.config.control_dt)):
        sim.step()
        done = ex.poll(sim.robot_qpos, float(sim.data.time))
        if done is not None:
            break
    assert done is not None
    assert done["status"] in ("done", "timeout")
    dist = float(np.linalg.norm(sim.robot_qpos[:2] - start))
    assert dist > 0.25
    assert not sim.fell


@pytest.mark.slow
def test_executor_turn_mpc_inplace():
    sim = Simulation(LocomotionConfig(), props=None, camera=None)
    sim.reset()
    ex = NavActionExecutor(
        sim.controller, plant="mpc", forward_speed=0.25, turn_tolerance_deg=8.0
    )
    yaw0 = _pelvis_yaw(sim)
    xy0 = sim.robot_qpos[:2].copy()
    ex.start(NavAction("turn_left", 30.0), sim.robot_qpos, float(sim.data.time))
    done = None
    for _ in range(int(25.0 / sim.config.control_dt)):
        sim.step()
        done = ex.poll(sim.robot_qpos, float(sim.data.time))
        if done is not None:
            break
    assert done is not None
    yaw1 = _pelvis_yaw(sim)
    err = abs(((yaw1 - yaw0) + math.pi) % (2 * math.pi) - math.pi)
    assert err > math.radians(15.0)
    # Mostly in place: lateral drift should stay small.
    assert float(np.linalg.norm(sim.robot_qpos[:2] - xy0)) < 0.6
    assert not sim.fell


@pytest.mark.slow
def test_executor_turn_unitree_rl_arc():
    sim = UnitreeRLSimulation(LocomotionConfig(), props=None, camera=None)
    sim.reset()
    ex = NavActionExecutor(
        sim.controller, plant="unitree_rl", forward_speed=0.30, turn_tolerance_deg=10.0
    )
    yaw0 = _pelvis_yaw(sim)
    ex.start(NavAction("turn_right", 30.0), sim.robot_qpos, float(sim.data.time))
    done = None
    for _ in range(int(30.0 / sim.config.control_dt)):
        sim.step()
        done = ex.poll(sim.robot_qpos, float(sim.data.time))
        if done is not None:
            break
    assert done is not None
    assert done["kind"] == "turn_arc"
    assert abs(done["yaw_change_deg"]) > 10.0
    assert not sim.fell


class _StubPolicy:
    """Emits a fixed sequence with a small wall delay (async smoke)."""

    def __init__(self, actions: list[str], delay_s: float = 0.05) -> None:
        self.actions = list(actions)
        self.delay_s = delay_s
        self.calls = 0

    def generate(self, frames, instruction: str) -> str:
        del frames, instruction
        time.sleep(self.delay_s)
        idx = min(self.calls, len(self.actions) - 1)
        self.calls += 1
        return self.actions[idx]


@pytest.mark.slow
def test_runtime_async_stub_policy_advances_sim():
    sim = Simulation(LocomotionConfig(), props=None, camera=None)
    sim.reset()
    policy = _StubPolicy(
        ["move forward 40 cm", "turn left 20 degrees", "stop"], delay_s=0.08
    )
    log_dir = Path("logs/navila_test_async")
    cfg = NavilaConfig(
        forward_speed=0.25,
        max_actions=10,
        log_dir=log_dir,
        save_frames=False,
        goal_tolerance=0.18,
        timeout_factor=4.0,
    )
    runtime = NavilaRuntime(
        sim, cfg, plant="mpc", policy=policy, enable_camera=False
    )
    runtime.submit_instruction("go somewhere")
    t0 = float(sim.data.time)
    wall0 = time.perf_counter()
    # While the first inference is in flight, the plant must keep stepping.
    saw_infer = False
    for _ in range(int(40.0 / sim.config.control_dt)):
        runtime.step()
        if runtime._inferring:
            saw_infer = True
            # Advance a bit more while in flight.
            for _ in range(20):
                runtime.step()
            assert float(sim.data.time) > t0
            break
        if runtime._finished:
            break
    assert saw_infer or policy.calls >= 1
    # Finish the episode.
    for _ in range(int(60.0 / sim.config.control_dt)):
        runtime.step()
        if runtime._finished and not runtime.executor.busy and not runtime._inferring:
            break
    assert policy.calls >= 1
    assert float(sim.data.time) > t0
    assert time.perf_counter() - wall0 > 0.05
    runtime.close()


def test_runtime_cancel_inflight_on_new_instruction():
    sim = Simulation(LocomotionConfig(), props=None, camera=None)
    policy = _StubPolicy(["move forward 30 cm", "stop"], delay_s=0.2)
    runtime = NavilaRuntime(
        sim,
        NavilaConfig(max_actions=5, save_frames=False, log_dir=Path("logs/navila_test_cancel")),
        plant="mpc",
        policy=policy,
        enable_camera=False,
    )
    runtime.submit_instruction("first")
    for _ in range(5):
        runtime.step()
    gen = runtime._infer_generation
    runtime.submit_instruction("second")
    assert runtime._infer_generation == gen + 1
    assert runtime.instruction == "second"
    assert not runtime.executor.busy
    runtime.close()


def test_parse_args_navila_script_no_model(tmp_path):
    from loka.run_loka import parse_args

    args = parse_args(
        ["--navila", "--navila-script", "stop", "--headless", "-T", "1", "--no-props"]
    )
    assert args.navila and args.no_llm
    with pytest.raises(SystemExit):
        parse_args(["--navila-script", "stop"])  # needs --navila
