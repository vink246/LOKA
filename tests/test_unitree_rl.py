"""unitree_rl_lab velocity policy: interface, goal commander and short rollouts."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from loka.control.gait import GaitConfig, MODE_STAND, MODE_WALK
from loka.control.locomotion import REPO_ROOT
from loka.control.unitree_rl import (
    DEFAULT_POLICY_DIR,
    G1_29DOF_SDK_JOINTS,
    MIN_WALK_SPEED,
    GoalVelocityCommander,
    UnitreeRLController,
    UnitreeVelocityPolicy,
    yaw_of,
)
from loka.sim import UnitreeRLSimulation

SCENE = REPO_ROOT / "models" / "g1" / "scene.xml"
LOW = np.array([-0.5, -0.3, -0.2])
HIGH = np.array([1.0, 0.3, 0.2])


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(SCENE))


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def test_joint_map_matches_mjcf(model):
    policy = UnitreeVelocityPolicy(model)
    cfg = yaml.safe_load((DEFAULT_POLICY_DIR / "deploy.yaml").read_text(encoding="utf-8"))
    ids_map = cfg["joint_ids_map"]
    assert sorted(ids_map) == list(range(29))
    for i, sdk in enumerate(ids_map):
        act = int(policy.act_idx[i])
        joint = int(model.actuator_trnid[act, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        assert name == G1_29DOF_SDK_JOINTS[sdk]
        assert policy.qpos_idx[i] == model.jnt_qposadr[joint]
        assert policy.dof_idx[i] == model.jnt_dofadr[joint]
    # Policy joint 0 is the left hip pitch, joint 3 the right hip roll.
    assert G1_29DOF_SDK_JOINTS[ids_map[0]] == "left_hip_pitch_joint"
    assert G1_29DOF_SDK_JOINTS[ids_map[4]] == "right_hip_roll_joint"
    # Gains come from SDK order: the knees are 150 / 4.
    knees = [i for i, sdk in enumerate(ids_map) if "knee" in G1_29DOF_SDK_JOINTS[sdk]]
    assert np.allclose(policy.kp[knees], 150.0) and np.allclose(policy.kd[knees], 4.0)


def test_observation_layout_and_history(model):
    policy = UnitreeVelocityPolicy(model)
    assert [term.name for term in policy.terms] == [
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos_rel",
        "joint_vel_rel",
        "last_action",
    ]
    qpos = np.zeros(model.nq)
    qpos[3:7] = _yaw_quat(0.3)
    qpos[policy.qpos_idx] = policy.default_q
    qvel = np.zeros(model.nv)
    qvel[3:6] = [0.5, -1.0, 2.0]
    first = policy.observe(qpos, qvel, np.array([0.4, 0.1, 5.0]))
    assert first.shape == (480,)
    # Every slot holds the first frame after a reset.
    ang = first[0:15].reshape(5, 3)
    assert np.allclose(ang, [0.1, -0.2, 0.4])
    grav = first[15:30].reshape(5, 3)
    assert np.allclose(grav, [0.0, 0.0, -1.0])
    cmd = first[30:45].reshape(5, 3)
    assert np.allclose(cmd, [0.4, 0.1, 0.2])  # ang_vel_z is clipped to 0.2

    qvel2 = qvel.copy()
    qvel2[policy.dof_idx[0]] = 2.0
    second = policy.observe(qpos, qvel2, np.zeros(3))
    cmd = second[30:45].reshape(5, 3)
    assert np.allclose(cmd[:4], [0.4, 0.1, 0.2]) and np.allclose(cmd[4], 0.0)
    vel = second[45 + 145 : 45 + 290].reshape(5, 29)
    assert vel[4, 0] == pytest.approx(0.1) and vel[3, 0] == 0.0


def test_npz_matches_onnx(model):
    ort = pytest.importorskip("onnxruntime")
    policy = UnitreeVelocityPolicy(model)
    session = ort.InferenceSession(str(DEFAULT_POLICY_DIR / "policy.onnx"))
    x = np.random.default_rng(0).normal(size=(1, 480)).astype(np.float32)
    expected = session.run(None, {"obs": x})[0][0]
    assert np.max(np.abs(policy.mlp(x[0]) - expected)) < 1e-4


def _commander(**cfg) -> GoalVelocityCommander:
    config = GaitConfig()
    config.goal_tolerance = 0.2
    for key, value in cfg.items():
        setattr(config, key, value)
    commander = GoalVelocityCommander(config, LOW, HIGH)
    commander.cruise = 0.15
    return commander


def test_commander_goal_geometry():
    ahead = _commander(goal_active=1.0, goal_x=3.0, goal_y=0.0).desired(np.zeros(2), 0.0)
    assert ahead[0] == pytest.approx(MIN_WALK_SPEED) and ahead[1] == pytest.approx(0.0)
    assert ahead[2] == pytest.approx(0.0)

    behind = _commander(goal_active=1.0, goal_x=-3.0, goal_y=0.0).desired(np.zeros(2), 0.0)
    assert behind[0] < 0.0 and abs(behind[2]) == pytest.approx(0.2)  # walks back, turns at max

    left = _commander(goal_active=1.0, goal_x=0.0, goal_y=2.0).desired(np.zeros(2), 0.0)
    assert left[0] == pytest.approx(0.0, abs=1e-9)
    assert 0.0 < left[1] <= 0.3 and left[2] == pytest.approx(0.2)

    # A body yawed toward the goal gets a pure forward command.
    facing = _commander(goal_active=1.0, goal_x=0.0, goal_y=2.0).desired(np.zeros(2), np.pi / 2)
    assert facing[0] == pytest.approx(MIN_WALK_SPEED) and abs(facing[1]) < 1e-9


def test_commander_arrival_latch_and_reengage():
    commander = _commander(goal_active=1.0, goal_x=1.0, goal_y=0.0)
    commander.desired(np.zeros(2), 0.0)
    assert commander.config.mode == MODE_WALK and not commander.arrived
    stop = np.array([1.0 - 0.15, 0.0])  # inside 0.8 * tol
    assert np.allclose(commander.desired(stop, 0.0), 0.0)
    assert commander.arrived and commander.config.mode == MODE_STAND
    drift = np.array([1.0 - 0.24, 0.0])  # inside tol + margin
    assert np.allclose(commander.desired(drift, 0.0), 0.0)
    assert commander.arrived
    away = np.array([1.0 - 0.30, 0.0])
    assert commander.desired(away, 0.0)[0] > 0.0
    assert not commander.arrived and commander.config.mode == MODE_WALK


def test_commander_goal_heading_and_walk_mode():
    commander = _commander(goal_active=1.0, goal_x=3.0, goal_y=0.0)
    commander.goal_heading = np.pi / 2
    cmd = commander.desired(np.zeros(2), 0.0)
    assert cmd[2] == pytest.approx(0.2)  # faces the goal heading, not the bearing
    commander.desired(np.array([2.9, 0.0]), 0.0)
    assert commander.arrived
    assert commander.desired(np.array([2.9, 0.0]), 0.0)[2] == pytest.approx(0.2)
    assert commander.desired(np.array([2.9, 0.0]), np.pi / 2)[2] == 0.0

    walk = _commander(mode=MODE_WALK, speed=0.5, heading=np.pi)
    cmd = walk.desired(np.zeros(2), 0.0)
    assert cmd[0] == pytest.approx(-0.5) and abs(cmd[2]) == pytest.approx(0.2)


def test_commander_ramp():
    commander = _commander(mode=MODE_WALK, speed=0.5, heading=0.0)
    first = commander.update(np.zeros(2), 0.0, 0.02)
    assert first[0] == pytest.approx(0.5 * 0.02)


def test_controller_task_targets(model):
    controller = UnitreeRLController(model)
    applied = controller.set_task_targets(
        {"gait.mode": "walk", "gait.speed": 0.15, "gait.goal_x": 3.0, "gait.goal_y": 1.0}
    )
    cfg = controller.gait.config
    assert applied["gait.goal_active"] == 1.0 and cfg.mode == MODE_WALK
    assert controller.commander.cruise == pytest.approx(0.15)
    controller.set_task_targets({"gait.goal_heading": 1.0})
    assert controller.commander.goal_heading == pytest.approx(1.0)
    controller.set_task_targets({"gait.goal_heading": float("nan")})
    assert controller.commander.goal_heading is None
    controller.set_task_targets({"gait.heading": 0.5})
    assert cfg.goal_active == 0.0
    controller.set_task_targets({"gait.goal_x": 1.0})
    controller.set_task_targets({"gait.mode": "stand"})
    assert cfg.goal_active == 0.0 and cfg.mode == MODE_STAND


def test_default_pose_on_floor():
    sim = UnitreeRLSimulation(model_path=SCENE)
    policy = sim.controller.policy
    assert np.allclose(sim.data.qpos[policy.qpos_idx], policy.default_q)
    soles = [
        sim.data.site_xpos[mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_foot_c{i}"), 2]
        for side in ("left", "right")
        for i in range(4)
    ]
    assert min(soles) == pytest.approx(0.0, abs=1e-9)
    assert sim.policy_decimation == 10 and sim.steps_per_control == 2


@pytest.mark.slow
def test_stand_rollout_holds():
    sim = UnitreeRLSimulation(model_path=SCENE)
    sim.run(3.0)
    assert not sim.fell
    assert np.linalg.norm(sim.data.qpos[:2]) < 0.05
    assert sim.data.qpos[2] > 0.7


@pytest.mark.slow
def test_walk_forward_tracks_command():
    sim = UnitreeRLSimulation(model_path=SCENE)
    sim.controller.set_task_targets({"gait.mode": "walk", "gait.speed": 0.5, "gait.heading": 0.0})
    sim.run(3.0)
    x0 = float(sim.data.qpos[0])
    sim.run(8.0)
    assert not sim.fell
    assert (float(sim.data.qpos[0]) - x0) / 5.0 == pytest.approx(0.5, rel=0.2)
    assert abs(yaw_of(sim.data.qpos[3:7])) < 0.2


@pytest.mark.slow
def test_baseline_goto_goal_smoke(tmp_path: Path):
    from loka.harness.pipeline import run_suite
    from loka.harness.config import load_suite_yaml
    from loka.harness.storage import load_episode

    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        yaml.safe_dump(
            {
                "output_dir": str(tmp_path / "out"),
                "seed": 1,
                "num_trials": 1,
                "init_noise": 0.005,
                "log_hz": 20,
                "workers": 1,
                "controller": {"config": "loka/config/g1.yaml", "stack": "legacy_dcm"},
                "baselines": ["unitree_rl"],
                "recording": {"enabled": False, "fps": 5, "width": 64, "height": 48},
                "tasks": [
                    {
                        "name": "goto_goal",
                        "type": "goto_goal",
                        "goal": {"xy": [1.0, 0.0], "speed": 0.15, "radius": 0.2},
                        "command_time_s": 0.0,
                        "perturbation_time_s": 1.0,
                        "timeout_s": 15.0,
                        "perturbations": [
                            {"name": "shove", "kind": "push", "force_n": 50.0, "direction": [0.0, 1.0], "duration_s": 0.1}
                        ],
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
    episode = load_episode(run_dir / "goto_goal__shove__unitree_rl__trial_00")
    assert episode.metadata["outcome"] == "success"
    assert episode.metadata["goal_via"] == "task_targets"
