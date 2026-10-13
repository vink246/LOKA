"""The unitree_rl_lab G1-29dof velocity policy, run on numpy against the MuJoCo plant.

The policy is the rsl_rl actor MLP exported by unitree_rl_lab (IsaacLab). Its
interface comes from the exported ``deploy.yaml``, read the same way as the
upstream C++ deploy (``deploy/include/isaaclab``):

* ``joint_ids_map[i]`` is the SDK motor index of policy joint ``i``.
* ``stiffness`` / ``damping`` are in SDK order; ``default_joint_pos`` and the
  action ``scale`` / ``offset`` are in policy order.
* Each observation term keeps ``history_length`` frames, oldest first. Terms
  are concatenated term by term (IsaacLab ``concatenate_terms``), unless the
  yaml sets ``use_gym_history``. A reset fills every slot with the first frame.
* The action is a joint position target ``raw * scale + offset``. The motor
  side runs ``kp * (q_des - q) - kd * qd`` at the physics rate.

The policy only takes a body-frame velocity command. ``GoalVelocityCommander``
turns the LOKA ``gait.*`` goal and heading targets into that command.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import yaml

from loka.control.gait import (
    GOAL_DEFAULT_SPEED,
    GOAL_HEADING_FREEZE,
    GOAL_MIN_SPEED,
    GOAL_REENGAGE_MARGIN,
    GOAL_SLOW_GAIN,
    GOAL_STOP_FRACTION,
    MODE_NAME_TO_CODE,
    MODE_STAND,
    MODE_WALK,
    GaitConfig,
    wrap_angle,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_DIR = REPO_ROOT / "models" / "unitree_rl" / "g1_29dof_velocity_v0"

#: SDK motor order (``LowCmd_.motor_cmd``) of the 29-DoF G1.
G1_29DOF_SDK_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

#: Sole contact sites; the reset pose puts the lowest of them on the floor.
SOLE_SITES = tuple(f"{side}_foot_c{i}" for side in ("left", "right") for i in range(4))

#: Yaw P gain from heading error to the ``ang_vel_z`` command [1/s].
YAW_GAIN = 1.0
#: Heading error below which a final-heading hold stops turning [rad].
YAW_DEADBAND = 0.05
#: Ramp on the angular velocity command [rad/s^2].
YAW_ACCEL = 0.5
#: Slowest linear command sent while walking [m/s]. The v0 policy stands
#: still below about 0.25 m/s (its velocity reward has std 0.5) and does not
#: turn in place, so a slower approach would never arrive.
MIN_WALK_SPEED = 0.30

GRAVITY_W = np.array([0.0, 0.0, -1.0])


def quat_to_mat(quat_wxyz: np.ndarray) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat_wxyz, dtype=float))
    return mat.reshape(3, 3)


def yaw_of(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = (float(v) for v in quat_wxyz)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def rpy_of(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat_wxyz)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return np.array([roll, pitch, yaw_of(quat_wxyz)])


class NumpyMLP:
    """``w{i}`` (out, in) and ``b{i}`` layers with ELU between them."""

    def __init__(self, path: Path) -> None:
        with np.load(path) as blob:
            n = len([key for key in blob.files if key.startswith("w")])
            self.weights = [blob[f"w{i}"].astype(np.float64) for i in range(n)]
            self.biases = [blob[f"b{i}"].astype(np.float64) for i in range(n)]
        self.input_dim = int(self.weights[0].shape[1])
        self.output_dim = int(self.weights[-1].shape[0])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = np.asarray(x, dtype=np.float64)
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = w @ h + b
            if i < last:
                h = np.where(h > 0.0, h, np.expm1(np.minimum(h, 0.0)))
        return h


@dataclass
class _ObsTerm:
    name: str
    scale: np.ndarray | None
    clip: tuple[float, float] | None
    history: int
    buffer: deque = field(default_factory=deque)

    def process(self, raw: np.ndarray) -> np.ndarray:
        value = np.asarray(raw, dtype=np.float64).copy()
        if self.clip is not None:
            value = np.clip(value, self.clip[0], self.clip[1])
        if self.scale is not None:
            value = value * self.scale
        return value

    def fill(self, value: np.ndarray) -> None:
        self.buffer = deque([value.copy() for _ in range(self.history)], maxlen=self.history)

    def push(self, value: np.ndarray) -> None:
        self.buffer.append(value)


class UnitreeVelocityPolicy:
    """Observation history, MLP and PD law for one exported policy directory."""

    def __init__(self, model: mujoco.MjModel, policy_dir: str | Path | None = None) -> None:
        self.policy_dir = Path(policy_dir) if policy_dir is not None else DEFAULT_POLICY_DIR
        cfg = yaml.safe_load((self.policy_dir / "deploy.yaml").read_text(encoding="utf-8"))
        self.cfg = cfg
        self.step_dt = float(cfg["step_dt"])
        ids_map = np.asarray(cfg["joint_ids_map"], dtype=int)
        n = len(ids_map)
        if n != len(G1_29DOF_SDK_JOINTS):
            raise ValueError(f"{self.policy_dir}: expected 29 joints, got {n}")
        self.ids_map = ids_map

        sdk_qpos, sdk_dof, sdk_act = [], [], []
        for name in G1_29DOF_SDK_JOINTS:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"joint {name!r} is not in the MuJoCo model")
            act = [
                a for a in range(model.nu)
                if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
                and int(model.actuator_trnid[a, 0]) == jid
            ]
            if len(act) != 1:
                raise ValueError(f"joint {name!r} needs exactly one actuator, found {len(act)}")
            sdk_qpos.append(int(model.jnt_qposadr[jid]))
            sdk_dof.append(int(model.jnt_dofadr[jid]))
            sdk_act.append(act[0])
        # Everything below is indexed in policy order.
        self.qpos_idx = np.asarray(sdk_qpos)[ids_map]
        self.dof_idx = np.asarray(sdk_dof)[ids_map]
        self.act_idx = np.asarray(sdk_act)[ids_map]
        self.kp = np.asarray(cfg["stiffness"], dtype=float)[ids_map]
        self.kd = np.asarray(cfg["damping"], dtype=float)[ids_map]
        self.default_q = np.asarray(cfg["default_joint_pos"], dtype=float)

        actions = cfg["actions"]
        if list(actions) != ["JointPositionAction"]:
            raise ValueError(f"only JointPositionAction is supported, got {list(actions)}")
        action = actions["JointPositionAction"]
        if action.get("joint_ids") is not None:
            raise ValueError("JointPositionAction over a joint subset is not supported")
        self.action_scale = np.asarray(action.get("scale") or np.ones(n), dtype=float)
        self.action_offset = np.asarray(action.get("offset") or np.zeros(n), dtype=float)
        clip = action.get("clip")
        self.action_clip = None if clip is None else np.asarray(clip, dtype=float)

        ranges = cfg["commands"]["base_velocity"]["ranges"]
        self.cmd_low = np.array([ranges["lin_vel_x"][0], ranges["lin_vel_y"][0], ranges["ang_vel_z"][0]], dtype=float)
        self.cmd_high = np.array([ranges["lin_vel_x"][1], ranges["lin_vel_y"][1], ranges["ang_vel_z"][1]], dtype=float)

        obs_cfg = dict(cfg["observations"])
        self.gym_history = bool(obs_cfg.pop("use_gym_history", False))
        if obs_cfg.pop("scale_first", False):
            raise ValueError("scale_first observations are not supported")
        self.terms: list[_ObsTerm] = []
        for name, term in obs_cfg.items():
            if name not in self._TERMS:
                raise ValueError(f"observation term {name!r} is not supported")
            scale = term.get("scale")
            clip = term.get("clip")
            self.terms.append(
                _ObsTerm(
                    name=name,
                    scale=None if scale is None else np.asarray(scale, dtype=float),
                    clip=None if clip is None else (float(clip[0]), float(clip[1])),
                    history=max(1, int(term.get("history_length") or 1)),
                )
            )

        self.mlp = NumpyMLP(self.policy_dir / "policy.npz")
        if self.mlp.output_dim != n:
            raise ValueError(f"policy outputs {self.mlp.output_dim} actions for {n} joints")
        self.last_action = np.zeros(n)
        self.q_des = self.default_q.copy()
        self._fresh = True
        dim = len(self.observe(np.zeros(model.nq), np.zeros(model.nv), np.zeros(3), dry=True))
        if dim != self.mlp.input_dim:
            raise ValueError(f"observation is {dim} wide, policy expects {self.mlp.input_dim}")

    _TERMS = (
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos",
        "joint_pos_rel",
        "joint_vel_rel",
        "last_action",
    )

    def reset(self) -> None:
        """Zero the last action. The history refills from the next observation."""
        self.last_action = np.zeros(len(self.ids_map))
        self.q_des = self.default_q.copy()
        self._fresh = True

    def clip_command(self, cmd: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(cmd, dtype=float), self.cmd_low, self.cmd_high)

    def _raw_term(self, name: str, qpos: np.ndarray, qvel: np.ndarray, cmd: np.ndarray) -> np.ndarray:
        if name == "base_ang_vel":
            return np.asarray(qvel[3:6], dtype=float)
        if name == "projected_gravity":
            quat = qpos[3:7]
            if not np.any(quat):
                quat = np.array([1.0, 0.0, 0.0, 0.0])
            return quat_to_mat(quat).T @ GRAVITY_W
        if name == "velocity_commands":
            return self.clip_command(cmd)
        if name == "joint_pos":
            return np.asarray(qpos[self.qpos_idx], dtype=float)
        if name == "joint_pos_rel":
            return np.asarray(qpos[self.qpos_idx], dtype=float) - self.default_q
        if name == "joint_vel_rel":
            return np.asarray(qvel[self.dof_idx], dtype=float)
        if name == "last_action":
            return self.last_action.copy()
        raise KeyError(name)

    def observe(self, qpos: np.ndarray, qvel: np.ndarray, cmd: np.ndarray, *, dry: bool = False) -> np.ndarray:
        """Push one frame into every term history and return the flat observation."""
        frames = [term.process(self._raw_term(term.name, qpos, qvel, cmd)) for term in self.terms]
        if dry:
            return np.concatenate([np.tile(f, term.history) for f, term in zip(frames, self.terms)])
        for term, frame in zip(self.terms, frames):
            if self._fresh:
                term.fill(frame)
            else:
                term.push(frame)
        self._fresh = False
        if self.gym_history:
            history = self.terms[0].history
            return np.concatenate([term.buffer[h] for h in range(history) for term in self.terms])
        return np.concatenate([np.concatenate(list(term.buffer)) for term in self.terms])

    def act(self, qpos: np.ndarray, qvel: np.ndarray, cmd: np.ndarray) -> np.ndarray:
        """One policy tick. Returns the joint position targets in policy order."""
        obs = self.observe(qpos, qvel, cmd)
        raw = self.mlp(obs.astype(np.float32))
        self.last_action = raw
        q_des = raw * self.action_scale + self.action_offset
        if self.action_clip is not None:
            q_des = np.clip(q_des, self.action_clip[:, 0], self.action_clip[:, 1])
        self.q_des = q_des
        return q_des

    def pd_torque(self, model: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        """Motor-side PD on the latest target, written into an ``nu`` vector."""
        q = np.asarray(qpos[self.qpos_idx], dtype=float)
        qd = np.asarray(qvel[self.dof_idx], dtype=float)
        tau = self.kp * (self.q_des - q) - self.kd * qd
        ctrl = np.zeros(model.nu)
        ctrl[self.act_idx] = tau
        limited = model.actuator_ctrllimited.astype(bool)
        ctrl[limited] = np.clip(
            ctrl[limited], model.actuator_ctrlrange[limited, 0], model.actuator_ctrlrange[limited, 1]
        )
        return ctrl

    def place_default_pose(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Set the policy's default joints and put the lowest sole point on z=0."""
        data.qpos[self.qpos_idx] = self.default_q
        data.qvel[:] = 0.0
        mujoco.mj_kinematics(model, data)
        heights = []
        for name in SOLE_SITES:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid >= 0:
                heights.append(float(data.site_xpos[sid, 2]))
        if heights:
            data.qpos[2] -= min(heights)
        mujoco.mj_forward(model, data)


class GoalVelocityCommander:
    """World goal and heading targets to a body-frame ``(vx, vy, wz)`` command.

    It follows the low-level goal rules in ``GaitPlanner._steer_to_goal``:
    approach speed ``min(cruise, GOAL_SLOW_GAIN * (d - tol))`` floored at
    ``GOAL_MIN_SPEED``, a stop inside ``GOAL_STOP_FRACTION * tol`` that latches,
    and a re-engage past ``tol + GOAL_REENGAGE_MARGIN``. Any nonzero linear
    command is then raised to ``min_speed``. The policy turns at most
    ``ang_vel_z`` max (0.2 rad/s for v0) and only while it walks, so the walk
    is holonomic: the velocity points at the goal in the body frame while yaw
    tracks ``goal_heading`` if one is set, else the bearing. With no goal,
    walk mode travels along ``heading`` at ``speed``.
    """

    def __init__(
        self,
        config: GaitConfig,
        low: np.ndarray,
        high: np.ndarray,
        min_speed: float = MIN_WALK_SPEED,
    ) -> None:
        self.config = config
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)
        self.min_speed = float(min_speed)
        self.goal_heading: float | None = None
        self.cruise = GOAL_DEFAULT_SPEED
        self.arrived = False
        self.command = np.zeros(3)
        self.target = np.zeros(3)

    def reset(self) -> None:
        self.arrived = False
        self.command = np.zeros(3)
        self.target = np.zeros(3)

    def _yaw_rate(self, yaw_target: float | None, yaw: float, deadband: float = 0.0) -> float:
        if yaw_target is None:
            return 0.0
        error = wrap_angle(yaw_target - yaw)
        if abs(error) < deadband:
            return 0.0
        return float(YAW_GAIN * error)

    def _fit(self, v_body: np.ndarray) -> np.ndarray:
        """Scale a body-frame xy velocity uniformly into the command box."""
        scale = 1.0
        for axis in (0, 1):
            v = float(v_body[axis])
            if v > self.high[axis] > 0.0:
                scale = min(scale, self.high[axis] / v)
            elif v < self.low[axis] < 0.0:
                scale = min(scale, self.low[axis] / v)
        return v_body * scale

    def desired(self, pelvis_xy: np.ndarray, yaw: float) -> np.ndarray:
        cfg = self.config
        v_world = np.zeros(2)
        yaw_target: float | None = None
        deadband = 0.0
        if float(cfg.goal_active) >= 0.5:
            goal = np.array([float(cfg.goal_x), float(cfg.goal_y)])
            delta = goal - np.asarray(pelvis_xy, dtype=float)[:2]
            dist = float(np.hypot(delta[0], delta[1]))
            tol = float(cfg.goal_tolerance)
            if self.arrived and dist > tol + GOAL_REENGAGE_MARGIN:
                self.arrived = False
            if not self.arrived and dist <= GOAL_STOP_FRACTION * tol:
                self.arrived = True
            if self.arrived:
                cfg.mode = MODE_STAND
                yaw_target = self.goal_heading
                deadband = YAW_DEADBAND
            else:
                cfg.mode = MODE_WALK
                speed = min(max(GOAL_SLOW_GAIN * (dist - tol), GOAL_MIN_SPEED), self.cruise)
                v_world = speed * delta / max(dist, 1e-9)
                if dist > GOAL_HEADING_FREEZE:
                    cfg.heading = float(np.arctan2(delta[1], delta[0]))
                yaw_target = self.goal_heading
                if yaw_target is None and dist > GOAL_HEADING_FREEZE:
                    yaw_target = cfg.heading
        elif float(cfg.mode) >= MODE_WALK:
            heading = float(cfg.heading)
            v_world = float(cfg.speed) * np.array([np.cos(heading), np.sin(heading)])
            yaw_target = heading
        c, s = np.cos(yaw), np.sin(yaw)
        v_body = np.array([c * v_world[0] + s * v_world[1], -s * v_world[0] + c * v_world[1]])
        norm = float(np.linalg.norm(v_body))
        if 1e-6 < norm < self.min_speed:
            v_body *= self.min_speed / norm
        v_body = self._fit(v_body)
        wz = float(np.clip(self._yaw_rate(yaw_target, yaw, deadband), self.low[2], self.high[2]))
        return np.array([v_body[0], v_body[1], wz])

    def update(self, pelvis_xy: np.ndarray, yaw: float, dt: float) -> np.ndarray:
        """Advance the ramped command by ``dt`` toward the current target."""
        self.target = self.desired(pelvis_xy, yaw)
        step = self.target[:2] - self.command[:2]
        limit = float(self.config.walk_accel) * dt
        norm = float(np.linalg.norm(step))
        if norm > limit:
            step *= limit / norm
        self.command[:2] += step
        self.command[2] += float(np.clip(self.target[2] - self.command[2], -YAW_ACCEL * dt, YAW_ACCEL * dt))
        return self.command.copy()


@dataclass
class UnitreeRLTelemetry:
    torque: np.ndarray
    rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))
    com_error: np.ndarray = field(default_factory=lambda: np.zeros(3))
    contact_forces: np.ndarray | None = None
    command: np.ndarray = field(default_factory=lambda: np.zeros(3))
    solve_ms: float = 0.0
    walking: bool = False


class _GaitState:
    """The slice of ``GaitPlanner`` that harness logging and goal helpers read."""

    def __init__(self, commander: GoalVelocityCommander) -> None:
        self._commander = commander
        self.config = commander.config

    @property
    def goal_arrived(self) -> bool:
        return bool(self._commander.arrived)

    def wants_walk(self) -> bool:
        return float(self.config.mode) >= MODE_WALK


class UnitreeRLController:
    """The RL policy behind the ``gait.*`` Task_Targets surface of the MPC stack.

    Accepted targets: ``gait.mode`` (stand / walk), ``gait.speed``,
    ``gait.heading``, ``gait.goal_x``, ``gait.goal_y``, ``gait.goal_active``,
    ``gait.goal_tolerance`` and ``gait.goal_heading`` (final yaw at the goal;
    NaN or None clears it).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        policy_dir: str | Path | None = None,
        min_speed: float = MIN_WALK_SPEED,
    ) -> None:
        self.model = model
        self.policy = UnitreeVelocityPolicy(model, policy_dir)
        config = GaitConfig()
        config.goal_tolerance = 0.15
        self.commander = GoalVelocityCommander(
            config, self.policy.cmd_low, self.policy.cmd_high, min_speed=min_speed
        )
        self.gait = _GaitState(self.commander)
        self.telemetry = UnitreeRLTelemetry(torque=np.zeros(model.nu))

    def reset(self) -> None:
        config = GaitConfig()
        config.goal_tolerance = 0.15
        self.commander.config = config
        self.gait.config = config
        self.commander.goal_heading = None
        self.commander.cruise = GOAL_DEFAULT_SPEED
        self.commander.reset()
        self.policy.reset()
        self.telemetry = UnitreeRLTelemetry(torque=np.zeros(self.model.nu))

    def place_default_pose(self, data: mujoco.MjData) -> None:
        self.policy.place_default_pose(self.model, data)

    def set_task_targets(self, updates: dict) -> dict:
        cfg = self.commander.config
        applied: dict = {}
        speed_max = float(self.policy.cmd_high[0])
        for name, value in updates.items():
            if name == "gait.mode":
                code = MODE_NAME_TO_CODE.get(str(value).strip().lower()) if isinstance(value, str) else float(value)
                if code is None:
                    print(f"     * [WARN] unknown gait.mode {value!r}")
                    continue
                cfg.mode = MODE_STAND if float(code) < MODE_WALK else MODE_WALK
                applied[name] = cfg.mode
            elif name == "gait.speed":
                cfg.speed = float(np.clip(float(value), 0.0, speed_max))
                applied[name] = cfg.speed
            elif name == "gait.heading":
                cfg.heading = wrap_angle(float(value))
                applied[name] = cfg.heading
            elif name in ("gait.goal_x", "gait.goal_y"):
                setattr(cfg, name.split(".", 1)[1], float(np.clip(float(value), -20.0, 20.0)))
                applied[name] = float(getattr(cfg, name.split(".", 1)[1]))
            elif name == "gait.goal_active":
                cfg.goal_active = 1.0 if float(value) >= 0.5 else 0.0
                applied[name] = cfg.goal_active
            elif name == "gait.goal_tolerance":
                cfg.goal_tolerance = float(np.clip(float(value), 0.05, 0.50))
                applied[name] = cfg.goal_tolerance
            elif name == "gait.goal_heading":
                if value is None or not np.isfinite(float(value)):
                    self.commander.goal_heading = None
                    applied[name] = None
                else:
                    self.commander.goal_heading = wrap_angle(float(value))
                    applied[name] = self.commander.goal_heading
            else:
                print(f"     * [WARN] {name} is not a unitree_rl target; ignored")
        moved = "gait.goal_x" in applied or "gait.goal_y" in applied
        if applied.get("gait.mode") == MODE_STAND:
            cfg.goal_active = 0.0
        elif "gait.goal_active" not in applied:
            if moved:
                cfg.goal_active = 1.0
            elif "gait.heading" in applied:
                cfg.goal_active = 0.0
        if float(cfg.goal_active) >= 0.5 and (moved or "gait.goal_active" in applied):
            self.commander.arrived = False
            self.commander.cruise = float(cfg.speed) if cfg.speed > 1e-4 else GOAL_DEFAULT_SPEED
            cfg.mode = MODE_WALK
            applied["gait.goal_active"] = 1.0
        return applied

    def policy_step(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        """Update the command and run the MLP once (call every ``policy.step_dt``)."""
        t0 = time.perf_counter()
        quat = np.asarray(qpos[3:7], dtype=float)
        cmd = self.commander.update(np.asarray(qpos[:2], dtype=float), yaw_of(quat), self.policy.step_dt)
        self.policy.act(qpos, qvel, cmd)
        self.telemetry.command = self.policy.clip_command(cmd)
        self.telemetry.rpy = rpy_of(quat)
        self.telemetry.walking = bool(np.linalg.norm(cmd) > 1e-3)
        self.telemetry.solve_ms = (time.perf_counter() - t0) * 1e3
        return cmd

    def compute_torque(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        """PD on the latest target. Call once per physics substep."""
        torque = self.policy.pd_torque(self.model, qpos, qvel)
        self.telemetry.torque = torque
        return torque
