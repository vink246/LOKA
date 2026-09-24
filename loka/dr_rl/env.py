"""Gymnasium wrapper around the LOKA Walker MuJoCo plant."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, SupportsFloat

import mujoco
import numpy as np

from loka.dr_rl.config import load_dr_config, walker_xml_path
from loka.dr_rl.observation import (
    build_observation,
    gait_phase_features,
    observation_size,
)
from loka.dr_rl.randomize import (
    apply_reset_randomization,
    capture_nominal_dynamics,
    restore_nominal_dynamics,
    sample_reset_randomization,
    sample_training_push,
)
from loka.walker_suite.faults import PlantFaults
from loka.walker_suite.outcomes import has_fallen, pitch, pos_x, world_height


def _excess_sq(value: float, target: float, deadband: float) -> float:
    return max(abs(float(value) - float(target)) - float(deadband), 0.0) ** 2


def feet_gait_match(
    contacts: np.ndarray,
    gait_time: float,
    *,
    period: float = 0.8,
    threshold: float = 0.55,
    offsets: tuple[float, ...] = (0.0, 0.5),
) -> float:
    """Unitree ``feet_gait``: +1 per foot whose contact matches the clock.

    Stance when ``(t/T + offset) mod 1 < threshold``. With T=0.8, offset
    [0, 0.5], threshold 0.55 there is always at least one stance foot (a
    walk, not a run), and t=0 is double support so standing is a match.
    """
    contacts = np.asarray(contacts, dtype=bool).reshape(-1)
    n = min(int(contacts.size), len(offsets))
    if n == 0:
        return 0.0
    frac = (float(gait_time) / max(float(period), 1e-6)) % 1.0
    match = 0.0
    for i in range(n):
        phase = (frac + float(offsets[i])) % 1.0
        want_stance = phase < float(threshold)
        if want_stance == bool(contacts[i]):
            match += 1.0
    return match


def sine_walk_targets(
    gait_time: float,
    *,
    period: float = 0.8,
    threshold: float = 0.55,
    offsets: tuple[float, ...] = (0.0, 0.5),
    hip_bias: float = 0.35,
    hip_amp: float = 0.40,
    knee_stance: float = -0.15,
    knee_amp: float = 0.80,
    ankle_stance: float = 0.0,
    ankle_amp: float = 0.20,
) -> np.ndarray:
    """Analytic walk: ``(n_feet, 3)`` hip / knee / ankle targets in radians.

    Hip is a cosine on the same clock as ``feet_gait`` (flexed at heel-strike).
    Knee/ankle stay near stance values on the ground and flex mid-swing.
    Left is offset 0.5 so this *is* the L/R mirror — do not also tax
    ``|q_left - q_right|`` at the same time (that is a hop/stand).
    """
    n = len(offsets)
    targets = np.zeros((n, 3), dtype=np.float64)
    frac = (float(gait_time) / max(float(period), 1e-6)) % 1.0
    swing = max(1.0 - float(threshold), 1e-6)
    for i, offset in enumerate(offsets):
        phase = (frac + float(offset)) % 1.0
        targets[i, 0] = float(hip_bias) + float(hip_amp) * float(
            np.cos(2.0 * np.pi * phase)
        )
        if phase < float(threshold):
            targets[i, 1] = float(knee_stance)
            targets[i, 2] = float(ankle_stance)
        else:
            s = (phase - float(threshold)) / swing
            lift = float(np.sin(np.pi * s))
            targets[i, 1] = float(knee_stance) - float(knee_amp) * lift
            targets[i, 2] = float(ankle_stance) + float(ankle_amp) * lift
    return targets


def pose_symmetry_errors(
    q_legs: np.ndarray, targets: np.ndarray
) -> tuple[float, float]:
    """Per-leg L2 tracking and |err_R − err_L| so one leg cannot do all the work."""
    q_legs = np.asarray(q_legs, dtype=np.float64).reshape(-1, 3)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1, 3)
    n = min(q_legs.shape[0], targets.shape[0])
    if n == 0:
        return 0.0, 0.0
    leg_err = np.sum((q_legs[:n] - targets[:n]) ** 2, axis=1)
    pose = float(np.sum(leg_err))
    sym = float(abs(leg_err[0] - leg_err[1])) if n >= 2 else 0.0
    return pose, sym


def cycle_lead_error(hip_right: np.ndarray, hip_left: np.ndarray) -> float:
    """``(mean q_hip_L − mean q_hip_R)²`` over a window.

    A scissor averages both hips to the same bias; a compass keeps one
    flexed and one extended. This is not same-time ``|q_L − q_R|``.
    """
    right = np.asarray(hip_right, dtype=np.float64).reshape(-1)
    left = np.asarray(hip_left, dtype=np.float64).reshape(-1)
    n = min(right.size, left.size)
    if n == 0:
        return 0.0
    return float((float(np.mean(left[:n])) - float(np.mean(right[:n]))) ** 2)


def cycle_hip_rom_error(
    hip_right: np.ndarray,
    hip_left: np.ndarray,
    *,
    rom_goal: float,
    ready: bool = True,
) -> float:
    """``Σ max(0, rom_goal − rom)²`` so a locked hip cannot hide behind pose."""
    if not ready or float(rom_goal) <= 0.0:
        return 0.0
    cost = 0.0
    for hip in (hip_right, hip_left):
        q = np.asarray(hip, dtype=np.float64).reshape(-1)
        if q.size == 0:
            continue
        rom = float(np.max(q) - np.min(q))
        cost += max(float(rom_goal) - rom, 0.0) ** 2
    return float(cost)


def locomotion_reward(
    vx: float,
    action: np.ndarray,
    *,
    speed_goal: float,
    forward_weight: float,
    healthy: bool,
    healthy_bonus: float,
    pitch_val: float,
    height: float,
    height_target: float,
    ctrl_cost_weight: float,
    pitch_cost_weight: float,
    height_cost_weight: float,
    vz: float = 0.0,
    pitch_rate: float = 0.0,
    prev_action: np.ndarray | None = None,
    n_foot_contacts: int = 2,
    joint_limit_violation: float = 0.0,
    vz_cost_weight: float = 0.0,
    pitch_rate_weight: float = 0.0,
    action_rate_weight: float = 0.0,
    flight_cost_weight: float = 0.0,
    joint_limit_weight: float = 0.0,
    overspeed_weight: float = 0.0,
    healthy_min_speed: float = 0.0,
    height_deadband: float = 0.0,
    pitch_deadband: float = 0.0,
    foot_slip: float = 0.0,
    slip_weight: float = 0.0,
    stride: float = 0.0,
    stride_weight: float = 0.0,
    clearance: float = 1.0,
    clearance_weight: float = 0.0,
    gait_match: float = 0.0,
    gait_weight: float = 0.0,
    pose_err: float = 0.0,
    pose_weight: float = 0.0,
    sym_err: float = 0.0,
    sym_weight: float = 0.0,
    lead_err: float = 0.0,
    lead_weight: float = 0.0,
    hip_rom_err: float = 0.0,
    hip_rom_weight: float = 0.0,
) -> tuple[float, dict[str, float]]:
    """Gymnasium Walker2d base plus extras to prefer a 1 m/s scissor walk.

    Base (Walker2d-v5 / SB3 Zoo): ``vx + alive - 0.001 ||a||^2``. Extras:
    overspeed tax, height/pitch deadbands, Isaac feet_slide, Isaac-style
    air-time on a supported landing, Unitree swing clearance, Unitree
    contact-clock match (gated by forward speed so marching in place
    cannot collect it). Sine-walk pose, L/R tracking balance, cycle-mean
    lead, and per-hip ROM are *not* speed-gated — a compass cannot hide
    by standing or creeping. Flight cost.
    """
    overspeed = max(float(vx) - float(speed_goal), 0.0)
    forward = float(forward_weight) * float(vx) - float(overspeed_weight) * (
        overspeed**2
    )
    moving = float(vx) >= float(healthy_min_speed)
    healthy_r = float(healthy_bonus) if (healthy and moving) else 0.0
    ctrl_cost = float(ctrl_cost_weight) * float(np.sum(np.square(action)))
    pitch_cost = float(pitch_cost_weight) * _excess_sq(pitch_val, 0.0, pitch_deadband)
    height_cost = float(height_cost_weight) * _excess_sq(
        height, height_target, height_deadband
    )
    vz_cost = float(vz_cost_weight) * float(vz**2)
    pitch_rate_cost = float(pitch_rate_weight) * float(pitch_rate**2)
    if prev_action is None:
        action_rate_cost = 0.0
    else:
        delta = np.asarray(action, dtype=np.float64) - np.asarray(
            prev_action, dtype=np.float64
        )
        action_rate_cost = float(action_rate_weight) * float(np.sum(np.square(delta)))
    flight_cost = float(flight_cost_weight) if int(n_foot_contacts) <= 0 else 0.0
    limit_cost = float(joint_limit_weight) * float(joint_limit_violation)
    slip_cost = float(slip_weight) * max(float(foot_slip), 0.0)
    stride_r = float(stride_weight) * max(float(stride), 0.0)
    clearance_r = float(clearance_weight) * float(np.clip(clearance, 0.0, 1.0))
    vx_scale = min(max(float(vx) / max(float(speed_goal), 1e-6), 0.0), 1.0)
    gait_r = float(gait_weight) * max(float(gait_match), 0.0) * vx_scale
    pose_cost = float(pose_weight) * max(float(pose_err), 0.0)
    sym_cost = float(sym_weight) * max(float(sym_err), 0.0)
    lead_cost = float(lead_weight) * max(float(lead_err), 0.0)
    hip_rom_cost = float(hip_rom_weight) * max(float(hip_rom_err), 0.0)
    reward = (
        forward
        + healthy_r
        + stride_r
        + clearance_r
        + gait_r
        - ctrl_cost
        - pitch_cost
        - height_cost
        - vz_cost
        - pitch_rate_cost
        - action_rate_cost
        - flight_cost
        - limit_cost
        - slip_cost
        - pose_cost
        - sym_cost
        - lead_cost
        - hip_rom_cost
    )
    return float(reward), {
        "reward_forward": forward,
        "reward_healthy": healthy_r,
        "reward_ctrl": -ctrl_cost,
        "reward_pitch": -pitch_cost,
        "reward_height": -height_cost,
        "reward_vz": -vz_cost,
        "reward_pitch_rate": -pitch_rate_cost,
        "reward_action_rate": -action_rate_cost,
        "reward_flight": -flight_cost,
        "reward_joint_limit": -limit_cost,
        "reward_slip": -slip_cost,
        "reward_stride": stride_r,
        "reward_clearance": clearance_r,
        "reward_gait": gait_r,
        "reward_pose": -pose_cost,
        "reward_sym": -sym_cost,
        "reward_lead": -lead_cost,
        "reward_hip_rom": -hip_rom_cost,
        "vel_x": float(vx),
        "n_foot_contacts": float(n_foot_contacts),
    }

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "loka.dr_rl requires gymnasium. Install with: pip install gymnasium"
    ) from exc


def _as_range(value: Any) -> tuple[float, float]:
    lo, hi = value
    return float(lo), float(hi)


class LokaWalkerDREnv(gym.Env):
    """Torque-controlled Walker with optional in-distribution domain randomization.

    Physics is native MuJoCo (`mujoco.mj_step`) on `models/walker/task.xml`.
    One env step is `frame_skip` simulation steps (default 0.01 s).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        xml_path: str | Path | None = None,
        domain_randomize: bool | None = None,
        observation_noise: bool | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.config = config if config is not None else load_dr_config()
        self.xml_path = Path(xml_path) if xml_path else walker_xml_path(self.config)
        if not self.xml_path.is_file():
            raise FileNotFoundError(f"Walker XML not found: {self.xml_path}")

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.nominal = capture_nominal_dynamics(self.model)
        self.plant = PlantFaults(self.model, self.data)

        self.frame_skip = int(self.config.get("frame_skip", 4))
        self.max_episode_steps = int(self.config.get("max_episode_steps", 1000))
        self.speed_goal = float(self.config.get("speed_goal", 1.0))
        self.reset_noise_scale = float(self.config.get("reset_noise_scale", 5e-3))
        self.dt = float(self.model.opt.timestep) * self.frame_skip

        dr_cfg = self.config.get("domain_randomization") or {}
        self.dr_cfg = dr_cfg
        if domain_randomize is None:
            domain_randomize = bool(dr_cfg.get("enabled", True))
        self.domain_randomize = bool(domain_randomize)

        obs_cfg = self.config.get("observation") or {}
        self.include_last_action = bool(obs_cfg.get("include_last_action", True))
        self.include_gait_phase = bool(obs_cfg.get("include_gait_phase", True))
        if observation_noise is None:
            observation_noise = self.domain_randomize
        self.observation_noise = bool(observation_noise)
        self.qpos_noise = float(obs_cfg.get("qpos_noise", 0.0))
        self.qvel_noise = float(obs_cfg.get("qvel_noise", 0.0))

        reward_cfg = self.config.get("reward") or {}
        self.forward_weight = float(reward_cfg.get("forward_weight", 1.0))
        self.overspeed_weight = float(reward_cfg.get("overspeed_weight", 3.0))
        self.healthy_bonus = float(reward_cfg.get("healthy_bonus", 1.0))
        self.healthy_min_speed = float(reward_cfg.get("healthy_min_speed", 0.0))
        self.ctrl_cost_weight = float(reward_cfg.get("ctrl_cost", 0.001))
        self.action_rate_weight = float(reward_cfg.get("action_rate", 0.001))
        self.pitch_cost_weight = float(reward_cfg.get("pitch_cost", 1.0))
        self.pitch_rate_weight = float(reward_cfg.get("pitch_rate", 0.05))
        self.pitch_deadband = float(reward_cfg.get("pitch_deadband", 0.25))
        self.height_target = float(reward_cfg.get("height_target", 1.2))
        self.height_cost_weight = float(reward_cfg.get("height_cost", 4.0))
        self.height_deadband = float(reward_cfg.get("height_deadband", 0.10))
        self.vz_cost_weight = float(reward_cfg.get("vz_cost", 0.0))
        self.flight_cost_weight = float(reward_cfg.get("flight_cost", 0.25))
        self.slip_weight = float(reward_cfg.get("slip_weight", 0.25))
        self.slip_deadband = float(reward_cfg.get("slip_deadband", 0.15))
        self.stride_weight = float(reward_cfg.get("stride_weight", 2.0))
        self.min_stride = float(reward_cfg.get("min_stride", 0.08))
        self.stride_cap = float(reward_cfg.get("stride_cap", 0.40))
        self.clearance_weight = float(reward_cfg.get("clearance_weight", 0.5))
        self.clearance_target = float(reward_cfg.get("clearance_target", 0.10))
        self.clearance_std = float(reward_cfg.get("clearance_std", 0.05))
        self.clearance_tanh = float(reward_cfg.get("clearance_tanh", 2.0))
        self.gait_weight = float(reward_cfg.get("gait_weight", 0.5))
        self.gait_period = float(reward_cfg.get("gait_period", 0.8))
        self.gait_threshold = float(reward_cfg.get("gait_threshold", 0.55))
        raw_offsets = reward_cfg.get("gait_offsets", [0.0, 0.5])
        self.gait_offsets = tuple(float(v) for v in raw_offsets)
        self.pose_weight = float(reward_cfg.get("pose_weight", 2.0))
        self.sym_weight = float(reward_cfg.get("sym_weight", 0.5))
        self.lead_weight = float(reward_cfg.get("lead_weight", 1.0))
        self.hip_rom_weight = float(reward_cfg.get("hip_rom_weight", 1.0))
        self.hip_bias = float(reward_cfg.get("hip_bias", 0.35))
        self.hip_amp = float(reward_cfg.get("hip_amp", 0.40))
        self.hip_rom_goal = float(reward_cfg.get("hip_rom_goal", 2.0 * self.hip_amp))
        self.knee_stance = float(reward_cfg.get("knee_stance", -0.15))
        self.knee_amp = float(reward_cfg.get("knee_amp", 0.80))
        self.ankle_stance = float(reward_cfg.get("ankle_stance", 0.0))
        self.ankle_amp = float(reward_cfg.get("ankle_amp", 0.20))
        self.joint_limit_weight = float(reward_cfg.get("joint_limit", 0.5))
        self.joint_limit_margin = float(reward_cfg.get("joint_limit_margin", 0.1))
        self.contact_dist = float(reward_cfg.get("contact_dist", 0.002))

        healthy_cfg = self.config.get("healthy") or {}
        self.height_range = _as_range(healthy_cfg.get("height_range", [0.8, 2.0]))
        self.pitch_abs = float(healthy_cfg.get("pitch_abs", 1.0))
        # Gym Walker2d-v5 ends the episode when unhealthy. The gait harness
        # keeps this false (suite fall only) so a lunge is not a 0.75 s cut.
        self.terminate_when_unhealthy = bool(
            healthy_cfg.get("terminate_when_unhealthy", False)
        )

        self._foot_geom_ids = []
        self._foot_body_ids = []
        for name in ("right_foot", "left_foot"):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self._foot_geom_ids.append(gid)
                self._foot_body_ids.append(int(self.model.geom_bodyid[gid]))
        self._foot_geom_ids = tuple(self._foot_geom_ids)
        self._foot_body_ids = tuple(self._foot_body_ids)
        self._leg_qpos_adr: list[tuple[int, int, int]] = []
        for side in ("right", "left"):
            addrs: list[int] = []
            for joint in ("hip", "knee", "ankle"):
                jid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{joint}"
                )
                addrs.append(
                    int(self.model.jnt_qposadr[jid]) if jid >= 0 else -1
                )
            self._leg_qpos_adr.append((addrs[0], addrs[1], addrs[2]))
        n_feet = len(self._foot_geom_ids)
        self._foot_air_time = np.zeros(n_feet, dtype=np.float64)
        self._swing_supported = np.zeros(n_feet, dtype=bool)
        self._prev_contacts = np.ones(n_feet, dtype=bool)

        n_obs = observation_size(
            self.model,
            include_last_action=self.include_last_action,
            include_gait_phase=self.include_gait_phase,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )

        self.render_mode = render_mode
        self._last_action = np.zeros(self.model.nu, dtype=np.float32)
        self._elapsed_steps = 0
        self._gait_time = 0.0
        hip_window = max(int(round(self.gait_period / self.dt)), 1)
        self._hip_hist_r: deque[float] = deque(maxlen=hip_window)
        self._hip_hist_l: deque[float] = deque(maxlen=hip_window)
        self._push_event = None
        self._push_injected = False
        self._np_random: np.random.Generator | None = None
        self._renderer = None

    def _obs(self) -> np.ndarray:
        noise_rng = self._np_random if self.observation_noise else None
        phase = None
        if self.include_gait_phase:
            phase = gait_phase_features(self._gait_time, self.gait_period)
        return build_observation(
            self.data,
            self._last_action,
            include_last_action=self.include_last_action,
            noise_rng=noise_rng,
            qpos_noise=self.qpos_noise if self.observation_noise else 0.0,
            qvel_noise=self.qvel_noise if self.observation_noise else 0.0,
            gait_phase=phase,
        )

    def is_healthy(self) -> bool:
        height = world_height(self.data)
        return (
            self.height_range[0] <= height <= self.height_range[1]
            and abs(pitch(self.data)) <= self.pitch_abs
        )

    def set_domain_randomize(self, enabled: bool) -> None:
        """Toggle reset-time DR (used by the nominal-warmup curriculum)."""
        self.domain_randomize = bool(enabled)
        self.observation_noise = bool(enabled)

    def _foot_contact_mask(self) -> np.ndarray:
        n_feet = len(self._foot_geom_ids)
        mask = np.zeros(n_feet, dtype=bool)
        if n_feet == 0:
            return mask
        geom_to_idx = {gid: i for i, gid in enumerate(self._foot_geom_ids)}
        for i in range(int(self.data.ncon)):
            contact = self.data.contact[i]
            if float(contact.dist) > self.contact_dist:
                continue
            for geom in (int(contact.geom1), int(contact.geom2)):
                idx = geom_to_idx.get(geom)
                if idx is not None:
                    mask[idx] = True
        return mask

    def _stance_feet(self) -> int:
        return int(np.count_nonzero(self._foot_contact_mask()))

    def _foot_slip(self, contacts: np.ndarray) -> float:
        """Isaac Lab ``feet_slide``, but at the *contact point* (not foot COM).

        Body COM speed over-counts ankle rotation and made any stance look
        like skating, so the policy learned a 0.4 s dive instead of a gait.
        """
        if not self._foot_geom_ids:
            return 0.0
        geom_to_idx = {gid: i for i, gid in enumerate(self._foot_geom_ids)}
        slip = 0.0
        counted = set()
        deadband = self.slip_deadband
        for i in range(int(self.data.ncon)):
            contact = self.data.contact[i]
            if float(contact.dist) > self.contact_dist:
                continue
            for geom in (int(contact.geom1), int(contact.geom2)):
                idx = geom_to_idx.get(geom)
                if idx is None or idx in counted or not bool(contacts[idx]):
                    continue
                counted.add(idx)
                body = int(self._foot_body_ids[idx])
                cvel = np.asarray(self.data.cvel[body], dtype=np.float64).reshape(-1)
                omega = cvel[:3]
                v_com = cvel[3:6]
                com = np.asarray(self.data.xpos[body], dtype=np.float64)
                pos = np.asarray(contact.pos, dtype=np.float64)
                v_contact = v_com + np.cross(omega, pos - com)
                speed = float(abs(v_contact[0]))
                slip += max(speed - deadband, 0.0)
        return slip

    def _foot_clearance(self) -> float:
        """Unitree G1 ``foot_clearance_reward`` (exp kernel, no gait clock).

        ``exp(-sum_i (z_i - h)^2 tanh(k |v_x,i|) / std)``. A planted foot
        with ~0 speed contributes nothing; a swinging foot scores when its
        body height is near ``clearance_target``. In ``(0, 1]``.
        """
        if not self._foot_body_ids:
            return 1.0
        err = 0.0
        target = self.clearance_target
        k = self.clearance_tanh
        std = max(self.clearance_std, 1e-6)
        for body in self._foot_body_ids:
            z = float(self.data.xpos[body][2])
            cvel = np.asarray(self.data.cvel[body], dtype=np.float64).reshape(-1)
            speed = float(abs(cvel[3])) if cvel.size > 3 else 0.0
            err += (z - target) ** 2 * float(np.tanh(k * speed))
        return float(np.exp(-err / std))

    def _stride_on_touchdown(self, contacts: np.ndarray) -> float:
        """Isaac ``feet_air_time``: pay swing duration when a foot lands.

        A stride counts only if the other foot was on the ground at some
        point during the swing (walk), not a two-foot leap. Duration is
        clipped so a late tap cannot jackpot.
        """
        contacts = np.asarray(contacts, dtype=bool).reshape(-1)
        n = contacts.size
        if n == 0:
            return 0.0
        prev = np.asarray(self._prev_contacts, dtype=bool).reshape(-1)
        if prev.size != n:
            prev = np.ones(n, dtype=bool)
        just_landed = contacts & ~prev
        stride = 0.0
        for i, landed in enumerate(just_landed):
            if not landed:
                continue
            air = float(self._foot_air_time[i])
            if air < self.min_stride or not bool(self._swing_supported[i]):
                continue
            stride += min(air, self.stride_cap)
        for i, in_contact in enumerate(contacts):
            if in_contact:
                self._foot_air_time[i] = 0.0
                self._swing_supported[i] = False
            else:
                self._foot_air_time[i] += self.dt
                other_down = False
                for j, other in enumerate(contacts):
                    if j != i and other:
                        other_down = True
                        break
                if other_down:
                    self._swing_supported[i] = True
        self._prev_contacts = np.array(contacts, dtype=bool, copy=True)
        return stride

    def _joint_limit_violation(self) -> float:
        cost = 0.0
        margin = self.joint_limit_margin
        for j in range(self.model.njnt):
            if not int(self.model.jnt_limited[j]):
                continue
            q = float(self.data.qpos[int(self.model.jnt_qposadr[j])])
            lo, hi = (float(v) for v in self.model.jnt_range[j])
            if q < lo + margin:
                cost += (lo + margin - q) ** 2
            elif q > hi - margin:
                cost += (q - (hi - margin)) ** 2
        return cost

    def _leg_q(self) -> np.ndarray:
        q = np.zeros((len(self._leg_qpos_adr), 3), dtype=np.float64)
        for i, addrs in enumerate(self._leg_qpos_adr):
            for k, adr in enumerate(addrs):
                if adr >= 0:
                    q[i, k] = float(self.data.qpos[adr])
        return q

    def _pose_symmetry(self) -> tuple[float, float]:
        targets = sine_walk_targets(
            self._gait_time,
            period=self.gait_period,
            threshold=self.gait_threshold,
            offsets=self.gait_offsets,
            hip_bias=self.hip_bias,
            hip_amp=self.hip_amp,
            knee_stance=self.knee_stance,
            knee_amp=self.knee_amp,
            ankle_stance=self.ankle_stance,
            ankle_amp=self.ankle_amp,
        )
        return pose_symmetry_errors(self._leg_q(), targets)

    def _cycle_style(self) -> tuple[float, float]:
        legs = self._leg_q()
        if legs.shape[0] < 2:
            return 0.0, 0.0
        self._hip_hist_r.append(float(legs[0, 0]))
        self._hip_hist_l.append(float(legs[1, 0]))
        right = np.fromiter(self._hip_hist_r, dtype=np.float64)
        left = np.fromiter(self._hip_hist_l, dtype=np.float64)
        ready = len(self._hip_hist_r) >= (self._hip_hist_r.maxlen or 1)
        return (
            cycle_lead_error(right, left),
            cycle_hip_rom_error(
                right, left, rom_goal=self.hip_rom_goal, ready=ready
            ),
        )

    def _reward(
        self,
        action: np.ndarray,
        x_before: float,
        prev_action: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        vx = (pos_x(self.data) - x_before) / self.dt
        qvel = self.data.qvel
        contacts = self._foot_contact_mask()
        pose_err, sym_err = self._pose_symmetry()
        lead_err, hip_rom_err = self._cycle_style()
        return locomotion_reward(
            vx,
            action,
            speed_goal=self.speed_goal,
            forward_weight=self.forward_weight,
            overspeed_weight=self.overspeed_weight,
            healthy=self.is_healthy(),
            healthy_bonus=self.healthy_bonus,
            healthy_min_speed=self.healthy_min_speed,
            pitch_val=pitch(self.data),
            height=world_height(self.data),
            height_target=self.height_target,
            height_deadband=self.height_deadband,
            pitch_deadband=self.pitch_deadband,
            ctrl_cost_weight=self.ctrl_cost_weight,
            pitch_cost_weight=self.pitch_cost_weight,
            height_cost_weight=self.height_cost_weight,
            vz=float(qvel[0]) if qvel.size else 0.0,
            pitch_rate=float(qvel[2]) if qvel.size > 2 else 0.0,
            prev_action=prev_action,
            n_foot_contacts=int(np.count_nonzero(contacts)),
            joint_limit_violation=self._joint_limit_violation(),
            vz_cost_weight=self.vz_cost_weight,
            pitch_rate_weight=self.pitch_rate_weight,
            action_rate_weight=self.action_rate_weight,
            flight_cost_weight=self.flight_cost_weight,
            joint_limit_weight=self.joint_limit_weight,
            foot_slip=self._foot_slip(contacts),
            slip_weight=self.slip_weight,
            stride=self._stride_on_touchdown(contacts),
            stride_weight=self.stride_weight,
            clearance=self._foot_clearance(),
            clearance_weight=self.clearance_weight,
            gait_match=feet_gait_match(
                contacts,
                self._gait_time,
                period=self.gait_period,
                threshold=self.gait_threshold,
                offsets=self.gait_offsets,
            ),
            gait_weight=self.gait_weight,
            pose_err=pose_err,
            pose_weight=self.pose_weight,
            sym_err=sym_err,
            sym_weight=self.sym_weight,
            lead_err=lead_err,
            lead_weight=self.lead_weight,
            hip_rom_err=hip_rom_err,
            hip_rom_weight=self.hip_rom_weight,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._np_random = self.np_random
        options = options or {}

        restore = not bool(options.get("keep_plant", False))
        if restore:
            restore_nominal_dynamics(self.model, self.data, self.nominal)

        use_dr = self.domain_randomize and not bool(options.get("disable_dr", False))
        sample = None
        if use_dr:
            sample = sample_reset_randomization(self.np_random, self.model, self.dr_cfg)
            apply_reset_randomization(self.model, self.data, self.nominal, sample)

        mujoco.mj_resetData(self.model, self.data)
        noise = self.reset_noise_scale
        self.data.qpos[:] = self.model.qpos0 + self.np_random.uniform(
            -noise, noise, size=self.model.nq
        )
        self.data.qvel[:] = self.np_random.uniform(-noise, noise, size=self.model.nv)
        mujoco.mj_forward(self.model, self.data)

        self.plant = PlantFaults(self.model, self.data)
        self._last_action = np.zeros(self.model.nu, dtype=np.float32)
        self._elapsed_steps = 0
        self._gait_time = 0.0
        self._hip_hist_r.clear()
        self._hip_hist_l.clear()
        self._push_injected = False
        self._push_event = None
        self._foot_air_time[:] = 0.0
        self._swing_supported[:] = False
        self._prev_contacts = self._foot_contact_mask()
        if use_dr:
            self._push_event = sample_training_push(
                self.np_random, self.model, self.dr_cfg
            )

        info: dict[str, Any] = {
            "pos_x": pos_x(self.data),
            "height": world_height(self.data),
            "pitch": pitch(self.data),
            "domain_randomize": use_dr,
        }
        if sample is not None:
            info["randomization"] = sample.as_dict()
        return self._obs(), info

    def _maybe_inject_push(self) -> None:
        if self._push_event is None or self._push_injected:
            return
        onset, resolved = self._push_event
        if float(self.data.time) + 1e-9 >= onset:
            self.plant.activate(resolved, float(self.data.time))
            self._push_injected = True

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        prev_action = self._last_action.copy()
        self._last_action = action.copy()
        x_before = pos_x(self.data)
        terminated = False

        for _ in range(self.frame_skip):
            self._maybe_inject_push()
            self.plant.apply_physics(float(self.data.time))
            self.data.ctrl[:] = action
            mujoco.mj_step(self.model, self.data)
            if has_fallen(self.data) or (
                self.terminate_when_unhealthy and not self.is_healthy()
            ):
                terminated = True
                break

        self._elapsed_steps += 1
        self._gait_time = float(self._elapsed_steps) * self.dt
        reward, reward_info = self._reward(action, x_before, prev_action)
        truncated = self._elapsed_steps >= self.max_episode_steps
        info = {
            "pos_x": pos_x(self.data),
            "height": world_height(self.data),
            "pitch": pitch(self.data),
            "fallen": bool(has_fallen(self.data)),
            "fault_active": bool(self.plant.is_active),
            **reward_info,
        }
        return self._obs(), reward, bool(terminated), bool(truncated), info

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        super().close()
