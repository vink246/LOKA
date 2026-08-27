"""Standing balance controller: centroidal MPC on top of a whole-body QP.

    state ──► ConvexMPC (50 Hz)  ──► desired contact forces
           └► WholeBodyController (500 Hz) ──► joint torques

The MPC decides *how hard each foot should push* to keep the centroidal state
on target. The WBC runs an order of magnitude faster and answers the different
question of *what torques realise those forces* while keeping the feet planted,
the torso upright, and the posture near nominal.

Nothing here knows about MuJoCo simulation or DDS: feed it ``(qpos, qvel)`` and
it returns 29 torques, so the same object drives the simulator today and a
hardware bridge later.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from loka.control import tuning
from loka.control.gait import (
    GAIT_PARAMETER_NAMES,
    MODE_WALK,
    GaitConfig,
    GaitScheduler,
)
from loka.control.mpc import CentroidalReference, CentroidalState, ConvexMPC, MPCConfig
from loka.control.robot import (
    NUM_LEG_JOINTS,
    QPOS_JOINT0,
    QVEL_JOINT0,
    G1Model,
    orientation_error,
    quat_to_mat,
    quat_to_rpy,
)
from loka.control.wbc import WBCConfig, WholeBodyController, WBCTargets

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "models" / "g1" / "scene.xml"

#: A sole point this far above the estimated ground is treated as airborne.
#: Tight on purpose: a heel peeling off the ground must leave the contact set
#: promptly, otherwise the QP keeps trying to hold it down with a force it is
#: not allowed to produce.
CONTACT_HEIGHT_TOLERANCE = 0.005

# Mild swing-leg joint feedforward. Aggressive knee FF caused forward tip-over;
# clearance is secondary to capture-point stepping (see gait.py).
SWING_KNEE_FLEX_AMP = 0.18  # rad extra knee flexion at mid-swing
SWING_HIP_PITCH_AMP = 0.08  # rad hip pitch (negative = retract)
SWING_JOINT_KP = 300.0
SWING_JOINT_W = 22.0


@dataclass
class StandConfig:
    model_path: str = str(DEFAULT_MODEL)
    #: Whole-body QP period. 500 Hz is comfortable for a 59-variable QP.
    control_dt: float = 0.002
    #: MPC solves once every this many control ticks.
    mpc_decimation: int = 10
    #: Starting guess for the floor height [m]. Refined from the soles
    #: whenever the robot is actually touching something.
    ground_height: float = 0.0
    #: Clamps on the task-space accelerations handed to the WBC.
    max_linear_acc: float = 20.0
    max_angular_acc: float = 40.0
    max_joint_acc: float = 200.0
    mpc: MPCConfig = field(default_factory=MPCConfig)
    wbc: WBCConfig = field(default_factory=WBCConfig)
    gait: GaitConfig = field(default_factory=GaitConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StandConfig":
        with open(path, "r", encoding="utf-8") as handle:
            raw: Mapping[str, Any] = yaml.safe_load(handle) or {}
        mpc = MPCConfig(**raw.pop("mpc", {}))
        wbc = WBCConfig(**raw.pop("wbc", {}))
        gait_raw = dict(raw.pop("gait", {}) or {})
        if "mode" in gait_raw and isinstance(gait_raw["mode"], str):
            from loka.control.gait import parse_gait_mode

            gait_raw["mode"] = parse_gait_mode(gait_raw["mode"])
        gait = GaitConfig(**gait_raw) if gait_raw else GaitConfig()
        if raw.get("model_path"):
            raw["model_path"] = str((REPO_ROOT / raw["model_path"]).resolve())
        return cls(mpc=mpc, wbc=wbc, gait=gait, **raw)


@dataclass
class StandCommand:
    """Operator / LOKA-facing setpoint.

    Lean offsets shift the CoM target within the support polygon:
    ``lean_x > 0`` is forward, ``lean_y > 0`` is left (world +y).
    """

    #: CoM height above the foot plane [m]. ``None`` keeps the nominal stance.
    height: float | None = None
    #: CoM offset from the middle of the support polygon [m], ``(x, y)``.
    com_offset_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    yaw: float = 0.0

    @property
    def lean_x(self) -> float:
        return float(self.com_offset_xy[0])

    @lean_x.setter
    def lean_x(self, value: float) -> None:
        self.com_offset_xy[0] = float(value)

    @property
    def lean_y(self) -> float:
        return float(self.com_offset_xy[1])

    @lean_y.setter
    def lean_y(self, value: float) -> None:
        self.com_offset_xy[1] = float(value)


#: Absolute lean ceiling [m]. Support-margin clamping is tighter in practice.
MAX_LEAN_XY = 0.06
#: Fraction of the current support margin the lean command may consume.
LEAN_MARGIN_FRACTION = 0.55
#: Height band relative to the nominal stance [m].
HEIGHT_BAND = (0.50, 0.78)
#: Yaw command clamp [rad].
MAX_YAW = 0.8


TASK_PARAMETER_NAMES = frozenset({
    "height",
    "yaw",
    "lean_x",
    "lean_y",
    "com_offset_x",
    "com_offset_y",
}) | GAIT_PARAMETER_NAMES


@dataclass
class StandTelemetry:
    com: np.ndarray
    com_reference: np.ndarray
    com_error: np.ndarray
    rpy: np.ndarray
    torque: np.ndarray
    contact_forces: np.ndarray
    desired_forces: np.ndarray
    contact_mask: np.ndarray
    mpc_cost: float
    solve_ms: float


class StandController:
    def __init__(self, config: StandConfig | None = None) -> None:
        self.config = config or StandConfig()
        self.robot = G1Model(self.config.model_path)
        self.command = StandCommand()

        self.mpc = ConvexMPC(
            self.config.mpc, self.robot.num_contacts, self.robot.total_mass
        )
        self.wbc = WholeBodyController(
            self.config.wbc,
            nv=self.robot.nv,
            num_joints=self.robot.nu,
            num_contacts=self.robot.num_contacts,
            torque_limit=self.robot.torque_limit,
        )

        self._calibrate_nominal_pose()
        self.reset()

    # -- setup ------------------------------------------------------------

    def _calibrate_nominal_pose(self) -> None:
        """Measure the nominal stance geometry off the model's keyframe.

        Everything downstream is expressed relative to the support polygon, so
        editing the keyframe is enough to retune the stance -- no constants in
        this file need to change.
        """
        robot = self.robot
        nominal = robot.nominal_qpos
        robot.update(nominal, np.zeros(robot.nv))
        foot_mid = robot.foot_center_positions().mean(axis=0)
        com = robot.data.subtree_com[0].copy()

        self.nominal_joint_pos = robot.nominal_joint_pos
        self.nominal_com_offset = com - foot_mid  # xy lever + stance height
        self.nominal_base_offset = nominal[0:3] - foot_mid
        self.nominal_height = float(self.nominal_com_offset[2])
        self._refresh_posture_gains()

    def _refresh_posture_gains(self) -> None:
        """Pre-build the per-joint posture gains for both leg states.

        Row 0 is the planted gain set, row 1 the airborne one; a leg picks its
        row each tick. The upper body is unaffected -- it never has contacts.
        """
        cfg = self.config.wbc
        nu = self.robot.nu
        self._posture_kp = np.full((2, nu), cfg.kp_posture_upper)
        self._posture_kd = np.full((2, nu), cfg.kd_posture_upper)
        self._posture_weight = np.full((2, nu), cfg.weight_posture_upper)
        legs = slice(0, NUM_LEG_JOINTS)
        self._posture_kp[0, legs] = cfg.kp_posture_legs
        self._posture_kd[0, legs] = cfg.kd_posture_legs
        self._posture_weight[0, legs] = cfg.weight_posture_legs
        self._posture_kp[1, legs] = cfg.kp_posture_swing
        self._posture_kd[1, legs] = cfg.kd_posture_swing
        self._posture_weight[1, legs] = cfg.weight_posture_swing

    def _leg_states(self, contact_mask: np.ndarray) -> np.ndarray:
        """Index into the gain tables: 0 where planted, 1 where airborne.

        A planted leg is held by its contact constraints, so a stiff posture
        task would only fight them. An airborne one has nothing else deciding
        where it goes, so it reverts to a position task aimed at the landing
        pose.
        """
        per_joint = np.zeros(self.robot.nu, dtype=np.intp)
        half = self.robot.num_contacts // 2
        for leg, points in enumerate((contact_mask[:half], contact_mask[half:])):
            if not points.any():
                joints = slice(leg * NUM_LEG_JOINTS // 2, (leg + 1) * NUM_LEG_JOINTS // 2)
                per_joint[joints] = 1
        return per_joint

    def reset(self) -> None:
        self._tick = 0
        self._desired_forces = self.mpc.last_forces.copy()
        self._ground_height = float(self.config.ground_height)
        self.telemetry: StandTelemetry | None = None
        self.gait = GaitScheduler(self.config.gait)
        self._last_gait = None

    # -- references -------------------------------------------------------

    def _contact_mask(self) -> np.ndarray:
        """Which sole points are on the ground, and refresh the floor estimate.

        Compared against an absolute ground height rather than against the
        lowest sole point: a relative test would call all eight points
        "planted" while the robot is in mid-air, and the QP would happily
        compute torques bracing against support that is not there.
        """
        heights = self.robot.data.site_xpos[self.robot.contact_site_ids][:, 2]
        mask = heights < self._ground_height + CONTACT_HEIGHT_TOLERANCE
        if mask.any():
            self._ground_height = float(heights[mask].min())
        return mask

    # -- main entry point -------------------------------------------------

    def compute_torque(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        started = time.perf_counter()
        cfg = self.config
        robot = self.robot

        robot.update(qpos, qvel)
        dynamics = robot.dynamics(np.asarray(qvel, dtype=float))

        base_quat = np.asarray(qpos[3:7], dtype=float)
        rot = quat_to_mat(base_quat)
        rpy = quat_to_rpy(base_quat)
        measured_mask = self._contact_mask()
        feet = robot.foot_center_positions()
        foot_mid = feet.mean(axis=0)
        ground_z = self._ground_height

        height = (
            self.nominal_height if self.command.height is None else self.command.height
        )

        gait_out = self.gait.step(
            dt=cfg.control_dt,
            com=dynamics.com,
            com_vel=dynamics.com_vel,
            foot_centers=feet,
            ground_z=ground_z,
            height=height,
            yaw=float(self.command.yaw),
            measured_mask=measured_mask,
        )
        self._last_gait = gait_out
        walking = float(self.gait.config.mode) >= MODE_WALK and (
            float(self.gait.config.speed) > 1e-4 or self.gait._cmd_speed > 1e-4
        )

        # Planned mask for walk; allow early measured contact on the swing foot.
        if walking:
            contact_mask = gait_out.contact_mask.copy()
            if gait_out.swing.active:
                half = slice(0, 4) if gait_out.swing.leg == 0 else slice(4, 8)
                # Only accept early touchdown late in swing, once the foot is near
                # the foothold (avoids toe-drag re-planting mid-swing).
                if (
                    measured_mask[half].any()
                    and gait_out.swing.s > 0.80
                    and float(np.linalg.norm(feet[gait_out.swing.leg, :2] - gait_out.swing.foothold[:2]))
                    < 0.04
                ):
                    contact_mask[half] = True
        else:
            contact_mask = measured_mask

        if walking:
            com_ref = np.array(
                [
                    gait_out.com_ref_xy[0] + self.command.com_offset_xy[0],
                    gait_out.com_ref_xy[1] + self.command.com_offset_xy[1],
                    ground_z + height,
                ]
            )
            face_yaw = float(self.gait.config.heading)
            if abs(float(self.command.yaw)) > 1e-3:
                face_yaw = float(self.command.yaw)
        else:
            com_ref = np.array(
                [
                    foot_mid[0] + self.nominal_com_offset[0] + self.command.com_offset_xy[0],
                    foot_mid[1] + self.nominal_com_offset[1] + self.command.com_offset_xy[1],
                    ground_z + height,
                ]
            )
            face_yaw = float(self.command.yaw)

        # --- centroidal MPC (decimated) ---
        if self._tick % cfg.mpc_decimation == 0:
            state = CentroidalState(
                rpy=rpy,
                com=dynamics.com,
                angular_velocity=rot @ qvel[3:6],
                com_velocity=dynamics.com_vel,
                inertia=robot.com_inertia(),
                contact_pos=dynamics.contact_pos,
            )
            reference = CentroidalReference(com=com_ref, yaw=face_yaw)
            self._desired_forces = self.mpc.solve(state, reference, contact_mask)
        self._tick += 1

        forces = self._desired_forces.copy()
        if not contact_mask.all():
            forces[~contact_mask] = 0.0

        support = float(contact_mask.mean())
        wbc_cfg = cfg.wbc
        vel_err = dynamics.com_vel.copy()
        if walking:
            # Track commanded planar velocity, but bleed it off hard if the torso
            # has run ahead of the support-tied CoM reference (forward tip).
            pos_err_fwd = float((dynamics.com - com_ref)[0] * np.cos(face_yaw)
                                + (dynamics.com - com_ref)[1] * np.sin(face_yaw))
            scale = float(np.clip(1.0 - pos_err_fwd / 0.05, 0.0, 1.0))
            vel_err[0] -= scale * gait_out.com_vel_cmd[0]
            vel_err[1] -= scale * gait_out.com_vel_cmd[1]
            # Extra damping when tipping forward past the feet.
            if pos_err_fwd > 0.03:
                vel_err[0] += 1.5 * pos_err_fwd * np.cos(face_yaw)
                vel_err[1] += 1.5 * pos_err_fwd * np.sin(face_yaw)
        base_linear_acc = np.clip(
            wbc_cfg.kp_base_position * (com_ref - dynamics.com)
            - wbc_cfg.kd_base_position * vel_err,
            -cfg.max_linear_acc,
            cfg.max_linear_acc,
        )
        yaw_quat = np.array(
            [np.cos(0.5 * face_yaw), 0.0, 0.0, np.sin(0.5 * face_yaw)]
        )
        base_angular_acc = np.clip(
            wbc_cfg.kp_base_orientation * orientation_error(base_quat, yaw_quat)
            - wbc_cfg.kd_base_orientation * qvel[3:6],
            -cfg.max_angular_acc,
            cfg.max_angular_acc,
        )
        joints = np.arange(robot.nu)
        state = self._leg_states(contact_mask)
        joint_acc = np.clip(
            self._posture_kp[state, joints] * (self.nominal_joint_pos - qpos[QPOS_JOINT0:])
            - self._posture_kd[state, joints] * qvel[QVEL_JOINT0:],
            -cfg.max_joint_acc,
            cfg.max_joint_acc,
        )
        joint_weights = self._posture_weight[state, joints].copy()

        # Walk: pin hip_yaw / ankle_roll on the *stance* leg only. Planted-leg
        # posture gains are zero by design for stand, so yaw otherwise drifts
        # into a pigeon-toe shuffle. Holding both legs + swing FF tips the base.
        swing_leg = int(gait_out.swing.leg) if (walking and gait_out.swing.active) else -1
        if walking:
            qj = np.asarray(qpos[QPOS_JOINT0:], dtype=float)
            dqj = np.asarray(qvel[QVEL_JOINT0:], dtype=float)
            for leg_i in (0, 1):
                if leg_i == swing_leg:
                    continue
                for local in (2, 5):  # hip_yaw, ankle_roll
                    ji = leg_i * (NUM_LEG_JOINTS // 2) + local
                    joint_acc[ji] = np.clip(
                        200.0 * (self.nominal_joint_pos[ji] - qj[ji]) - 14.0 * dqj[ji],
                        -cfg.max_joint_acc,
                        cfg.max_joint_acc,
                    )
                    joint_weights[ji] = max(float(joint_weights[ji]), 32.0)

        swing_jacobian = None
        swing_acc = None
        if walking and gait_out.swing.active:
            leg = int(gait_out.swing.leg)

            # Optional soft Cartesian XY (weight 0 by default — even mild 3D/XY
            # swing tasks tip this plant when stacked on knee FF + stance yaw).
            if wbc_cfg.weight_swing_foot > 1.0:
                site_id = int(robot.foot_center_site_ids[leg])
                qvel_arr = np.asarray(qvel, dtype=float)
                j_f = robot.site_jacobian(site_id)
                p = feet[leg]
                v = j_f @ qvel_arr
                p_des = gait_out.swing.des_pos
                bias_s = robot.site_bias_acc(site_id, qvel_arr)
                swing_acc = (
                    wbc_cfg.kp_swing_foot * (p_des - p)
                    - wbc_cfg.kd_swing_foot * v
                    - bias_s
                )
                swing_acc[0] = float(
                    np.clip(swing_acc[0], -cfg.max_linear_acc, cfg.max_linear_acc)
                )
                swing_acc[1] = float(
                    np.clip(swing_acc[1], -cfg.max_linear_acc, cfg.max_linear_acc)
                )
                swing_acc[2] = 0.0
                swing_jacobian = j_f

            # Knee/hip lift schedule. Pure joint FF + ss_lateral_bias is the
            # stable envelope; DLS XY previously fought the stance base.
            j0 = leg * (NUM_LEG_JOINTS // 2)
            j1 = j0 + (NUM_LEG_JOINTS // 2)
            q_leg = np.asarray(qpos[QPOS_JOINT0 + j0 : QPOS_JOINT0 + j1], dtype=float)
            dq_leg = np.asarray(qvel[QVEL_JOINT0 + j0 : QVEL_JOINT0 + j1], dtype=float)
            s = float(gait_out.swing.s)
            lift = 4.0 * s * (1.0 - s)  # peaks at 1.0 mid-swing
            q_des = q_leg.copy()
            q_des[0] = float(self.nominal_joint_pos[j0 + 0]) - SWING_HIP_PITCH_AMP * lift
            q_des[3] = float(self.nominal_joint_pos[j0 + 3]) + SWING_KNEE_FLEX_AMP * lift
            q_des[2] = float(self.nominal_joint_pos[j0 + 2])  # hip_yaw
            q_des[5] = float(self.nominal_joint_pos[j0 + 5])  # ankle_roll
            joint_acc[j0:j1] = np.clip(
                SWING_JOINT_KP * (q_des - q_leg) - 18.0 * dq_leg,
                -cfg.max_joint_acc,
                cfg.max_joint_acc,
            )
            joint_weights[j0:j1] = SWING_JOINT_W
            joint_weights[j0 + 0] = SWING_JOINT_W * 1.5  # hip pitch
            joint_weights[j0 + 3] = SWING_JOINT_W * 2.0  # knee — primary lift actuator
            joint_weights[j0 + 2] = 40.0
            joint_weights[j0 + 5] = 40.0

        solution = self.wbc.solve(
            dynamics,
            WBCTargets(
                base_linear_acc=base_linear_acc,
                base_angular_acc=base_angular_acc,
                joint_acc=joint_acc,
                contact_forces=forces,
                joint_weights=joint_weights,
                base_weight_scale=max(support, 0.2 if walking else support),
                swing_jacobian=swing_jacobian,
                swing_acc=swing_acc,
            ),
            contact_mask=contact_mask,
        )

        self.telemetry = StandTelemetry(
            com=dynamics.com,
            com_reference=com_ref,
            com_error=dynamics.com - com_ref,
            rpy=rpy,
            torque=solution.torque,
            contact_forces=solution.forces,
            desired_forces=forces,
            contact_mask=contact_mask,
            mpc_cost=self.mpc.cost,
            solve_ms=(time.perf_counter() - started) * 1e3,
        )
        return solution.torque

    # -- runtime tuning (LOKA hooks) --------------------------------------

    def task_snapshot(self) -> dict[str, float]:
        """Current task setpoints the orchestrator may rewrite."""
        height = (
            self.nominal_height if self.command.height is None else float(self.command.height)
        )
        snap = {
            "height": height,
            "yaw": float(self.command.yaw),
            "lean_x": self.command.lean_x,
            "lean_y": self.command.lean_y,
        }
        snap.update(self.gait.snapshot())
        return snap

    def lean_limits(self) -> dict[str, float]:
        """Physics-aware lean clamps from the current support polygon."""
        margin = self.robot.support_margin()
        return {
            "lean_x_min": -LEAN_MARGIN_FRACTION * float(margin[0]),
            "lean_x_max": LEAN_MARGIN_FRACTION * float(margin[1]),
            "lean_y_min": -LEAN_MARGIN_FRACTION * float(margin[2]),
            "lean_y_max": LEAN_MARGIN_FRACTION * float(margin[3]),
            "height_min": HEIGHT_BAND[0],
            "height_max": min(HEIGHT_BAND[1], self.nominal_height),
            "yaw_min": -MAX_YAW,
            "yaw_max": MAX_YAW,
        }

    def set_task_targets(self, updates: Mapping[str, float]) -> dict[str, float]:
        """Apply clamped task / gait setpoints; return what was kept.

        Unknown keys are ignored with a warning print so an LLM typo cannot
        abort the apply path mid-scratchpad. ``gait.mode`` may be a string
        (``walk`` / ``stand`` / …) or a numeric code.
        """
        limits = self.lean_limits()
        applied: dict[str, float] = {}
        gait_updates = {}
        for raw_name, raw_value in updates.items():
            name = str(raw_name).strip()
            if name in ("com_offset_x",):
                name = "lean_x"
            elif name in ("com_offset_y",):
                name = "lean_y"
            if name in GAIT_PARAMETER_NAMES:
                gait_updates[name] = raw_value
                continue
            if name not in TASK_PARAMETER_NAMES:
                print(f"     * [WARN] Unknown task parameter '{raw_name}' (ignored)")
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                print(f"     * [WARN] Non-numeric task parameter '{name}' (ignored)")
                continue

            if name == "height":
                value = float(np.clip(value, limits["height_min"], limits["height_max"]))
                self.command.height = value
            elif name == "yaw":
                value = float(np.clip(value, limits["yaw_min"], limits["yaw_max"]))
                self.command.yaw = value
            elif name == "lean_x":
                value = float(
                    np.clip(
                        value,
                        max(limits["lean_x_min"], -MAX_LEAN_XY),
                        min(limits["lean_x_max"], MAX_LEAN_XY),
                    )
                )
                self.command.lean_x = value
            elif name == "lean_y":
                value = float(
                    np.clip(
                        value,
                        max(limits["lean_y_min"], -MAX_LEAN_XY),
                        min(limits["lean_y_max"], MAX_LEAN_XY),
                    )
                )
                self.command.lean_y = value
            applied[name] = value

        if gait_updates:
            # Keep StandConfig.gait and scheduler in sync.
            gait_applied = self.gait.apply_updates(gait_updates)
            applied.update(gait_applied)
            self.config.gait = self.gait.config
            # Entering walk with zero speed: nudge a default crawl so mode alone works.
            if (
                "gait.mode" in gait_applied
                and gait_applied["gait.mode"] >= MODE_WALK
                and self.gait.config.speed <= 1e-6
            ):
                self.gait.config.speed = 0.25
                applied["gait.speed"] = 0.25  # crawl default; safer than 0.3 with lift
            # Align facing with travel heading when starting a walk if yaw~0.
            if (
                "gait.mode" in gait_applied
                and gait_applied["gait.mode"] >= MODE_WALK
                and abs(self.command.yaw) < 1e-6
                and "gait.heading" in gait_applied
            ):
                self.command.yaw = float(self.gait.config.heading)
                applied["yaw"] = float(self.command.yaw)
        return applied

    def tunables(self) -> dict[str, float]:
        """Current value of every runtime-mutable parameter, by dotted path."""
        return tuning.snapshot(self.config)

    def update_weights(self, **overrides: float) -> dict[str, float]:
        """Patch weights and gains in place; return the clamped values applied.

        Safe to call between control ticks while the robot is standing on the
        result: no OSQP problem is rebuilt, so there is no pause for the plant
        to fall into. Names are the dotted paths in
        :mod:`loka.control.tuning`, which disambiguate the several fields that
        exist on both layers with very different magnitudes.
        """
        applied = tuning.apply(self.config, overrides)
        groups = {tuning.BY_PATH[path].group for path in applied}
        if "mpc" in groups:
            self.mpc.refresh()
        if "wbc" in groups:
            self.wbc.refresh()
            self._refresh_posture_gains()
        return applied
