"""Locomotion controller: convex SRBD MPC on top of a whole-body QP.

    state ──► GaitScheduler    (500 Hz) ──► contacts, CoM / DCM ref, swing Bézier
           ──► ConvexMPC        (50 Hz) ──► desired contact forces (finite-foot)
           └► WholeBodyController (500 Hz) ──► joint torques

The gait layer decides *who is on the ground and where the body should be*;
standing is simply the case where it hands back both feet and a stationary
reference. The MPC decides *how hard each foot should push* to track that
centroidal reference, with CoP constrained to the sole and airborne contacts
on the Bézier. The WBC runs an order of magnitude faster and answers the
different question of *what torques realise those forces* while keeping the
planted feet still, the torso upright, and the swing foot on its arc.

Papers: Di Carlo et al. IROS 2018 (this QP); Sleiman et al. TRO 2021 (CoP in
the sole, not a nonlinear centroidal NLP); Galliker et al. Humanoids 2022
(short horizon because the gait is the reference — we keep DCM + Bézier, not
ocs2).

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
    EARLY_PLANT_S,
    TOUCHDOWN_HEIGHT,
    TOUCHDOWN_RADIUS,
    GAIT_PARAMETER_NAMES,
    MODE_WALK,
    GaitConfig,
    GaitScheduler,
    circ_mid,
    heading_frame,
    wrap_angle,
)
from loka.control.mpc import (
    ConvexMPC,
    CentroidalReference,
    CentroidalState,
    MPCConfig,
)
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

# Per-leg joint order in the Unitree DDS layout.
HIP_PITCH, HIP_ROLL, HIP_YAW, KNEE, ANKLE_PITCH, ANKLE_ROLL = range(6)

#: Hip-yaw hold while walking, on both legs, so the feet stay square to the
#: pelvis (whose yaw the base orientation task already owns).
STANCE_YAW_KP = 150.0
STANCE_YAW_KD = 15.0
STANCE_YAW_WEIGHT = 30.0

#: Swing-leg null space: damping only, like a planted leg. The 3-row Cartesian
#: foot task cannot determine six joints, and near knee extension the leg
#: Jacobian is ill-conditioned, so a stiffness term here would fight the foot
#: task while an unregularised null space lets the QP fling the leg. What is
#: needed is somewhere for excess joint velocity to go.
SWING_NULLSPACE_W = 4.0
SWING_NULLSPACE_KD = 20.0

#: The foot-centre task constrains position only, leaving the ankle free. Hold
#: the sole to the stand pose on the way *up* so it does not flap; release
#: pitch and roll on the descending arc and hand them to a world-flat
#: orientation task so the four sites arrive together. Pinning them to the
#: stand keyframe for the whole swing left the right sole on a toe; releasing
#: them with no plane left them flopping. Hip yaw is not a landing DOF and
#: stays held.
SWING_ANKLE_KP = 200.0
SWING_ANKLE_KD = 20.0
SWING_ANKLE_W = 20.0
SWING_ANKLE_RELEASE_S = 0.70

#: Stance-knee singularity guard while walking. Planted-leg posture is
#: damping only, which left the 24° stand flexion unheld and let the
#: CoM-height task slam the knee through the −5° stop. A stiff keyframe
#: hold stopped the vault but pinned the pelvis so the swing foot only
#: cleared 8 mm. Two layers: a light bias at the keyframe, and a hard
#: one-sided kick only when the knee is near the stop.
STANCE_KNEE_MIN = 0.262  # 15 deg; emergency band
STANCE_KNEE_KP = 150.0
STANCE_KNEE_KD = 15.0
STANCE_KNEE_W = 12.0
STANCE_KNEE_STOP_KP = 800.0
STANCE_KNEE_STOP_KD = 80.0
STANCE_KNEE_STOP_W = 200.0

#: Floor on the base task weight, so single support does not silently hand the
#: torso over to the posture task.
MIN_BASE_WEIGHT_SCALE = 0.4

#: Lateral CoM stiffness as a fraction of ``kp_base_position`` while walking.
#: The sole's leftover CoP travel is ~0.05 m/s²; kp=50 on a centimetre of
#: error already spends that. Standing keeps the full gain.
WALK_KP_XY_SCALE = 0.5


@dataclass
class LocomotionConfig:
    model_path: str = str(DEFAULT_MODEL)
    #: Whole-body QP period. 500 Hz is comfortable for a 59-variable QP.
    control_dt: float = 0.002
    #: MPC solves once every this many control ticks.
    mpc_decimation: int = 10
    #: Starting guess for the floor height [m]. Refined from the soles
    #: whenever the robot is actually touching something.
    ground_height: float = 0.0
    #: Clamps on the task-space accelerations handed to the WBC. The CoM cap is
    #: deliberately near what the soles can actually deliver; a swing foot is a
    #: light limb in free space and gets a much wider budget.
    max_linear_acc: float = 20.0
    max_angular_acc: float = 40.0
    max_joint_acc: float = 200.0
    max_swing_acc: float = 120.0
    #: Cap on the late-swing sole-orientation acceleration [rad/s²]. Soft on
    #: purpose: this should lay the sole down, not slap it.
    max_swing_ang_acc: float = 80.0
    mpc: MPCConfig = field(default_factory=MPCConfig)
    wbc: WBCConfig = field(default_factory=WBCConfig)
    gait: GaitConfig = field(default_factory=GaitConfig)
    #: Which locomotion stack ``Simulation`` builds. See ``loka.control.stacks``.
    stack: str = "legacy_dcm"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LocomotionConfig":
        with open(path, "r", encoding="utf-8") as handle:
            raw: Mapping[str, Any] = yaml.safe_load(handle) or {}
        mpc = MPCConfig(**raw.pop("mpc", {}))
        wbc = WBCConfig(**raw.pop("wbc", {}))
        gait_raw = dict(raw.pop("gait", {}) or {})
        if "mode" in gait_raw and isinstance(gait_raw["mode"], str):
            from loka.control.gait import parse_gait_mode

            gait_raw["mode"] = parse_gait_mode(gait_raw["mode"])
        gait = GaitConfig(**gait_raw) if gait_raw else GaitConfig()
        raw.pop("alip", None)
        raw.pop("wbqp", None)
        if raw.get("model_path"):
            raw["model_path"] = str((REPO_ROOT / raw["model_path"]).resolve())
        return cls(mpc=mpc, wbc=wbc, gait=gait, **raw)


@dataclass
class LocomotionCommand:
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
#: Yaw command clamp [rad]. While standing this is an offset from the feet,
#: not a world heading. Walking ignores it; ``gait.heading`` owns travel yaw.
MAX_YAW = 0.5
#: CoM-height slew while walking [m/s]. Standing snaps to the command.
HEIGHT_SLEW_WALK = 0.05
#: Lowest CoM height a walk will chase. Deeper crouches are a stand.
WALK_HEIGHT_MIN = 0.60


TASK_PARAMETER_NAMES = frozenset({
    "height",
    "yaw",
    "lean_x",
    "lean_y",
    "com_offset_x",
    "com_offset_y",
}) | GAIT_PARAMETER_NAMES


@dataclass
class LocomotionTelemetry:
    com: np.ndarray
    com_reference: np.ndarray
    com_error: np.ndarray
    com_velocity: np.ndarray
    com_velocity_reference: np.ndarray
    rpy: np.ndarray
    torque: np.ndarray
    contact_forces: np.ndarray
    desired_forces: np.ndarray
    contact_mask: np.ndarray
    mpc_cost: float | None
    solve_ms: float
    walking: bool = False
    gait_phase: float = 0.0
    #: Swing-foot height above the foot line, and its distance from the
    #: commanded point on the swing arc [m]. Zero while both feet are planted.
    swing_clearance: float = 0.0
    swing_error: float = 0.0
    #: Angle between the swing sole's +z and world +z [rad]. Zero is flat.
    swing_tilt: float = 0.0
    yaw_ref: float = 0.0
    heading_goal: float = 0.0
    heading_error: float = 0.0


class LocomotionController:
    def __init__(
        self,
        config: LocomotionConfig | None = None,
        *,
        foothold_policy=None,
    ) -> None:
        self.config = config or LocomotionConfig()
        self._foothold_policy = foothold_policy
        self.robot = G1Model(self.config.model_path)
        self.command = LocomotionCommand()

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
        self._build_knee_table()

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

    def _build_knee_table(self) -> None:
        """Stance-knee angle versus CoM height, from a short foot-fixed IK.

        The standing keyframe is the tall pose. Pulling a crouched walk back
        to that knee fights the height command, so the bias interpolates a
        pose whose feet stay where the keyframe put them.
        """
        robot = self.robot
        q_nom = robot.nominal_qpos.copy()
        robot.update(q_nom, np.zeros(robot.nv))
        feet0 = robot.foot_center_positions().copy()
        pelvis_z = float(q_nom[2])
        heights = np.arange(WALK_HEIGHT_MIN, self.nominal_height + 1e-9, 0.02)
        if heights.size == 0 or abs(heights[-1] - self.nominal_height) > 1e-6:
            heights = np.append(heights, self.nominal_height)
        table = np.zeros((len(heights), 2))
        q = q_nom.copy()
        per = NUM_LEG_JOINTS // 2
        leg_cols = np.arange(QVEL_JOINT0, QVEL_JOINT0 + NUM_LEG_JOINTS)
        for i, h in enumerate(heights):
            q[:] = q_nom
            q[2] = pelvis_z - (self.nominal_height - float(h))
            for _ in range(20):
                robot.update(q, np.zeros(robot.nv))
                err = (feet0 - robot.foot_center_positions()).reshape(-1)
                rows = []
                for leg in (0, 1):
                    jp, _ = robot.site_spatial_jacobian(
                        int(robot.foot_center_site_ids[leg])
                    )
                    rows.append(jp)
                jac = np.vstack(rows)[:, leg_cols]
                dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-3 * np.eye(6), err)
                q[QPOS_JOINT0:QPOS_JOINT0 + NUM_LEG_JOINTS] += dq
            table[i, 0] = q[QPOS_JOINT0 + KNEE]
            table[i, 1] = q[QPOS_JOINT0 + per + KNEE]
        robot.update(q_nom, np.zeros(robot.nv))
        self._knee_heights = heights
        self._knee_table = table

    def _knee_bias(self, height: float) -> np.ndarray:
        return np.array([
            float(np.interp(height, self._knee_heights, self._knee_table[:, leg]))
            for leg in (0, 1)
        ])

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
        self.telemetry: LocomotionTelemetry | None = None
        self.gait = GaitScheduler(
            self.config.gait,
            policy=self._foothold_policy,
            stack_name=getattr(self.config, "stack", "legacy_dcm"),
        )
        self._last_gait = None
        self._last_joint_weights = np.zeros(self.robot.nu)
        self._last_swing_orient_active = False
        self._last_knee_guard = np.zeros(2, dtype=bool)
        self._last_com_z_scale = 1.0
        self._height_ref = None
        self._knee_heights = np.zeros(1)
        self._knee_table = np.zeros((1, 2))

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

    def _contact_pos_preview(
        self,
        *,
        foot_pose: np.ndarray,
        feet: np.ndarray,
        contact_pos: np.ndarray,
        ground_z: float,
        foot_yaw: np.ndarray | None = None,
        measured_yaw: np.ndarray | None = None,
    ) -> np.ndarray:
        """Sole-point positions over the MPC horizon, ``(horizon, nc, 3)``.

        The gait plans foot *centres* (xy) plus Bézier clearance (z). The MPC
        reasons about the eight sole points, so each planned centre is
        re-decorated with the sole offsets measured on the robot right now.
        Airborne sites sit at the swing height, not on the ground-height
        foothold — ``fz`` is already gated to zero by the schedule.
        """
        pose = np.asarray(foot_pose, dtype=float)
        if pose.ndim != 3 or pose.shape[1:] != (2, 3):
            # Back-compat if a caller still hands ``(horizon, 2, 2)`` xy.
            xy = pose.reshape(-1, 2, 2)
            pose = np.zeros((xy.shape[0], 2, 3))
            pose[:, :, :2] = xy
        horizon = int(pose.shape[0])
        per_foot = self.robot.num_contacts // 2
        offsets = contact_pos - np.repeat(feet, per_foot, axis=0)
        seq = np.zeros((horizon, self.robot.num_contacts, 3))
        planned = None if foot_yaw is None else np.asarray(foot_yaw, dtype=float)
        measured = None if measured_yaw is None else np.asarray(measured_yaw, dtype=float)
        for leg in (0, 1):
            rows = slice(leg * per_foot, (leg + 1) * per_foot)
            local = offsets[rows, :2]
            if planned is not None and measured is not None:
                for k in range(horizon):
                    dy = wrap_angle(float(planned[k, leg]) - float(measured[leg]))
                    c, s = np.cos(dy), np.sin(dy)
                    rot = np.array([[c, -s], [s, c]])
                    seq[k, rows, :2] = pose[k, leg, :2] + local @ rot.T
            else:
                seq[:, rows, :2] = pose[:, leg, None, :2] + local
            seq[:, rows, 2] = ground_z + pose[:, leg, 2, None]
        return seq

    def compute_torque(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        started = time.perf_counter()
        cfg = self.config
        robot = self.robot
        qvel_arr = np.asarray(qvel, dtype=float)

        robot.update(qpos, qvel)
        dynamics = robot.dynamics(qvel_arr)

        base_quat = np.asarray(qpos[3:7], dtype=float)
        rot = quat_to_mat(base_quat)
        rpy = quat_to_rpy(base_quat)
        measured_mask = self._contact_mask()
        feet = robot.foot_center_positions()
        foot_mid = feet.mean(axis=0)
        ground_z = self._ground_height

        height_cmd = (
            self.nominal_height if self.command.height is None else float(self.command.height)
        )
        if self._height_ref is None:
            self._height_ref = height_cmd
        # The gait clock below decides ``walking``; slew using the previous
        # tick so a height step cannot jump ω₀ inside the step in progress.
        was_walking = self.gait.walking
        if was_walking:
            goal_h = max(height_cmd, WALK_HEIGHT_MIN)
            step_h = HEIGHT_SLEW_WALK * cfg.control_dt
            self._height_ref += float(np.clip(goal_h - self._height_ref, -step_h, step_h))
        else:
            self._height_ref = height_cmd
        height = float(self._height_ref)

        stance_i = int(np.argmin(feet[:, 2]))
        self.gait.mass = float(robot.total_mass)
        gait_out = self.gait.step(
            dt=cfg.control_dt,
            com=dynamics.com,
            com_vel=dynamics.com_vel,
            foot_centers=feet,
            ground_z=ground_z,
            height=height,
            measured_mask=measured_mask,
            foot_yaws=robot.foot_yaws(),
            L_meas=robot.angular_momentum_about(feet[stance_i]),
        )
        self._last_gait = gait_out
        walking = gait_out.walking

        swing_accepted = False
        if walking:
            contact_mask = gait_out.contact_mask.copy()
            swing = gait_out.swing
            if swing.active and swing.s > EARLY_PLANT_S:
                # Sleiman 6-DoF contact: a toe scuff is not a plant. Step 5
                # used to drop the Cartesian task at s=0.83 with the centre
                # 11 mm up and 21 mm short (6° pitched sole, one site in
                # contact, xy inside TOUCHDOWN_RADIUS). The ZMP then jumped
                # back onto a short foot and capture saturated.
                rows = slice(0, 4) if swing.leg == 0 else slice(4, 8)
                near = float(np.linalg.norm(feet[swing.leg, :2] - swing.foothold[:2]))
                seated = float(feet[swing.leg, 2] - ground_z) <= TOUCHDOWN_HEIGHT
                if (
                    seated
                    and measured_mask[rows].any()
                    and near < TOUCHDOWN_RADIUS
                ):
                    contact_mask[rows] = True
                    swing_accepted = True
            # Lean is in the travel frame. Walking faces the plan, not command.yaw.
            forward, left = heading_frame(float(gait_out.yaw_ref))
            lean = (
                self.command.com_offset_xy[0] * forward
                + self.command.com_offset_xy[1] * left
            )
            com_ref = np.array(
                [
                    gait_out.com_ref_xy[0] + lean[0],
                    gait_out.com_ref_xy[1] + lean[1],
                    ground_z + height,
                ]
            )
            com_vel_ref = np.array([gait_out.com_vel_ref[0], gait_out.com_vel_ref[1], 0.0])
            face_yaw = float(gait_out.yaw_ref)
            yaw_rate_ref = float(gait_out.yaw_rate_ref)
        else:
            contact_mask = measured_mask
            foot_yaws = robot.foot_yaws()
            stance_yaw = float(circ_mid(foot_yaws[0], foot_yaws[1]))
            forward, left = heading_frame(stance_yaw)
            offset = self.nominal_com_offset[:2] + self.command.com_offset_xy
            lean = offset[0] * forward + offset[1] * left
            com_ref = np.array(
                [foot_mid[0] + lean[0], foot_mid[1] + lean[1], ground_z + height]
            )
            com_vel_ref = np.zeros(3)
            face_yaw = stance_yaw + float(self.command.yaw)
            yaw_rate_ref = 0.0

        # --- centroidal MPC (decimated) ---
        if self._tick % cfg.mpc_decimation == 0:
            schedule = None
            contact_pos_seq = None
            if walking:
                legs_planted, foot_pose, foot_yaw = self.gait.preview(
                    horizon=cfg.mpc.horizon, dt=cfg.mpc.dt
                )
                schedule = np.repeat(legs_planted, robot.num_contacts // 2, axis=1)
                contact_pos_seq = self._contact_pos_preview(
                    foot_pose=foot_pose,
                    feet=feet,
                    contact_pos=dynamics.contact_pos,
                    ground_z=ground_z,
                    foot_yaw=foot_yaw,
                    measured_yaw=robot.foot_yaws(),
                )
            state = CentroidalState(
                rpy=rpy,
                com=dynamics.com,
                angular_velocity=rot @ qvel[3:6],
                com_velocity=dynamics.com_vel,
                inertia=robot.com_inertia(),
                contact_pos=dynamics.contact_pos,
            )
            reference = CentroidalReference(
                com=com_ref,
                yaw=face_yaw,
                yaw_rate=yaw_rate_ref,
                com_velocity=com_vel_ref,
            )
            self._desired_forces = self.mpc.solve(
                state,
                reference,
                contact_mask,
                schedule=schedule,
                contact_pos_seq=contact_pos_seq,
                foot_yaw_seq=foot_yaw if walking else None,
            )
        self._tick += 1

        forces = self._desired_forces.copy()
        if not contact_mask.all():
            forces[~contact_mask] = 0.0

        support = float(contact_mask.mean())
        wbc_cfg = cfg.wbc
        # Vertical CoM stiffness tracks how much sole is actually planted.
        # Standing (support = 1) is unchanged. Single support cannot hold
        # stand height the way two feet can, and demanding it vaults the
        # pelvis over the stance foot. Damping stays, so a drop is still
        # caught; only the spring is relaxed. Lateral gain is lowered while
        # walking so the QP does not spend the whole sole on an unachievable
        # vref (see WALK_KP_XY_SCALE).
        kp_xy = wbc_cfg.kp_base_position * (WALK_KP_XY_SCALE if walking else 1.0)
        kp_com = np.array(
            [
                kp_xy,
                kp_xy,
                wbc_cfg.kp_base_position * support,
            ]
        )
        base_linear_acc = np.clip(
            kp_com * (com_ref - dynamics.com)
            - wbc_cfg.kd_base_position * (dynamics.com_vel - com_vel_ref),
            -cfg.max_linear_acc,
            cfg.max_linear_acc,
        )
        self._last_com_z_scale = support
        yaw_quat = np.array(
            [np.cos(0.5 * face_yaw), 0.0, 0.0, np.sin(0.5 * face_yaw)]
        )
        omega_ref = rot.T @ np.array([0.0, 0.0, yaw_rate_ref])
        base_angular_acc = np.clip(
            wbc_cfg.kp_base_orientation * orientation_error(base_quat, yaw_quat)
            - wbc_cfg.kd_base_orientation * (qvel[3:6] - omega_ref),
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

        qj = np.asarray(qpos[QPOS_JOINT0:], dtype=float)
        dqj = qvel_arr[QVEL_JOINT0:]

        def hold_joint(
            index: int, kp: float, kd: float, weight: float, target: float | None = None
        ) -> None:
            """Pin one joint, overriding the posture row. Default target is nominal."""
            if target is None:
                target = float(self.nominal_joint_pos[index])
            joint_acc[index] = np.clip(
                kp * (target - qj[index]) - kd * dqj[index],
                -cfg.max_joint_acc,
                cfg.max_joint_acc,
            )
            joint_weights[index] = max(float(joint_weights[index]), weight)

        per_leg = NUM_LEG_JOINTS // 2
        foot_yaws_now = robot.foot_yaws()
        pelvis_yaw = float(rpy[2])
        if walking:
            # Hip yaw holds the foot square to the pelvis yaw the base task
            # is tracking, not to the standing keyframe. A turn otherwise
            # asks the torso to yaw while the hip pins the foot to yaw 0.
            for leg_i in (0, 1):
                yaw_joint = leg_i * per_leg + HIP_YAW
                target = self.nominal_joint_pos[yaw_joint] + wrap_angle(
                    float(foot_yaws_now[leg_i]) - face_yaw
                )
                hold_joint(
                    yaw_joint, STANCE_YAW_KP, STANCE_YAW_KD, STANCE_YAW_WEIGHT, target
                )

        swing_jacobian = None
        swing_acc = None
        swing_orient_jacobian = None
        swing_orient_acc = None
        if walking and gait_out.swing.active and not swing_accepted:
            leg = int(gait_out.swing.leg)
            swing = gait_out.swing

            # The Cartesian foot task is the *primary* swing task: it is what
            # actually executes the planned foothold. The arc's own acceleration
            # does the work and the gains only mop up residual error, so a
            # 0.25 s swing does not have to be conjured out of position error.
            site_id = int(robot.foot_center_site_ids[leg])
            j_f, j_w = robot.site_spatial_jacobian(site_id)
            bias_lin, bias_ang = robot.site_spatial_bias_acc(site_id, qvel_arr)
            foot_vel = j_f @ qvel_arr
            swing_acc = np.clip(
                swing.des_acc
                + wbc_cfg.kp_swing_foot * (swing.des_pos - feet[leg])
                + wbc_cfg.kd_swing_foot * (swing.des_vel - foot_vel)
                - bias_lin,
                -cfg.max_swing_acc,
                cfg.max_swing_acc,
            )
            swing_jacobian = j_f

            # Sole yaw tracks the swing arc for the whole swing. Pitch and
            # roll stay with the ankle hold until the descending arc, then
            # this task lays the sole flat. Zeroing those rows keeps the
            # residual 3-wide so the QP sparsity does not change.
            foot_quat = robot.site_quat(site_id)
            des_quat = np.array(
                [np.cos(0.5 * swing.des_yaw), 0.0, 0.0, np.sin(0.5 * swing.des_yaw)]
            )
            err_world = quat_to_mat(foot_quat) @ orientation_error(foot_quat, des_quat)
            omega = j_w @ qvel_arr
            omega_ref_foot = np.array([0.0, 0.0, swing.des_yaw_rate])
            swing_orient_acc = np.clip(
                wbc_cfg.kp_swing_orient * err_world
                - wbc_cfg.kd_swing_orient * (omega - omega_ref_foot)
                - bias_ang,
                -cfg.max_swing_ang_acc,
                cfg.max_swing_ang_acc,
            )
            swing_orient_jacobian = j_w.copy()
            if swing.s <= SWING_ANKLE_RELEASE_S:
                swing_orient_jacobian[:2] = 0.0
                swing_orient_acc = swing_orient_acc.copy()
                swing_orient_acc[:2] = 0.0

            # Damping-only null space, plus a hip-yaw hold so the foot stays
            # square. Ankle pitch/roll stay pinned on the way up and are
            # released past SWING_ANKLE_RELEASE_S so the orientation task
            # owns the sole plane.
            j0 = leg * per_leg
            joint_acc[j0 : j0 + per_leg] = np.clip(
                -SWING_NULLSPACE_KD * dqj[j0 : j0 + per_leg],
                -cfg.max_joint_acc,
                cfg.max_joint_acc,
            )
            joint_weights[j0 : j0 + per_leg] = SWING_NULLSPACE_W
            yaw_target = self.nominal_joint_pos[j0 + HIP_YAW] + wrap_angle(
                float(swing.des_yaw) - pelvis_yaw
            )
            hold_joint(
                j0 + HIP_YAW, SWING_ANKLE_KP, SWING_ANKLE_KD, SWING_ANKLE_W, yaw_target
            )
            if swing.s <= SWING_ANKLE_RELEASE_S:
                for local in (ANKLE_PITCH, ANKLE_ROLL):
                    hold_joint(j0 + local, SWING_ANKLE_KP, SWING_ANKLE_KD, SWING_ANKLE_W)

        knee_guard = np.zeros(2, dtype=bool)
        if walking:
            # After the swing overwrite so only planted knees are eligible.
            for leg_i in (0, 1):
                knee = leg_i * per_leg + KNEE
                if state[knee] != 0:
                    continue
                if qj[knee] < STANCE_KNEE_MIN:
                    hold_joint(
                        knee,
                        STANCE_KNEE_STOP_KP,
                        STANCE_KNEE_STOP_KD,
                        STANCE_KNEE_STOP_W,
                    )
                else:
                    knees = self._knee_bias(height)
                    hold_joint(
                        knee,
                        STANCE_KNEE_KP,
                        STANCE_KNEE_KD,
                        STANCE_KNEE_W,
                        float(knees[leg_i]),
                    )
                knee_guard[leg_i] = True

        self._last_joint_weights = joint_weights
        self._last_swing_orient_active = swing_orient_jacobian is not None
        self._last_knee_guard = knee_guard

        solution = self.wbc.solve(
            dynamics,
            WBCTargets(
                base_linear_acc=base_linear_acc,
                base_angular_acc=base_angular_acc,
                joint_acc=joint_acc,
                contact_forces=forces,
                joint_weights=joint_weights,
                # Fewer contacts means less authority over the base, so the
                # base tasks stand down rather than demand what the soles
                # cannot deliver. Single support is half the contacts, and the
                # floor keeps the task from vanishing there.
                base_weight_scale=max(support, MIN_BASE_WEIGHT_SCALE),
                swing_jacobian=swing_jacobian,
                swing_acc=swing_acc,
                swing_orient_jacobian=swing_orient_jacobian,
                swing_orient_acc=swing_orient_acc,
            ),
            contact_mask=contact_mask,
        )

        swing = gait_out.swing
        self.telemetry = LocomotionTelemetry(
            com=dynamics.com,
            com_reference=com_ref,
            com_error=dynamics.com - com_ref,
            com_velocity=dynamics.com_vel,
            com_velocity_reference=com_vel_ref,
            rpy=rpy,
            torque=solution.torque,
            contact_forces=solution.forces,
            desired_forces=forces,
            contact_mask=contact_mask,
            mpc_cost=self.mpc.cost,
            solve_ms=(time.perf_counter() - started) * 1e3,
            walking=walking,
            gait_phase=float(gait_out.phase),
            swing_clearance=(
                float(feet[swing.leg, 2] - ground_z) if swing.active else 0.0
            ),
            swing_error=(
                float(np.linalg.norm(swing.des_pos - feet[swing.leg]))
                if swing.active
                else 0.0
            ),
            swing_tilt=(
                float(robot.sole_tilt(swing.leg)) if swing.active else 0.0
            ),
            yaw_ref=float(face_yaw),
            heading_goal=float(self.gait.config.heading),
            heading_error=float(wrap_angle(self.gait.config.heading - face_yaw)),
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
        yaw = None if self._last_gait is None else float(self._last_gait.stance_yaw)
        margin = self.robot.support_margin(yaw=yaw)
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
            # Keep LocomotionConfig.gait and scheduler in sync.
            gait_applied = self.gait.apply_updates(gait_updates)
            applied.update(gait_applied)
            self.config.gait = self.gait.config
            # Entering walk with zero speed: nudge a default crawl so mode alone works.
            if (
                "gait.mode" in gait_applied
                and gait_applied["gait.mode"] == MODE_WALK
                and self.gait.config.speed <= 1e-6
                and "gait.speed" not in gait_updates
                and "gait.heading" not in gait_updates
                and float(self.gait.config.goal_active) < 0.5
            ):
                self.gait.config.speed = 0.25
                applied["gait.speed"] = 0.25
            applied.update(self.gait.apply_speed_schedule())
            self.config.gait = self.gait.config
        return applied

    def qp_failures(self) -> dict[str, int]:
        """Solver fallback counts. Missing layers are omitted, not zero."""
        out: dict[str, int] = {}
        wbc = getattr(self, "wbc", None)
        mpc = getattr(self, "mpc", None)
        if wbc is not None:
            out["wbc"] = int(wbc._qp.failures)
        if mpc is not None:
            out["mpc"] = int(mpc._qp.failures)
        return out

    @property
    def last_gait(self):
        return self._last_gait

    def tunables(self) -> dict[str, float]:
        """Current value of every knob this stack accepts, by dotted path."""
        stack = getattr(self.config, "stack", "legacy_dcm")
        allowed = tuning.paths_for_stack(stack)
        return {
            path: value
            for path, value in tuning.snapshot(self.config).items()
            if path in allowed
        }

    def update_weights(self, **overrides: float) -> dict[str, float]:
        """Patch weights and gains in place; return the clamped values applied.

        Safe to call between control ticks while the robot is standing on the
        result: no OSQP problem is rebuilt, so there is no pause for the plant
        to fall into. Names are the dotted paths in
        :mod:`loka.control.tuning`, which disambiguate the several fields that
        exist on both layers with very different magnitudes.
        """
        stack = getattr(self.config, "stack", "legacy_dcm")
        disallowed = sorted(set(overrides) - tuning.paths_for_stack(stack))
        if disallowed:
            raise KeyError(f"Not tunable on stack {stack}: {disallowed}")
        applied = tuning.apply(self.config, overrides)
        groups = {tuning.BY_PATH[path].group for path in applied}
        if "mpc" in groups:
            self.mpc.refresh()
        if "wbc" in groups:
            self.wbc.refresh()
            self._refresh_posture_gains()
        return applied
