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
from collections import deque
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
from loka.control.contact_estimate import SWING_CONTACT_N, FootLoadEstimator
from loka.control.stabilizer import DCMStabilizer
from loka.control.terrain import STAIR_STEP_DZ, TerrainBelief, classical_stack

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
#: CoM height above the support while a height change is in the plan
#: horizon, and the lowest the reach clamp may take it [m]. The trailing leg
#: has to reach the lower tread while the CoM moves over the upper one. At
#: 0.60 the stance ankle hits its 0.87 rad dorsiflexion stop in a stair swing.
TERRAIN_WALK_HEIGHT = 0.68
TERRAIN_HEIGHT_MIN = 0.52
#: How far ahead of the pelvis a height change starts the stair crouch [m].
#: The height reference slews at 0.05 m/s, so a 0.1 m crouch needs 2 s.
TERRAIN_LOOKAHEAD = 0.5
#: On a step that changes tread, track the DCM plan's forward rate rather
#: than the cruise speed.
STAIR_PLAN_VELOCITY = True
#: Ceiling on that forward rate [m/s].
STAIR_SPEED_CAP = float("inf")
#: WBC pull toward the MPC contact forces on terrain, relative to the flat
#: weight. At the flat weight the WBC re-solves the distribution from its
#: own base and posture tasks, and across a riser it unloads the trailing
#: foot the MPC is pushing with, braking the CoM short of the upper tread.
TERRAIN_FORCE_WEIGHT_SCALE = 100.0
#: Waist-pitch lean on a step that changes tread [rad]; 0 is upright.
#: Forward lean keeps the hips over the upper tread (stair_feasibility.py).
#: An empirical heuristic for this stack: model-based stair controllers in
#: the literature keep the torso upright.
TERRAIN_TORSO_LEAN = 0.15
#: ``step``: full lean exactly while a step changes tread, none otherwise.
#: ``ramp``: fade in while a climb is in the plan horizon and out after it,
#: at TORSO_LEAN_RATE. ``off``: upright.
TERRAIN_LEAN_MODE = "ramp"
TORSO_LEAN_RATE = 0.3  # rad/s
WAIST_PITCH = 14
TORSO_LEAN_KP = 100.0
TORSO_LEAN_KD = 20.0
#: Largest CoM-to-sole distance a planted leg can hold [m]. The nominal
#: stance is 0.70 m tall with each sole 0.12 m to the side.
LEG_REACH = 0.78  # with trailing-foot toe-off; 0.72 flat-footed
#: Soft leg joint limits on terrain: margin from each stop [rad], the
#: approach-speed gain [1/s], damping [1/s] and task weight.
JOINT_LIMIT_MARGIN = 0.10
JOINT_LIMIT_GAIN = 5.0
JOINT_LIMIT_DAMPING = 20.0
JOINT_LIMIT_WEIGHT = 50.0
#: Hard joint-range rows in the terrain-mode WBC (see _joint_acc_bounds).
#: Off: on 0.10 m stairs they lowered the mean highest tread from 0.42 m to
#: 0.36 m and left the curb at 5/5; the soft dampers stay.
JOINT_LIMIT_ROWS = False
JOINT_LIMIT_ROW_MARGIN = 0.01  # rad
JOINT_LIMIT_HORIZON = 0.05  # s
#: A swing foot this far above its target that already carries load met
#: ground early [m].
EARLY_CONTACT_HEIGHT = 0.015
#: Swing phase before which load on the foot is a scuff, not a landing.
EARLY_CONTACT_MIN_S = 0.30
#: Load on a swinging foot, in the middle of its swing, that means it hit
#: something [N]. A free swing reads under 10 N on the flat walk.
OBSTRUCTION_N = 40.0
OBSTRUCTION_MIN_S = 0.05
OBSTRUCTION_MAX_S = 0.70
#: Position feedback folded into the CoM velocity reference of a
#: quasi-static stair step [1/s], and its cap [m/s].
QS_POSITION_GAIN = 3.0
QS_PULL_SPEED = 0.25
#: Base task weight scale through a quasi-static stair step. The swing foot
#: travels a whole tread, and at the walking weights its task (500) dragged
#: the pelvis after it and pitched the torso over the stance toe.
QS_BASE_WEIGHT_SCALE = 4.0
#: Stance-knee bias weight on terrain. Without a bias a crouched walk falls
#: within two seconds; the target is geometric, so it agrees with the CoM task.
TERRAIN_KNEE_W = 12.0


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
        #: Terrain-mode QP: the same problem plus hard joint-range rows on the
        #: legs. A separate instance, so the flat QP's structure is untouched.
        self._wbc_terrain: WholeBodyController | None = None
        #: Classical terrain stack's DCM stabilizer (terrain.TERRAIN_STACK).
        self._stabilizer: DCMStabilizer | None = None
        #: Terrain the controller believes in. Empty is the flat-floor controller.
        self.terrain = TerrainBelief()
        import mujoco

        self._leg_body_ids = {
            side: tuple(
                mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_{link}")
                for link in ("hip_pitch_link", "knee_link", "ankle_pitch_link")
            )
            for side in ("left", "right")
        }
        self._load_estimator = FootLoadEstimator(self.robot)
        #: World force on each foot, from joint torques (see contact_estimate).
        self.foot_load = np.zeros((2, 3))

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

    def recalibrate_stance(self) -> None:
        """Re-measure the nominal stance after the model's mass changes."""
        self._calibrate_nominal_pose()

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
        self.gait.terrain = self.terrain
        #: Ground height under each foot, from where its sole last rested.
        self._leg_z = np.full(2, self._ground_height)
        self.foot_load = np.zeros((2, 3))
        #: Footfalls, early contacts and swing obstructions, for the runtime to
        #: drain. Bounded, so a plant with no runtime does not accumulate them.
        self.terrain_events: deque = deque(maxlen=200)
        self._obstruction_key = None
        self._lean = 0.0
        self._was_classical = False
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
        if self.terrain.active:
            return self._terrain_contact_mask(heights)
        mask = heights < self._ground_height + CONTACT_HEIGHT_TOLERANCE
        if mask.any():
            self._ground_height = float(heights[mask].min())
        # Kept current so a terrain belief that arrives mid-run starts from
        # where the soles are.
        per = self.robot.num_contacts // 2
        for leg in (0, 1):
            rows = slice(leg * per, (leg + 1) * per)
            if mask[rows].any():
                self._leg_z[leg] = float(heights[rows][mask[rows]].min())
        return mask

    def _terrain_contact_mask(self, heights: np.ndarray) -> np.ndarray:
        """Per-foot contact on terrain, from kinematics and leg-torque load.

        A planted foot is compared against the height its own sole last
        rested at, so a foot on a tread and a foot on the floor are both
        down. A swinging foot counts only once its leg carries load: the
        arc ends at the believed tread, and a tread that is lower than
        believed would otherwise be "touched" in mid-air.
        """
        per = self.robot.num_contacts // 2
        mask = np.zeros(self.robot.num_contacts, dtype=bool)
        last = self._last_gait
        swing = last.swing if last is not None and last.walking else None
        for leg in (0, 1):
            rows = slice(leg * per, (leg + 1) * per)
            h = heights[rows]
            loaded = float(self.foot_load[leg, 2]) > SWING_CONTACT_N
            if swing is not None and swing.active and int(swing.leg) == leg:
                if not loaded:
                    continue
                m = h < float(h.min()) + CONTACT_HEIGHT_TOLERANCE
            else:
                m = h < self._leg_z[leg] + CONTACT_HEIGHT_TOLERANCE
                if not m.any() and loaded:
                    m = h < float(h.min()) + CONTACT_HEIGHT_TOLERANCE
            if m.any():
                self._leg_z[leg] = float(h[m].min())
            mask[rows] = m
        floor = heights < self._ground_height + CONTACT_HEIGHT_TOLERANCE
        if floor.any():
            self._ground_height = float(heights[floor].min())
        return mask

    def _geometric_knees(self, com_error: np.ndarray) -> np.ndarray:
        """Knee angles that place each hip at its current position plus the CoM error.

        Two-link leg: hip-pitch joint to knee to ankle-pitch joint, lengths
        read off the belief kinematics. On stairs the two knees differ, and a
        table indexed by one CoM height pulls both toward a symmetric stance.
        """
        data = self.robot.data
        knees = np.zeros(2)
        shift = np.asarray(com_error, dtype=float).reshape(3)
        for leg, side in enumerate(("left", "right")):
            hip = data.xpos[self._leg_body_ids[side][0]]
            knee = data.xpos[self._leg_body_ids[side][1]]
            ankle = data.xpos[self._leg_body_ids[side][2]]
            l1 = float(np.linalg.norm(knee - hip))
            l2 = float(np.linalg.norm(ankle - knee))
            reach = float(np.linalg.norm(hip + shift - ankle))
            cos_inner = (l1 * l1 + l2 * l2 - reach * reach) / (2.0 * l1 * l2)
            knees[leg] = float(np.pi - np.arccos(np.clip(cos_inner, -1.0, 1.0)))
        return np.clip(knees, STANCE_KNEE_MIN, 2.2)

    def _terrain_near(self, xy: np.ndarray, yaw: float) -> bool:
        """A height change in the belief within :data:`TERRAIN_LOOKAHEAD` ahead or under the feet."""
        forward = np.array([np.cos(yaw), np.sin(yaw)])
        dist = np.linspace(-0.3, TERRAIN_LOOKAHEAD, 19)
        points = xy[None, :] + dist[:, None] * forward[None, :]
        heights = self.terrain.heights_at(points, self._ground_height)
        return float(heights.max() - heights.min()) >= STAIR_STEP_DZ

    def _toe_off(self, contact_mask, feet, com, yaw: float, com_vel=None) -> np.ndarray:
        """Trailing-foot toe-off between treads (Englsberger 2014; Griffin 2017).

        In double support with the feet on different treads, once the CoM is
        past the lower foot's toes its heel can carry no load and holding it
        flat forces the deep crouch that drives the stance ankle into its
        stop. The heel points leave the contact set; the foot then pivots on
        its toes, with pitch left free.
        """
        per = self.robot.num_contacts // 2
        if not np.all(contact_mask) or abs(float(self._leg_z[0] - self._leg_z[1])) < STAIR_STEP_DZ:
            return contact_mask
        lower = int(np.argmin(self._leg_z))
        upper = 1 - lower
        sites = self.robot.data.site_xpos[self.robot.contact_site_ids][:, :2]
        toes = sites[lower * per + 2 : lower * per + 4]
        polygon = np.vstack([toes, sites[upper * per : (upper + 1) * per]])
        forward = np.array([np.cos(yaw), np.sin(yaw)])
        left = np.array([-forward[1], forward[0]])
        # Capture point, as in IHMC's toe-off gate: it must lie in the toe-off
        # support polygon (trailing toes plus the leading foot), here its
        # bounding box in the travel frame, and past the trailing toes.
        omega = float(np.sqrt(9.81 / max(float(com[2] - self._leg_z.min()), 0.3)))
        dcm = np.asarray(com[:2]) + np.asarray(com_vel[:2]) / omega
        along = polygon @ forward
        across = polygon @ left
        d_along, d_across = float(dcm @ forward), float(dcm @ left)
        inside = (
            float((toes @ forward).max()) < d_along <= float(along.max())
            and float(across.min()) <= d_across <= float(across.max())
        )
        if not inside:
            return contact_mask
        mask = contact_mask.copy()
        mask[lower * per : lower * per + 2] = False  # heel points c0, c1
        return mask

    def _joint_limit_dampers(self, q, dq, joint_acc, joint_weights, acc_cap) -> None:
        """Soft joint-position limits on the legs: a damper inside a margin of each stop.

        The QP has no joint-range rows; on terrain the ankles reach their
        ±15° roll and 0.87 rad dorsiflexion stops, where the plant silently
        blocks commanded motion. Inside JOINT_LIMIT_MARGIN the joint's target
        acceleration brakes motion toward the stop (velocity damper, as in
        Faverjon and Tournassoud; Celerier et al. 2025).
        """
        ranges = self.robot.model.jnt_range[1 : 1 + NUM_LEG_JOINTS]
        for i in range(NUM_LEG_JOINTS):
            lo, hi = float(ranges[i, 0]), float(ranges[i, 1])
            if hi - lo <= 1e-6:
                continue
            d_hi, d_lo = hi - float(q[i]), float(q[i]) - lo
            if d_hi < JOINT_LIMIT_MARGIN and dq[i] > -0.05:
                vel = -JOINT_LIMIT_GAIN * (JOINT_LIMIT_MARGIN - d_hi)
            elif d_lo < JOINT_LIMIT_MARGIN and dq[i] < 0.05:
                vel = JOINT_LIMIT_GAIN * (JOINT_LIMIT_MARGIN - d_lo)
            else:
                continue
            joint_acc[i] = float(np.clip(JOINT_LIMIT_DAMPING * (vel - dq[i]), -acc_cap, acc_cap))
            joint_weights[i] = max(float(joint_weights[i]), JOINT_LIMIT_WEIGHT)

    def _stabilizer_forces(self, dynamics, gait_out, com_ref, com_vel_ref, contact_mask,
                           base_quat, rot, face_yaw) -> np.ndarray:
        """Classical stack: DCM feedback to a CMP, distributed over the soles."""
        if self._stabilizer is None:
            wbc_cfg = self.config.wbc
            self._stabilizer = DCMStabilizer(
                self.robot.num_contacts, self.robot.total_mass,
                mu=float(wbc_cfg.friction_mu), fz_min=float(wbc_cfg.fz_min),
                fz_max=float(wbc_cfg.fz_max),
            )
        omega = self.gait.omega
        dcm_ref = np.asarray(gait_out.dcm_ref, dtype=float).reshape(2)
        dcm_rate_ref = omega * (dcm_ref - np.asarray(self.gait.vrp_now(), dtype=float).reshape(2))
        yaw_quat = np.array([np.cos(0.5 * face_yaw), 0.0, 0.0, np.sin(0.5 * face_yaw)])
        force, moment = self._stabilizer.desired_wrench(
            com=dynamics.com,
            com_vel=dynamics.com_vel,
            com_z_ref=float(com_ref[2]),
            com_vz_ref=float(com_vel_ref[2]),
            dcm_ref=dcm_ref,
            dcm_rate_ref=dcm_rate_ref,
            omega=omega,
            orientation_error=rot @ orientation_error(base_quat, yaw_quat),
            angular_momentum=self.robot.angular_momentum_about(dynamics.com),
            inertia=self.robot.com_inertia(),
        )
        return self._stabilizer.distribute(force, moment, dynamics.com, dynamics.contact_pos, contact_mask)

    def _torso_lean(self, active: bool, gait_out, dt: float) -> float:
        """Waist-pitch lean for this tick [rad], per TERRAIN_LEAN_MODE."""
        if classical_stack():
            # The classical stack keeps the trunk upright.
            self._lean = 0.0
            return 0.0
        if TERRAIN_LEAN_MODE == "off" or TERRAIN_TORSO_LEAN <= 0.0:
            self._lean = 0.0
            return 0.0
        if TERRAIN_LEAN_MODE == "step":
            self._lean = TERRAIN_TORSO_LEAN if active and gait_out.stair_transfer else 0.0
            return self._lean
        target = TERRAIN_TORSO_LEAN if active and (gait_out.ascent_ahead or gait_out.stair_transfer) else 0.0
        step = TORSO_LEAN_RATE * dt
        self._lean += float(np.clip(target - self._lean, -step, step))
        if self._lean < 1e-4:
            self._lean = 0.0
        return self._lean

    def _terrain_wbc(self) -> WholeBodyController:
        if self._wbc_terrain is None:
            self._wbc_terrain = WholeBodyController(
                self.config.wbc,
                nv=self.robot.nv,
                num_joints=self.robot.nu,
                num_contacts=self.robot.num_contacts,
                torque_limit=self.robot.torque_limit,
                limited_joints=np.arange(NUM_LEG_JOINTS),
            )
        return self._wbc_terrain

    def _joint_acc_bounds(self, q, dq, acc_cap: float) -> tuple[np.ndarray, np.ndarray]:
        """Hard leg joint-range rows: no stop reached within one preview horizon.

        ``q + q̇T + ½q̈T² ∈ [lo + m, hi − m]`` gives a bound on q̈ (the
        discrete velocity-damper form of Faverjon and Tournassoud; Celerier
        et al. 2025). Capped at the joint acceleration limit and kept
        ordered, so a joint already past the margin is pushed back rather
        than making the QP infeasible.
        """
        ranges = self.robot.model.jnt_range[1 : 1 + NUM_LEG_JOINTS]
        qi = np.asarray(q[:NUM_LEG_JOINTS], dtype=float)
        dqi = np.asarray(dq[:NUM_LEG_JOINTS], dtype=float)
        t = JOINT_LIMIT_HORIZON
        lo = 2.0 * (ranges[:, 0] + JOINT_LIMIT_ROW_MARGIN - qi - dqi * t) / (t * t)
        hi = 2.0 * (ranges[:, 1] - JOINT_LIMIT_ROW_MARGIN - qi - dqi * t) / (t * t)
        lo = np.clip(lo, -acc_cap, acc_cap)
        hi = np.clip(hi, -acc_cap, acc_cap)
        mid = 0.5 * (lo + hi)
        swap = lo > hi
        lo[swap] = mid[swap]
        hi[swap] = mid[swap]
        return lo, hi

    def planted_mask(self) -> np.ndarray:
        """Sole points on the ground at the belief's current kinematics.

        Same rule as the controller's own contact set, without updating any
        estimate. On terrain each foot is compared with its own ground.
        """
        heights = self.robot.data.site_xpos[self.robot.contact_site_ids][:, 2]
        if not self.terrain.active:
            return heights < self._ground_height + CONTACT_HEIGHT_TOLERANCE
        per = self.robot.num_contacts // 2
        ref = np.repeat(self._leg_z, per)
        mask = heights < ref + CONTACT_HEIGHT_TOLERANCE
        last = self._last_gait
        swing = last.swing if last is not None and last.walking else None
        if swing is not None and swing.active:
            # As in _terrain_contact_mask: a swinging foot is down only once
            # its leg carries load; stepping down, its sole passes below the
            # tread it left while still in the air.
            leg = int(swing.leg)
            if float(self.foot_load[leg, 2]) <= SWING_CONTACT_N:
                mask[leg * per : (leg + 1) * per] = False
        return mask

    def _reach_clamp(self, com_ref, com_vel_ref, feet, contact_mask):
        """Keep the CoM height reference within reach of every planted foot.

        Only binds while the feet are on different treads: the trailing leg
        cannot hold the CoM at stance height above the upper tread.
        """
        if abs(float(self._leg_z[0] - self._leg_z[1])) < STAIR_STEP_DZ:
            return com_ref, com_vel_ref
        per = self.robot.num_contacts // 2
        ceiling = np.inf
        for leg in (0, 1):
            if not np.asarray(contact_mask[leg * per : (leg + 1) * per]).any():
                continue
            dxy = float(np.linalg.norm(com_ref[:2] - feet[leg, :2]))
            reach = float(np.sqrt(max(LEG_REACH * LEG_REACH - dxy * dxy, 0.0)))
            ceiling = min(ceiling, float(self._leg_z[leg]) + reach)
        floor = float(self._leg_z.max()) + TERRAIN_HEIGHT_MIN
        ceiling = max(ceiling, floor)
        if com_ref[2] > ceiling:
            com_ref = com_ref.copy()
            com_ref[2] = ceiling
            com_vel_ref = com_vel_ref.copy()
            com_vel_ref[2] = min(float(com_vel_ref[2]), 0.0)
        return com_ref, com_vel_ref

    @property
    def load_estimator(self) -> FootLoadEstimator:
        """Leg-torque contact estimate from the last :meth:`sense_proprioception`."""
        return self._load_estimator

    def sense_proprioception(self, qpos, qvel, qacc, joint_torque) -> None:
        """Encoder state, its derivative and motor torque after a plant step.

        Feeds the foot-load estimate. Nothing here is a contact or force
        reading from the simulator.
        """
        self.foot_load = self._load_estimator.update(qpos, qvel, qacc, joint_torque)

    def _check_obstruction(self, feet: np.ndarray) -> None:
        """A swing foot carrying load mid-swing hit something: a riser or a curb.

        Read from the leg's joint torques (see contact_estimate), on any
        belief, flat or not. One event per swing. Late swing is left to the
        touchdown logic, where load is the landing itself.
        """
        last = self._last_gait
        if last is None or not last.walking or not last.swing.active:
            return
        swing = last.swing
        leg = int(swing.leg)
        if not (OBSTRUCTION_MIN_S <= swing.s <= OBSTRUCTION_MAX_S):
            return
        load = self.foot_load[leg]
        if float(np.linalg.norm(load)) <= OBSTRUCTION_N:
            return
        key = (int(self.gait.step_index), leg)
        if key == self._obstruction_key:
            return
        self._obstruction_key = key
        self.terrain_events.append({
            "kind": "swing_obstructed",
            "t": self._tick * self.config.control_dt,
            "leg": leg,
            "s": float(swing.s),
            "xy": [float(v) for v in feet[leg, :2]],
            "foot_z": float(feet[leg, 2]),
            "target_z": float(swing.foothold[2]),
            "load_n": [float(v) for v in load],
        })

    def _check_early_contact(self, feet: np.ndarray) -> None:
        """A swing foot carrying load above its target met a higher tread."""
        last = self._last_gait
        if last is None or not last.walking or not last.swing.active:
            return
        swing = last.swing
        leg = int(swing.leg)
        if swing.s < EARLY_CONTACT_MIN_S or float(self.foot_load[leg, 2]) <= SWING_CONTACT_N:
            return
        target = float(self.gait._landing_z(self._ground_height))
        sole = self.robot.data.site_xpos[self.robot.contact_site_ids][
            leg * 4 : (leg + 1) * 4, 2
        ]
        measured = float(sole.min())
        if measured <= target + EARLY_CONTACT_HEIGHT:
            return
        self.gait.note_measured_landing(measured)
        self.terrain_events.append({
            "kind": "early_contact",
            "t": self._tick * self.config.control_dt,
            "leg": leg,
            "xy": [float(v) for v in feet[leg, :2]],
            "believed_z": target,
            "measured_z": measured,
            "load_n": [float(v) for v in self.foot_load[leg]],
        })

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
        absolute: bool = False,
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
            if not absolute:
                seq[:, rows, 2] = ground_z + pose[:, leg, 2, None]
                continue
            # Terrain: heights are world z. A foot planned to stay where it
            # stands uses its measured sites.
            here = np.linalg.norm(pose[:, leg, :2] - feet[leg, :2], axis=1) < 0.01
            seq[:, rows, 2] = pose[:, leg, 2, None] + offsets[rows, 2][None, :]
            if here.any():
                seq[here, rows, 2] = contact_pos[rows, 2][None, :]
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
        terrain = self.terrain.active
        measured_mask = self._contact_mask()
        feet = robot.foot_center_positions()
        foot_mid = feet.mean(axis=0)
        ground_z = self._ground_height
        self._check_obstruction(feet)
        if terrain:
            self._check_early_contact(feet)

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
            last = self._last_gait
            if terrain and (
                (last is not None and last.terrain_ahead)
                or self._terrain_near(np.asarray(qpos[:2], dtype=float), float(rpy[2]))
            ):
                goal_h = min(max(height_cmd, TERRAIN_HEIGHT_MIN), TERRAIN_WALK_HEIGHT)
            step_h = HEIGHT_SLEW_WALK * cfg.control_dt
            self._height_ref += float(np.clip(goal_h - self._height_ref, -step_h, step_h))
        elif terrain:
            # Terrain walking may have lowered the CoM; rise back to the stand
            # height at the walking slew rather than in one tick.
            step_h = HEIGHT_SLEW_WALK * cfg.control_dt
            self._height_ref += float(np.clip(height_cmd - self._height_ref, -step_h, step_h))
        else:
            self._height_ref = height_cmd
        height = float(self._height_ref)

        stance_i = int(np.argmin(feet[:, 2]))
        if terrain and self._last_gait is not None and self._last_gait.support_leg >= 0:
            # On stairs the stance foot is often the higher one.
            stance_i = int(self._last_gait.support_leg)
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
            pelvis_xy=np.asarray(qpos[:2], dtype=float),
            support_z=self._leg_z.copy() if terrain else None,
        )
        self._last_gait = gait_out
        walking = gait_out.walking
        if self.gait.touchdowns:
            for record in self.gait.touchdowns:
                self.terrain_events.append({
                    "kind": "touchdown",
                    "t": self._tick * cfg.control_dt,
                    **record,
                })
            self.gait.touchdowns.clear()

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
                seated = float(feet[swing.leg, 2] - self.gait._landing_z(ground_z)) <= TOUCHDOWN_HEIGHT
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
            # Along the plan's facing, track the ramped speed command. The DCM
            # rate can point backwards when the plan lags; its lateral part is
            # the step orbit and has to stay. The opening orbit commands about
            # 0.25 m/s sideways before the stride exists. The closed loop is
            # late reversing that, so the first two footholds step right. Hold
            # the pelvis over the feet until both of those steps are done.
            lateral = float(np.dot(gait_out.com_vel_ref, left))
            com_xy = np.asarray(gait_out.com_ref_xy, dtype=float).copy()
            if self.gait.step_index < 3:
                lateral = 0.0
                com_xy = com_xy + float(np.dot(dynamics.com[:2] - com_xy, left)) * left
            base_z, base_vz = ground_z, 0.0
            if terrain and gait_out.support_z is not None:
                base_z, base_vz = float(gait_out.support_z), float(gait_out.support_vz)
            com_ref = np.array(
                [
                    com_xy[0] + lean[0],
                    com_xy[1] + lean[1],
                    base_z + height,
                ]
            )
            travel = float(self.gait.cmd_speed) * forward + lateral * left
            if STAIR_PLAN_VELOCITY and terrain and gait_out.stair_transfer and not gait_out.stair_step:
                # A step over a riser is at least a sole length plus two edge
                # margins, several cruise strides; the CoM covers it in one
                # step period only if it tracks the plan's own rate.
                plan_speed = float(np.dot(np.asarray(gait_out.com_vel_ref, dtype=float), forward))
                speed = min(max(plan_speed, float(self.gait.cmd_speed)), STAIR_SPEED_CAP)
                travel = speed * forward + lateral * left
            if terrain and gait_out.stair_step:
                # Quasi-static stair step: the gait's own profile, not the
                # cruise speed, plus position feedback. The MPC may weight
                # horizontal position at zero (velocity tracking only), and
                # then nothing but this term moves the CoM onto the sole.
                com_xy = np.asarray(gait_out.com_ref_xy, dtype=float).copy()
                com_ref[:2] = com_xy + lean
                pull = QS_POSITION_GAIN * (com_ref[:2] - dynamics.com[:2])
                norm = float(np.linalg.norm(pull))
                if norm > QS_PULL_SPEED:
                    pull *= QS_PULL_SPEED / norm
                travel = np.asarray(gait_out.com_vel_ref, dtype=float).reshape(2) + pull
            com_vel_ref = np.array([travel[0], travel[1], base_vz])
            face_yaw = float(gait_out.yaw_ref)
            yaw_rate_ref = float(gait_out.yaw_rate_ref)
        else:
            contact_mask = measured_mask
            foot_yaws = robot.foot_yaws()
            stance_yaw = float(circ_mid(foot_yaws[0], foot_yaws[1]))
            forward, left = heading_frame(stance_yaw)
            offset = self.nominal_com_offset[:2] + self.command.com_offset_xy
            lean = offset[0] * forward + offset[1] * left
            base_z = float(self._leg_z.mean()) if terrain else ground_z
            com_ref = np.array(
                [foot_mid[0] + lean[0], foot_mid[1] + lean[1], base_z + height]
            )
            com_vel_ref = np.zeros(3)
            face_yaw = stance_yaw + float(self.command.yaw)
            yaw_rate_ref = 0.0
        if terrain:
            com_ref, com_vel_ref = self._reach_clamp(com_ref, com_vel_ref, feet, contact_mask)

        classical = bool(terrain and walking and classical_stack())
        # --- centroidal MPC (decimated) ---
        # The first tick after a classical walk solves at once: the stored
        # forces are the stabilizer's, for the walking contact set.
        mpc_due = self._tick % cfg.mpc_decimation == 0 or self._was_classical
        self._was_classical = classical
        if mpc_due and not classical:
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
                    absolute=terrain,
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

        if terrain and walking:
            contact_mask = self._toe_off(
                contact_mask, feet, dynamics.com, float(rpy[2]), com_vel=dynamics.com_vel
            )
        if classical:
            self._desired_forces = self._stabilizer_forces(
                dynamics, gait_out, com_ref, com_vel_ref, contact_mask, base_quat, rot, face_yaw
            )
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
        if terrain and walking and gait_out.stair_step:
            # A stair step dwells in single support; the CoM has to be where
            # the plan put it before the foot lifts, not 3 cm off.
            kp_xy = wbc_cfg.kp_base_position
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
        qs_planted = bool(terrain and walking and gait_out.stair_step and np.all(contact_mask))
        if walking and not qs_planted:
            # Hip yaw holds the foot square to the pelvis yaw the base task
            # is tracking, not to the standing keyframe. A turn otherwise
            # asks the torso to yaw while the hip pins the foot to yaw 0.
            # Two planted feet in a quasi-static stair step pin the feet
            # already, and the hold then blocks the sideways weight shift.
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

        if not walking:
            # A stand that starts on a toe-first landing sees only the toe
            # sites. The QP then loads the toe, the ankle torque lifts the
            # heel further, and nothing brings it back. Level that sole about
            # the sites that are down. Yaw stays free. With both soles rocked
            # the whole body is tilted, and leveling one foot fights that.
            per_foot = robot.num_contacts // 2
            planted = [int(measured_mask[leg * per_foot : (leg + 1) * per_foot].sum()) for leg in (0, 1)]
            partial = [leg for leg in (0, 1) if 0 < planted[leg] < per_foot]
            if len(partial) == 1 and planted[1 - partial[0]] == per_foot:
                leg = partial[0]
                site_id = int(robot.foot_center_site_ids[leg])
                _, j_w = robot.site_spatial_jacobian(site_id)
                _, bias_ang = robot.site_spatial_bias_acc(site_id, qvel_arr)
                foot_quat = robot.site_quat(site_id)
                yaw = float(foot_yaws_now[leg])
                flat = np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)])
                err_world = quat_to_mat(foot_quat) @ orientation_error(foot_quat, flat)
                omega = j_w @ qvel_arr
                level_acc = np.clip(
                    wbc_cfg.kp_swing_orient * err_world
                    - wbc_cfg.kd_swing_orient * omega
                    - bias_ang,
                    -cfg.max_swing_ang_acc,
                    cfg.max_swing_ang_acc,
                )
                swing_orient_jacobian = j_w.copy()
                swing_orient_jacobian[2] = 0.0
                swing_orient_acc = level_acc.copy()
                swing_orient_acc[2] = 0.0

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
                    knee_w = STANCE_KNEE_W
                    if terrain:
                        # Each knee bends to put its hip where the CoM
                        # reference wants it, whatever the other leg does.
                        knees = self._geometric_knees(com_ref - dynamics.com)
                        knee_w = TERRAIN_KNEE_W
                        if gait_out.stair_step and bool(np.all(contact_mask)):
                            # Two planted feet hold the knees, as in a stand.
                            knee_w = 0.0
                    else:
                        knees = self._knee_bias(height)
                    if knee_w > 0.0:
                        hold_joint(
                            knee,
                            STANCE_KNEE_KP,
                            STANCE_KNEE_KD,
                            knee_w,
                            float(knees[leg_i]),
                        )
                knee_guard[leg_i] = True

        lean = self._torso_lean(terrain and walking, gait_out, cfg.control_dt)
        if lean > 0.0:
            # Forward torso lean over the upper tread (stair_feasibility.py).
            hold_joint(WAIST_PITCH, TORSO_LEAN_KP, TORSO_LEAN_KD,
                       float(cfg.wbc.weight_posture_upper),
                       float(self.nominal_joint_pos[WAIST_PITCH]) + lean)

        if terrain:
            self._joint_limit_dampers(qj, dqj, joint_acc, joint_weights, cfg.max_joint_acc)

        self._last_joint_weights = joint_weights
        self._last_swing_orient_active = swing_orient_jacobian is not None
        self._last_knee_guard = knee_guard

        wbc = self._terrain_wbc() if terrain and JOINT_LIMIT_ROWS else self.wbc
        solution = wbc.solve(
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
                base_weight_scale=(
                    QS_BASE_WEIGHT_SCALE
                    if terrain and walking and gait_out.stair_step
                    else max(support, MIN_BASE_WEIGHT_SCALE)
                ),
                swing_jacobian=swing_jacobian,
                swing_acc=swing_acc,
                swing_orient_jacobian=swing_orient_jacobian,
                swing_orient_acc=swing_orient_acc,
                force_weight_scale=TERRAIN_FORCE_WEIGHT_SCALE if terrain else 1.0,
                joint_acc_bounds=(
                    self._joint_acc_bounds(qj, dqj, cfg.max_joint_acc)
                    if terrain and JOINT_LIMIT_ROWS
                    else None
                ),
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
                float(feet[swing.leg, 2] - self.gait._landing_z(ground_z)) if swing.active else 0.0
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
            terrain_wbc = getattr(self, "_wbc_terrain", None)
            if terrain_wbc is not None:
                out["wbc"] += int(terrain_wbc._qp.failures)
        if mpc is not None:
            out["mpc"] = int(mpc._qp.failures)
        stabilizer = getattr(self, "_stabilizer", None)
        if stabilizer is not None:
            out["stabilizer"] = int(stabilizer._qp.failures)
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
            if self._wbc_terrain is not None:
                self._wbc_terrain.refresh()
            # The stabilizer bakes friction and force bounds into its QP; it is
            # rebuilt from the patched config on the next classical tick.
            self._stabilizer = None
            self._refresh_posture_gains()
        return applied
