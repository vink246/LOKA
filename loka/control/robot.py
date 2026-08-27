"""Rigid-body model of the Unitree G1 used by the MPC and WBC layers.

The controller keeps its *own* ``MjModel`` / ``MjData`` pair, separate from
whatever is simulating (or driving) the plant. Everything the controller
believes about the robot -- masses, inertias, gear ratios, friction -- lives
here, so LOKA can mutate the controller's belief without touching the plant.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

#: Actuated joints, in Unitree DDS motor order: 12 leg, 3 waist, 14 arm.
GRAVITY = 9.81

NUM_JOINTS = 29
NUM_LEG_JOINTS = 12
#: Floating base occupies qpos[0:7] and qvel[0:6].
QPOS_JOINT0 = 7
QVEL_JOINT0 = 6

LEFT_FOOT_BODY = "left_ankle_roll_link"
RIGHT_FOOT_BODY = "right_ankle_roll_link"

#: Sole-plane contact points, ordered left-foot-first. Defined as sites in the
#: MJCF so the geometry stays in one place.
CONTACT_SITES = (
    "left_foot_c0",
    "left_foot_c1",
    "left_foot_c2",
    "left_foot_c3",
    "right_foot_c0",
    "right_foot_c1",
    "right_foot_c2",
    "right_foot_c3",
)
FOOT_CENTER_SITES = ("left_foot_center", "right_foot_center")


@dataclass
class DynamicsTerms:
    """Everything the whole-body QP needs about the current state.

    The equation of motion is ``M q̈ + h = Sᵀ τ + J_cᵀ f`` with ``h`` already
    net of passive (damping / spring) forces, since the plant applies those
    itself.
    """

    mass_matrix: np.ndarray  # (nv, nv)
    bias: np.ndarray  # (nv,) gravity + Coriolis - passive
    contact_jacobian: np.ndarray  # (3 * nc, nv)
    contact_bias_acc: np.ndarray  # (3 * nc,) J̇ q̇
    contact_pos: np.ndarray  # (nc, 3) world
    contact_vel: np.ndarray  # (nc, 3) world
    com: np.ndarray  # (3,) world
    com_vel: np.ndarray  # (3,) world


class G1Model:
    """Kinematics and dynamics queries for the 29-DoF G1."""

    def __init__(self, mjcf_path: str | Path) -> None:
        path = Path(mjcf_path)
        if not path.is_file():
            raise FileNotFoundError(f"G1 MJCF not found: {path}")
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self._scratch = mujoco.MjData(self.model)

        self.nv = int(self.model.nv)
        self.nu = int(self.model.nu)
        if self.nu != NUM_JOINTS:
            raise ValueError(f"Expected {NUM_JOINTS} actuators, model has {self.nu}")

        self.contact_site_ids = np.array(
            [self._site_id(name) for name in CONTACT_SITES], dtype=np.int32
        )
        self.foot_center_site_ids = np.array(
            [self._site_id(name) for name in FOOT_CENTER_SITES], dtype=np.int32
        )
        self.foot_body_ids = np.array(
            [self._body_id(LEFT_FOOT_BODY), self._body_id(RIGHT_FOOT_BODY)],
            dtype=np.int32,
        )
        self.num_contacts = len(self.contact_site_ids)

        self.total_mass = float(self.model.body_mass.sum())
        self.torque_limit = np.abs(self.model.actuator_ctrlrange[:, 1]).astype(float)
        self.nominal_qpos = self._keyframe_qpos("stand")

        self._jacp = np.zeros((3, self.nv))
        self._jacr = np.zeros((3, self.nv))
        self._full_m = np.zeros((self.nv, self.nv))

    # -- lookups ----------------------------------------------------------

    def _site_id(self, name: str) -> int:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise ValueError(f"Site '{name}' missing from model")
        return sid

    def _body_id(self, name: str) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Body '{name}' missing from model")
        return bid

    def _keyframe_qpos(self, name: str) -> np.ndarray:
        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, name)
        if kid < 0:
            raise ValueError(f"Keyframe '{name}' missing from model")
        return self.model.key_qpos[kid].copy()

    @property
    def nominal_joint_pos(self) -> np.ndarray:
        return self.nominal_qpos[QPOS_JOINT0:].copy()

    # -- state ------------------------------------------------------------

    def update(self, qpos: Sequence[float], qvel: Sequence[float]) -> None:
        """Refresh kinematics and dynamics quantities for the given state.

        Deliberately skips collision detection and constraint solving -- the
        controller only needs position, velocity and inertial quantities, and
        skipping the rest roughly halves the per-tick cost.
        """
        d = self.data
        d.qpos[:] = qpos
        d.qvel[:] = qvel
        m = self.model
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_crb(m, d)
        mujoco.mj_comVel(m, d)
        mujoco.mj_passive(m, d)
        mujoco.mj_rne(m, d, 0, d.qfrc_bias)

    # -- queries ----------------------------------------------------------

    def site_jacobian(self, site_id: int, data: mujoco.MjData | None = None) -> np.ndarray:
        d = self.data if data is None else data
        self._jacp[:] = 0.0
        mujoco.mj_jacSite(self.model, d, self._jacp, None, int(site_id))
        return self._jacp.copy()

    def contact_jacobian(self, data: mujoco.MjData | None = None) -> np.ndarray:
        """Stacked linear Jacobians of the sole contact points, ``(3nc, nv)``."""
        rows = [self.site_jacobian(sid, data) for sid in self.contact_site_ids]
        return np.vstack(rows)

    def _contact_bias_acc(self, qvel: np.ndarray, jac: np.ndarray) -> np.ndarray:
        """``J̇ q̇`` for the contact points, by finite-differencing ``J q̇``.

        Exact analytic ``J̇`` is not exposed by MuJoCo; a one-sided difference
        along the current velocity is cheap (one extra kinematics pass) and
        accurate to well below the contact-constraint tolerances.
        """
        eps = 1e-6
        s = self._scratch
        s.qpos[:] = self.data.qpos
        mujoco.mj_integratePos(self.model, s.qpos, qvel, eps)
        s.qvel[:] = qvel
        mujoco.mj_kinematics(self.model, s)
        mujoco.mj_comPos(self.model, s)
        jac_next = self.contact_jacobian(s)
        return (jac_next - jac) @ qvel / eps

    def site_bias_acc(self, site_id: int, qvel: np.ndarray) -> np.ndarray:
        """``J̇ q̇`` for one site via finite-differencing ``J q̇``."""
        eps = 1e-6
        jac = self.site_jacobian(site_id)
        s = self._scratch
        s.qpos[:] = self.data.qpos
        mujoco.mj_integratePos(self.model, s.qpos, qvel, eps)
        s.qvel[:] = qvel
        mujoco.mj_kinematics(self.model, s)
        mujoco.mj_comPos(self.model, s)
        jac_next = self.site_jacobian(site_id, s)
        return (jac_next - jac) @ qvel / eps

    def com_inertia(self) -> np.ndarray:
        """Composite rigid-body rotational inertia about the CoM, world frame.

        Parallel-axis sum over every body; only the MPC needs it, so it is
        computed on demand rather than as part of :meth:`dynamics`.
        """
        m, d = self.model, self.data
        com = d.subtree_com[0]
        mass = m.body_mass[1:]  # body 0 is the world
        rot = d.ximat[1:].reshape(-1, 3, 3)
        principal = m.body_inertia[1:]
        rotated = np.einsum("bij,bj,bkj->ik", rot, principal, rot)
        offset = d.xipos[1:] - com
        parallel = np.einsum("b,b->", mass, np.einsum("bi,bi->b", offset, offset)) * np.eye(3)
        parallel -= np.einsum("b,bi,bj->ij", mass, offset, offset)
        return rotated + parallel

    def dynamics(self, qvel: np.ndarray) -> DynamicsTerms:
        """Bundle the quantities the whole-body QP consumes."""
        m, d = self.model, self.data
        mujoco.mj_fullM(m, self._full_m, d.qM)
        jac = self.contact_jacobian()
        contact_pos = d.site_xpos[self.contact_site_ids].copy()
        contact_vel = (jac @ qvel).reshape(-1, 3)

        com_jac = np.zeros((3, self.nv))
        mujoco.mj_jacSubtreeCom(m, d, com_jac, 0)

        return DynamicsTerms(
            mass_matrix=self._full_m.copy(),
            bias=d.qfrc_bias - d.qfrc_passive,
            contact_jacobian=jac,
            contact_bias_acc=self._contact_bias_acc(qvel, jac),
            contact_pos=contact_pos,
            contact_vel=contact_vel,
            com=d.subtree_com[0].copy(),
            com_vel=com_jac @ qvel,
        )

    def foot_center_positions(self) -> np.ndarray:
        """World positions of the two foot-centre sites, ``(2, 3)``."""
        return self.data.site_xpos[self.foot_center_site_ids].copy()

    def support_margin(self) -> np.ndarray:
        """CoM distance to the support-polygon edge, ``[-x, +x, -y, +y]`` [m].

        Uses the axis-aligned bounding box of the planted sole points, which is
        exact for the rectangular double-support stance this controller holds.
        """
        contacts = self.data.site_xpos[self.contact_site_ids]
        com = self.data.subtree_com[0]
        return np.array(
            [
                com[0] - contacts[:, 0].min(),
                contacts[:, 0].max() - com[0],
                com[1] - contacts[:, 1].min(),
                contacts[:, 1].max() - com[1],
            ]
        )

    def capture_velocity_limit(self) -> np.ndarray:
        """Largest CoM velocity absorbable without stepping, per margin [m/s].

        The capture point of a linear inverted pendulum sits at ``c + ċ/ω``
        with ``ω = √(g/h)``; it has to land inside the support polygon or no
        contact force can arrest the fall. Multiplying each margin by ``ω``
        turns that into a velocity budget, which is the honest way to say how
        robust a *non-stepping* balance controller can possibly be.
        """
        contacts = self.data.site_xpos[self.contact_site_ids]
        height = float(self.data.subtree_com[0][2] - contacts[:, 2].mean())
        omega = np.sqrt(GRAVITY / height)
        return self.support_margin() * omega


def quat_to_rpy(quat: Sequence[float]) -> np.ndarray:
    """MuJoCo / Unitree ``[w, x, y, z]`` quaternion to roll-pitch-yaw."""
    w, x, y, z = (float(v) for v in quat)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def quat_to_mat(quat: Sequence[float]) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=float))
    return mat.reshape(3, 3)


def orientation_error(quat: Sequence[float], quat_des: Sequence[float]) -> np.ndarray:
    """Rotation vector taking the current orientation to the desired one.

    Expressed in the *body* frame, matching MuJoCo's free-joint angular
    velocity convention (``qvel[3:6]`` is body-frame ``ω``).
    """
    q_cur = np.asarray(quat, dtype=float)
    q_des = np.asarray(quat_des, dtype=float)
    q_err = np.zeros(4)
    q_cur_inv = np.zeros(4)
    mujoco.mju_negQuat(q_cur_inv, q_cur)
    mujoco.mju_mulQuat(q_err, q_cur_inv, q_des)
    rotvec = np.zeros(3)
    mujoco.mju_quat2Vel(rotvec, q_err, 1.0)
    return rotvec
