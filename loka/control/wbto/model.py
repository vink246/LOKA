"""Pinocchio twin of the controller's MuJoCo belief model.

Pinocchio parses the same robot MJCF the controller loads, so masses,
inertias, rotor armature, joint limits and the sole sites are identical to the
belief ``MjModel`` (mass matrix and inverse dynamics agree to 1e-13). The arms
are locked at the stand pose, which leaves the floating base, both legs and
the waist: 21 velocities instead of 35.

MuJoCo and Pinocchio disagree on two conventions for a free joint, and every
state crossing the boundary goes through :meth:`WholeBodyModel.state_from_mujoco`:

- quaternion order: MuJoCo ``[w, x, y, z]``, Pinocchio ``[x, y, z, w]``;
- base linear velocity: MuJoCo world frame, Pinocchio body frame. The angular
  velocity is body frame in both.
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pinocchio as pin

FOOT_FRAMES = ("left_foot_center", "right_foot_center")
#: Joints removed from the optimisation and held at the stand pose.
LOCKED_JOINT_KEYS = ("shoulder", "elbow", "wrist")
#: Sole extent about the foot-centre frame [m]: the plant's contact spheres
#: sit at -0.05 and +0.12 along the foot from the ankle, +-0.025 across.
SOLE_HALF_LENGTH = 0.085
SOLE_HALF_WIDTH = 0.025


def robot_mjcf(model_path: str | Path) -> Path:
    """The robot file behind a scene file (Pinocchio does not follow ``<include>``)."""
    path = Path(model_path)
    root = ET.parse(path).getroot()
    if root.find("worldbody/body") is not None:
        return path
    for include in root.iter("include"):
        candidate = path.parent / include.get("file", "")
        if candidate.is_file() and ET.parse(candidate).getroot().find("worldbody/body") is not None:
            return candidate
    raise ValueError(f"no robot body found in {path} or its includes")


class WholeBodyModel:
    """Reduced Pinocchio model plus the maps to and from MuJoCo coordinates."""

    def __init__(self, model_path: str | Path, keyframe: str = "stand") -> None:
        self.mj_model = mujoco.MjModel.from_xml_path(str(model_path))
        self.full = pin.buildModelFromMJCF(str(robot_mjcf(model_path)))
        if self.full.nv != self.mj_model.nv:
            raise ValueError(
                f"Pinocchio parsed {self.full.nv} velocities, MuJoCo has {self.mj_model.nv}"
            )
        key = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if key < 0:
            raise ValueError(f"keyframe '{keyframe}' missing from {model_path}")
        self.mj_qpos0 = self.mj_model.key_qpos[key].copy()
        q0_full = self.mj_qpos0.copy()
        q0_full[3:7] = self.mj_qpos0[[4, 5, 6, 3]]

        locked = [
            self.full.getJointId(name)
            for name in self.full.names
            if any(key in name for key in LOCKED_JOINT_KEYS)
        ]
        self.model = pin.buildReducedModel(self.full, locked, q0_full)
        # Crocoddyl's floating-base actuation finds the unactuated joint by this name.
        self.model.names[1] = "root_joint"
        self.data = self.model.createData()
        self.nq, self.nv = self.model.nq, self.model.nv
        self.nu = self.nv - 6

        # Joint j of the reduced model is MuJoCo dof ``mj_dof[j]``.
        names = [self.model.names[i] for i in range(2, self.model.njoints)]
        self.mj_dof = np.array(
            [
                self.mj_model.jnt_dofadr[
                    mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in names
            ],
            dtype=int,
        )
        self.mj_qadr = self.mj_dof + 1
        #: Actuator index for each reduced joint, and for every locked joint.
        act_dof = self.mj_model.jnt_dofadr[self.mj_model.actuator_trnid[:, 0]]
        self.act_of_joint = np.array([int(np.nonzero(act_dof == d)[0][0]) for d in self.mj_dof])
        self.locked_act = np.array(
            [a for a in range(self.mj_model.nu) if a not in set(self.act_of_joint.tolist())],
            dtype=int,
        )
        self.locked_dof = act_dof[self.locked_act]
        self.locked_q0 = self.mj_qpos0[self.locked_dof + 1].copy()

        self.foot_ids = [self.model.getFrameId(name) for name in FOOT_FRAMES]
        self.q0 = self.config_from_mujoco(self.mj_qpos0)
        self.x0 = np.concatenate([self.q0, np.zeros(self.nv)])
        self.torque_limit = np.asarray(self.model.effortLimit[6:], dtype=float).copy()
        self.mass = float(pin.computeTotalMass(self.model))

    # -- coordinate maps --------------------------------------------------

    def config_from_mujoco(self, qpos: np.ndarray) -> np.ndarray:
        q = np.empty(self.nq)
        q[:3] = qpos[:3]
        q[3:7] = np.asarray(qpos)[[4, 5, 6, 3]]
        q[7:] = np.asarray(qpos)[self.mj_qadr]
        return q

    def state_from_mujoco(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        q = self.config_from_mujoco(qpos)
        rot = pin.Quaternion(q[6], q[3], q[4], q[5]).toRotationMatrix()
        v = np.empty(self.nv)
        v[:3] = rot.T @ np.asarray(qvel)[:3]
        v[3:6] = np.asarray(qvel)[3:6]
        v[6:] = np.asarray(qvel)[self.mj_dof]
        return np.concatenate([q, v])

    # -- kinematics -------------------------------------------------------

    def foot_placements(self, q: np.ndarray) -> list[pin.SE3]:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return [self.data.oMf[fid].copy() for fid in self.foot_ids]

    def com(self, q: np.ndarray) -> np.ndarray:
        return pin.centerOfMass(self.model, self.data, q).copy()
