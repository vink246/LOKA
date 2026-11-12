"""Foot load from joint torques: contact sensing without foot force sensors.

A factory G1 has joint encoders, an IMU and motor currents, but no force
sensing in the feet. Each leg's joint rows of the equation of motion,

    M_leg q̈ + h_leg = τ_leg + τ_passive,leg + J_leg(foot)ᵀ f_foot,

involve only that foot's contact, so the force the ground puts on a foot is
recovered from the leg's applied torques (motor current) and the belief
model's inverse dynamics at the measured state. Six leg joints resolve the
full six-axis wrench, so the load split between two planted feet is
measured, not assumed. ``contact_force`` is that wrench mapped back to
generalized coordinates; the anomaly detector reads it in place of a foot
force sensor. On hardware ``q̈`` comes from filtered differentiation of the
encoder velocities and the joint torque from motor current (``tau_est``).
"""

from __future__ import annotations

import mujoco
import numpy as np

from loka.control.robot import NUM_LEG_JOINTS, QVEL_JOINT0, G1Model

#: Vertical load on a swinging foot that counts as touching the ground [N].
#: About a tenth of body weight, well above the residual of a free leg.
SWING_CONTACT_N = 35.0
#: A leg joint this close to its range end is resting on its stop [rad]. The
#: stop's reaction torque is in neither the motor current nor the model, so
#: that joint's row is left out of the foot-wrench fit.
JOINT_STOP_BAND = 0.005


def rigid_body_inverse(model, data) -> np.ndarray:
    """``M(q) q̈ + h(q, q̇)`` at ``data``'s state, rotor armature included.

    ``mj_rne`` leaves out ``dof_armature · q̈``, which on the G1's legs is up
    to a few Nm during a swing. Needs kinematics, CRB and velocities computed.
    """
    inverse = np.empty(model.nv)
    mujoco.mj_rne(model, data, 1, inverse)
    return inverse + model.dof_armature * data.qacc


class FootLoadEstimator:
    """World-frame wrench on each foot centre, from proprioception only."""

    def __init__(self, robot: G1Model) -> None:
        self.robot = robot
        self.data = mujoco.MjData(robot.model)
        per = NUM_LEG_JOINTS // 2
        self.leg_dofs = [
            np.arange(QVEL_JOINT0 + leg * per, QVEL_JOINT0 + (leg + 1) * per) for leg in (0, 1)
        ]
        nv = robot.model.nv
        self.forces = np.zeros((2, 3))
        #: Force then moment about each foot centre, world frame.
        self.wrenches = np.zeros((2, 6))
        #: Generalized force of both foot wrenches, ``(nv,)``.
        self.contact_force = np.zeros(nv)
        #: Belief passive joint force (damping) at the measured state, ``(nv,)``.
        self.passive = np.zeros(nv)
        self._jacp = np.zeros((3, nv))
        self._jacr = np.zeros((3, nv))
        model = robot.model
        self._leg_qpos = [
            np.array([int(model.jnt_qposadr[model.dof_jntid[d]]) for d in dofs]) for dofs in self.leg_dofs
        ]
        self._leg_range = [
            np.array([model.jnt_range[model.dof_jntid[d]] for d in dofs]) for dofs in self.leg_dofs
        ]
        self._leg_limited = [
            np.array([bool(model.jnt_limited[model.dof_jntid[d]]) for d in dofs]) for dofs in self.leg_dofs
        ]

    def update(self, qpos, qvel, qacc, joint_torque) -> np.ndarray:
        """``joint_torque`` is the applied actuator torque per DOF, ``(nv,)``."""
        model, data = self.robot.model, self.data
        nv = model.nv
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.qacc[:] = np.asarray(qacc, dtype=float)[:nv]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_crb(model, data)
        mujoco.mj_comVel(model, data)
        mujoco.mj_passive(model, data)
        inverse = rigid_body_inverse(model, data)
        self.passive = data.qfrc_passive.copy()
        residual = inverse - np.asarray(joint_torque, dtype=float)[:nv] - self.passive
        qpos_arr = np.asarray(qpos, dtype=float)
        contact = np.zeros(nv)
        for leg in (0, 1):
            site = int(self.robot.foot_center_site_ids[leg])
            self._jacp[:] = 0.0
            self._jacr[:] = 0.0
            mujoco.mj_jacSite(model, data, self._jacp, self._jacr, site)
            dofs = self.leg_dofs[leg]
            # Heel and toe contacts put a moment on the sole as well as a
            # force, so the six leg joints are resolved into a full wrench.
            jt = np.hstack([self._jacp[:, dofs].T, self._jacr[:, dofs].T])  # (6, 6)
            q = qpos_arr[self._leg_qpos[leg]]
            lo, hi = self._leg_range[leg][:, 0], self._leg_range[leg][:, 1]
            free = ~(self._leg_limited[leg] & ((q - lo < JOINT_STOP_BAND) | (hi - q < JOINT_STOP_BAND)))
            if free.all():
                wrench, *_ = np.linalg.lstsq(jt, residual[dofs], rcond=1e-6)
            else:
                # Minimum-norm wrench from the joints that are not on a stop.
                wrench, *_ = np.linalg.lstsq(jt[free], residual[dofs][free], rcond=1e-6)
            self.wrenches[leg] = wrench
            self.forces[leg] = wrench[:3]
            contact += self._jacp.T @ wrench[:3] + self._jacr.T @ wrench[3:]
        self.contact_force = contact
        return self.forces.copy()
