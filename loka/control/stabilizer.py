"""DCM stabilizer and contact-wrench distribution (classical terrain stack).

The humanoid stair controllers in the literature (Englsberger et al., TRO
2015; IHMC's Atlas, Griffin et al. 2018; Caron et al., ICRA 2019; AIST's
BaselineWalkingController) close the balance loop on the divergent component
of motion rather than through a centroidal MPC:

1. DCM feedback. With ``ξ = c + ċ/ω`` and the plan's ``ξ_ref``, ``ξ̇_ref``,
   the desired centroidal moment pivot is

       r_cmp = ξ − ξ̇_ref/ω + (k_ξ/ω)(ξ − ξ_ref),

   which makes the DCM error decay at ``k_ξ``. The horizontal force follows
   from the pendulum, ``f_xy = m ω² (c_xy − r_cmp)``; the vertical force
   tracks the CoM height plan.
2. Angular momentum. The desired moment about the CoM keeps the trunk
   upright and damps the centroidal angular momentum, ``M = I k_R e_R − k_L L``.
3. Wrench distribution. One small QP spreads that net wrench over the planted
   sole corners inside their friction pyramids. Keeping the moment about the
   CoM at the desired value is what puts the centre of pressure at the CMP;
   if the CMP leaves the support polygon, the QP returns the closest
   feasible wrench.

The whole-body QP then tracks these forces as it tracks the MPC's.
"""

from __future__ import annotations

import numpy as np

from loka.control.qp import QP, Sparsity

GRAVITY = 9.81
#: DCM error decay rate [1/s]. ``1 + k_ξ/ω`` is about 1.8 at walking height.
DCM_GAIN = 3.0
#: CoM height tracking [1/s², 1/s].
HEIGHT_KP = 50.0
HEIGHT_KD = 14.0
#: Trunk orientation stiffness [1/s²] and angular-momentum damping [1/s].
ORIENTATION_KP = 30.0
MOMENTUM_DAMPING = 8.0
#: Wrench-tracking weights and force regularization of the distribution QP.
FORCE_WEIGHT = 1.0
MOMENT_WEIGHT = 4.0
FORCE_REGULARIZATION = 1e-4
#: The vertical force never drops below this fraction of weight while walking.
MIN_VERTICAL = 0.2


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


class DCMStabilizer:
    """DCM feedback to a CMP, then a contact-wrench distribution QP."""

    def __init__(self, num_contacts: int, mass: float, *, mu: float, fz_min: float,
                 fz_max: float) -> None:
        self.num_contacts = int(num_contacts)
        self.mass = float(mass)
        self.mu = float(mu)
        self.fz_min = float(fz_min)
        self.fz_max = float(fz_max)
        n = 3 * self.num_contacts
        self._n = n
        rows = 5 * self.num_contacts
        self._constraint = np.zeros((rows, n))
        self._lower = np.full(rows, -np.inf)
        self._upper = np.zeros(rows)
        block = np.array([
            [1.0, 0.0, -self.mu],
            [-1.0, 0.0, -self.mu],
            [0.0, 1.0, -self.mu],
            [0.0, -1.0, -self.mu],
            [0.0, 0.0, 1.0],
        ])
        mask = np.zeros((rows, n), dtype=bool)
        for i in range(self.num_contacts):
            self._constraint[5 * i:5 * i + 5, 3 * i:3 * i + 3] = block
            mask[5 * i:5 * i + 5, 3 * i:3 * i + 3] = True
        self._qp = QP(
            "wrench_distribution",
            hessian_pattern=Sparsity.upper_triangular(n),
            constraint_pattern=Sparsity(mask),
        )
        self.last_cmp = np.zeros(2)
        self.last_wrench = np.zeros(6)

    def desired_wrench(self, *, com, com_vel, com_z_ref, com_vz_ref, dcm_ref, dcm_rate_ref,
                       omega, orientation_error, angular_momentum, inertia):
        """Net force and moment about the CoM the contacts should apply."""
        com = np.asarray(com, dtype=float)
        vel = np.asarray(com_vel, dtype=float)
        omega = max(float(omega), 1e-3)
        dcm = com[:2] + vel[:2] / omega
        err = dcm - np.asarray(dcm_ref, dtype=float)
        cmp = dcm - np.asarray(dcm_rate_ref, dtype=float) / omega + (DCM_GAIN / omega) * err
        force = np.zeros(3)
        force[:2] = self.mass * omega * omega * (com[:2] - cmp)
        acc_z = HEIGHT_KP * (float(com_z_ref) - com[2]) + HEIGHT_KD * (float(com_vz_ref) - vel[2])
        force[2] = max(self.mass * (GRAVITY + acc_z), MIN_VERTICAL * self.mass * GRAVITY)
        moment = (
            np.asarray(inertia, dtype=float) @ (ORIENTATION_KP * np.asarray(orientation_error, dtype=float))
            - MOMENTUM_DAMPING * np.asarray(angular_momentum, dtype=float)
        )
        self.last_cmp = cmp
        self.last_wrench = np.concatenate([force, moment])
        return force, moment

    def distribute(self, force, moment, com, contact_pos, planted) -> np.ndarray:
        """Contact forces (nc, 3) whose sum and moment about ``com`` match the wrench."""
        pos = np.asarray(contact_pos, dtype=float).reshape(-1, 3)
        planted = np.asarray(planted, dtype=bool).reshape(-1)
        a_force = np.tile(np.eye(3), (1, self.num_contacts))
        a_moment = np.hstack([_skew(p - np.asarray(com, dtype=float)) for p in pos])
        hessian = (
            FORCE_WEIGHT * a_force.T @ a_force
            + MOMENT_WEIGHT * a_moment.T @ a_moment
            + FORCE_REGULARIZATION * np.eye(self._n)
        )
        gradient = -(FORCE_WEIGHT * a_force.T @ np.asarray(force, dtype=float)
                     + MOMENT_WEIGHT * a_moment.T @ np.asarray(moment, dtype=float))
        for i, down in enumerate(planted):
            self._lower[5 * i + 4] = self.fz_min if down else 0.0
            self._upper[5 * i + 4] = self.fz_max if down else 0.0
        if not planted.any():
            return np.zeros((self.num_contacts, 3))
        solution = self._qp.solve(hessian, gradient, self._constraint, self._lower, self._upper)
        forces = solution.reshape(self.num_contacts, 3)
        forces[~planted] = 0.0
        return forces
