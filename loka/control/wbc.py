"""Whole-body QP that turns MPC contact forces into joint torques.

One quadratic program per control tick over ``z = [q̈, f]``:

    minimise   w_c‖J_c q̈ + J̇_c q̇ − a_c‖² + Σ wᵢ‖zᵢ − z*ᵢ‖²
    subject to  (M q̈ + h − J_cᵀ f)[base rows] = 0     unactuated base
                |f_xy| ≤ μ f_z,  f_z ∈ [f_min, f_max]  friction
                |τ(z)| ≤ τ_max                          actuator limits

Only physics is a hard constraint. Keeping the feet planted is a *cost*, and
that choice is what makes the controller survive contact changes: as a heel
lands, a hard "hold this point still" row fights the velocity the foot already
has, and the program can go infeasible in the one millisecond when the robot
most needs an answer. Penalising the same residual instead keeps the feasible
set non-empty at every instant -- ``f = 0`` with a free-falling ``q̈`` always
satisfies the remaining rows -- so the solver degrades smoothly rather than
handing back a garbage iterate.

The remaining tasks -- contact-force tracking from the MPC, base orientation,
base position, joint posture -- are direct preferences on a slice of ``z``.

The task set follows the usual weighted-WBC recipe (Kim et al., "Highly
Dynamic Quadruped Locomotion via WBIC", and the MPC+WBC humanoid stacks built
on it).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from loka.control.qp import QP, Sparsity
from loka.control.robot import NUM_LEG_JOINTS, QVEL_JOINT0, DynamicsTerms


@dataclass
class WBCConfig:
    friction_mu: float = 0.5
    fz_min: float = 1.0
    fz_max: float = 500.0
    #: Velocity feedback on the contact points. Pure damping: MuJoCo already
    #: resolves penetration, this only bleeds off residual sliding.
    contact_kd: float = 20.0

    #: Task weights. Accelerations are in m/s² or rad/s², forces in N, so the
    #: force weight is small by construction.
    #: Keeping the feet planted outranks every other task by a wide margin --
    #: this is a constraint in all but name.
    weight_contact: float = 2000.0
    weight_base_position: float = 10.0
    weight_base_orientation: float = 40.0
    #: With both feet planted the twelve leg joints are already pinned by the
    #: contact task and the base motion, so a leg posture task can only fight
    #: them -- it is kept just heavy enough to condition the null space.
    #: The waist and arms carry no contacts and genuinely need holding.
    weight_posture_legs: float = 0.02
    weight_posture_upper: float = 5.0
    weight_force: float = 5e-3
    #: Ridge term keeping the Hessian strictly positive definite.
    regularization: float = 1e-6

    #: Task-space PD gains feeding the desired accelerations.
    #: A planted biped can only accelerate its CoM by shifting the centre of
    #: pressure inside the soles, which caps the useful gain at roughly
    #: ``g · (half foot length) / (CoM height)`` ÷ (error worth correcting).
    #: Asking for more than that saturates the QP and turns the PD law into a
    #: bang-bang controller that overshoots straight out of the support
    #: polygon -- the single biggest cause of falls before this was tuned down.
    kp_base_position: float = 50.0
    kd_base_position: float = 15.0
    kp_base_orientation: float = 300.0
    kd_base_orientation: float = 30.0
    #: Legs get damping only *while planted*. A stiffness term there would pull
    #: the knees back towards the keyframe and fight every commanded crouch or
    #: weight shift; what a planted leg needs is somewhere for excess velocity
    #: to go, since the contact task already pins its configuration.
    kp_posture_legs: float = 0.0
    kd_posture_legs: float = 2.0
    #: An airborne leg has no contact holding it, so it reverts to a real
    #: position task that carries it back to the landing pose. While walking the
    #: Cartesian foot task takes over and this is demoted to a null-space bias.
    weight_posture_swing: float = 5.0
    kp_posture_swing: float = 200.0
    kd_posture_swing: float = 20.0
    #: Cartesian swing-foot tracking on the foot-centre Jacobian. This is what
    #: executes the planned foothold. Stance contact is weighted ~2000, so this
    #: has to be hundreds — at 60 the QP inverted the lift and the foot only
    #: cleared 8 mm. At zero the footstep planner has no actuator at all.
    weight_swing_foot: float = 500.0
    #: Error feedback only -- the arc supplies its own acceleration -- so these
    #: are moderate, with ``kd ≈ 2√kp`` for a settled, non-ringing landing.
    #: 500/40 (was 300/30) closes the last centimetre once the QP actually
    #: tracks the task; 300 left a 10 mm hover at s = 1 that held the clock.
    kp_swing_foot: float = 500.0
    kd_swing_foot: float = 40.0
    #: Late-swing sole orientation: world-flat roll/pitch so the four sites
    #: arrive together. Below the Cartesian weight so a height residual still
    #: wins if the two disagree; ``kd ≈ 2√kp`` again.
    weight_swing_orient: float = 40.0
    kp_swing_orient: float = 200.0
    kd_swing_orient: float = 28.0
    kp_posture_upper: float = 100.0
    kd_posture_upper: float = 10.0


@dataclass
class WBCTargets:
    base_linear_acc: np.ndarray  # (3,) world
    base_angular_acc: np.ndarray  # (3,) body frame
    joint_acc: np.ndarray  # (nj,)
    contact_forces: np.ndarray  # (nc, 3) world, from the MPC
    #: Per-joint posture weight, overriding the configured defaults. Lets the
    #: caller hand an airborne leg back to a position task.
    joint_weights: np.ndarray | None = None  # (nj,)
    #: Scales both base tasks. Should fall towards zero as the robot runs out
    #: of contacts to push against, since a base acceleration it cannot produce
    #: is an invitation for the QP to thrash the limbs trying.
    base_weight_scale: float = 1.0
    #: Optional Cartesian swing-foot task: ``J_f q̈ ≈ a*`` with
    #: ``a* = kp(p*-p) - kd v`` (world frame).
    swing_jacobian: np.ndarray | None = None  # (3, nv)
    swing_acc: np.ndarray | None = None  # (3,)
    #: Optional sole-orientation task, same residual form on ``J_ω``.
    swing_orient_jacobian: np.ndarray | None = None  # (3, nv)
    swing_orient_acc: np.ndarray | None = None  # (3,)


@dataclass
class WBCSolution:
    torque: np.ndarray  # (nj,)
    qacc: np.ndarray  # (nv,)
    forces: np.ndarray  # (nc, 3)


class WholeBodyController:
    def __init__(self, config: WBCConfig, nv: int, num_joints: int, num_contacts: int,
                 torque_limit: np.ndarray) -> None:
        self.config = config
        self.nv = nv
        self.num_joints = num_joints
        self.num_contacts = num_contacts
        self.force_dim = 3 * num_contacts
        self.num_vars = nv + self.force_dim
        self.torque_limit = np.asarray(torque_limit, dtype=float)

        self.n_base_rows = QVEL_JOINT0
        self.n_friction_rows = 5 * num_contacts
        self.n_torque_rows = num_joints
        self.num_constraints = (
            self.n_base_rows + self.n_friction_rows + self.n_torque_rows
        )

        self._constraint = np.zeros((self.num_constraints, self.num_vars))
        self._lower = np.zeros(self.num_constraints)
        self._upper = np.zeros(self.num_constraints)
        self._hessian = np.zeros((self.num_vars, self.num_vars))
        self._diagonal = self._default_weights()
        self._qp = QP(
            "wbc",
            hessian_pattern=self._hessian_pattern(),
            constraint_pattern=self._constraint_pattern(),
        )
        self._write_friction_rows()

    # -- static problem structure ----------------------------------------

    @property
    def _joint_slice(self) -> slice:
        return slice(QVEL_JOINT0, QVEL_JOINT0 + self.num_joints)

    @property
    def _friction_slice(self) -> slice:
        return slice(self.n_base_rows, self.n_base_rows + self.n_friction_rows)

    @property
    def _torque_slice(self) -> slice:
        start = self.n_base_rows + self.n_friction_rows
        return slice(start, start + self.n_torque_rows)

    def _default_weights(self) -> np.ndarray:
        cfg = self.config
        weights = np.empty(self.num_vars)
        weights[0:3] = cfg.weight_base_position
        weights[3:6] = cfg.weight_base_orientation
        posture = np.full(self.num_joints, cfg.weight_posture_legs)
        posture[NUM_LEG_JOINTS:] = cfg.weight_posture_upper  # waist + arms
        weights[self._joint_slice] = posture
        weights[self.nv :] = cfg.weight_force
        return weights

    def _hessian_pattern(self) -> Sparsity:
        """``JᵀJ`` fills the acceleration block; forces stay diagonal."""
        mask = np.zeros((self.num_vars, self.num_vars), dtype=bool)
        mask[: self.nv, : self.nv] = True
        idx = np.arange(self.nv, self.num_vars)
        mask[idx, idx] = True
        return Sparsity(np.triu(mask))

    def _constraint_pattern(self) -> Sparsity:
        """Which entries of ``A`` are ever non-zero.

        Friction rows touch only their own three force columns; spelling that
        out drops a large block of structural zeros OSQP would otherwise
        factorise.
        """
        mask = np.zeros((self.num_constraints, self.num_vars), dtype=bool)
        mask[: self.n_base_rows, :] = True  # base rows of M and J_cᵀ
        for i in range(self.num_contacts):
            r = self._friction_slice.start + 5 * i
            c = self.nv + 3 * i
            mask[r : r + 5, c : c + 3] = True
        mask[self._torque_slice, :] = True
        return Sparsity(mask)

    def _write_friction_rows(self) -> None:
        """Friction pyramid and normal-force bounds; constant across solves."""
        cfg = self.config
        mu = cfg.friction_mu
        block = np.array(
            [
                [1.0, 0.0, -mu],
                [-1.0, 0.0, -mu],
                [0.0, 1.0, -mu],
                [0.0, -1.0, -mu],
                [0.0, 0.0, 1.0],
            ]
        )
        rows = self._friction_slice
        for i in range(self.num_contacts):
            r = rows.start + 5 * i
            c = self.nv + 3 * i
            self._constraint[r : r + 5, c : c + 3] = block
            self._lower[r : r + 4] = -np.inf
            self._upper[r : r + 4] = 0.0
            self._lower[r + 4] = cfg.fz_min
            self._upper[r + 4] = cfg.fz_max

    def refresh(self) -> None:
        """Re-read config values cached outside :meth:`solve`.

        The base and posture weights are supplied per tick by the caller, and
        the task gains live in the config the caller reads, so only the force
        weight and the friction block are stale after a live edit.
        """
        self._diagonal = self._default_weights()
        self._write_friction_rows()

    # -- solve ------------------------------------------------------------

    def solve(
        self,
        dynamics: DynamicsTerms,
        targets: WBCTargets,
        contact_mask: np.ndarray | None = None,
        contact_hold_mask: np.ndarray | None = None,
    ) -> WBCSolution:
        cfg = self.config
        nv = self.nv
        mass_matrix = dynamics.mass_matrix
        bias = dynamics.bias
        jac = dynamics.contact_jacobian
        jac_t = jac.T

        # Base rows of the equation of motion: the floating base has no
        # actuator, so contact forces alone must balance its inertia.
        rows = slice(0, self.n_base_rows)
        self._constraint[rows, :nv] = mass_matrix[: self.n_base_rows, :]
        self._constraint[rows, nv:] = -jac_t[: self.n_base_rows, :]
        self._lower[rows] = -bias[: self.n_base_rows]
        self._upper[rows] = -bias[: self.n_base_rows]

        # Actuator limits, written on τ = (M q̈ + h - J_cᵀ f)[actuated].
        rows = self._torque_slice
        self._constraint[rows, :nv] = mass_matrix[QVEL_JOINT0:, :]
        self._constraint[rows, nv:] = -jac_t[QVEL_JOINT0:, :]
        joint_bias = bias[QVEL_JOINT0:]
        self._lower[rows] = -self.torque_limit - joint_bias
        self._upper[rows] = self.torque_limit - joint_bias

        # Lifted contacts (if any) carry no force and are dropped from the
        # contact task; everything else about the problem is unchanged.
        # ``contact_hold_mask`` may be a subset: a peeling swing foot can
        # still take force without the kinematic hold pinning it to the ground.
        if contact_mask is None:
            planted = np.ones(self.num_contacts, dtype=bool)
        else:
            planted = np.asarray(contact_mask, dtype=bool)
        if contact_hold_mask is None:
            held = planted
        else:
            held = np.asarray(contact_hold_mask, dtype=bool) & planted
        fric = self._friction_slice
        for i, in_contact in enumerate(planted):
            r = fric.start + 5 * i
            self._lower[r + 4] = cfg.fz_min if in_contact else 0.0
            self._upper[r + 4] = cfg.fz_max if in_contact else 0.0

        # Contact task: J q̈ ≈ a_c - J̇q̇, weighted heavily and folded into the
        # cost. Rows for airborne (or peeling) points are zeroed so they
        # contribute nothing.
        active = np.repeat(held, 3)
        jac_active = jac * active[:, None]
        contact_acc = -cfg.contact_kd * dynamics.contact_vel.reshape(-1)
        contact_rhs = (contact_acc - dynamics.contact_bias_acc) * active

        weights = self._diagonal
        if targets.joint_weights is not None:
            weights[self._joint_slice] = targets.joint_weights
        weights[: self.n_base_rows] = targets.base_weight_scale * np.concatenate(
            (np.full(3, cfg.weight_base_position), np.full(3, cfg.weight_base_orientation))
        )

        target = np.empty(self.num_vars)
        target[0:3] = targets.base_linear_acc
        target[3:6] = targets.base_angular_acc
        target[self._joint_slice] = targets.joint_acc
        target[nv:] = targets.contact_forces.reshape(-1)

        hessian = self._hessian
        hessian[:nv, :nv] = cfg.weight_contact * (jac_active.T @ jac_active)
        # The acceleration diagonal adds onto the freshly written JᵀJ; the
        # force block is diagonal only, so it is assigned rather than summed.
        acc = np.arange(nv)
        force = np.arange(nv, self.num_vars)
        hessian[acc, acc] += weights[:nv] + cfg.regularization
        hessian[force, force] = weights[nv:] + cfg.regularization
        gradient = -weights * target
        gradient[:nv] -= cfg.weight_contact * (jac_active.T @ contact_rhs)

        # Cartesian / orientation swing-foot tasks (Phase B). Hessian already
        # dense on q̈. Each is an optional 3-row residual ``J q̈ ≈ a*``.
        def add_swing_task(jacobian, acc, weight: float) -> None:
            if jacobian is None or acc is None or weight <= 0.0:
                return
            j_s = np.asarray(jacobian, dtype=float).reshape(3, nv)
            a_s = np.asarray(acc, dtype=float).reshape(3)
            hessian[:nv, :nv] += weight * (j_s.T @ j_s)
            gradient[:nv] -= weight * (j_s.T @ a_s)

        add_swing_task(targets.swing_jacobian, targets.swing_acc, cfg.weight_swing_foot)
        add_swing_task(
            targets.swing_orient_jacobian,
            targets.swing_orient_acc,
            cfg.weight_swing_orient,
        )

        solution = self._qp.solve(
            hessian, gradient, self._constraint, self._lower, self._upper
        )
        qacc = solution[:nv]
        forces = solution[nv:]
        torque = (mass_matrix @ qacc + bias - jac_t @ forces)[QVEL_JOINT0:]
        torque = np.clip(torque, -self.torque_limit, self.torque_limit)
        return WBCSolution(
            torque=torque, qacc=qacc, forces=forces.reshape(self.num_contacts, 3)
        )
