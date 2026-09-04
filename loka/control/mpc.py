"""Convex single-rigid-body MPC over finite-foot ground-reaction forces.

This is the Di Carlo / MIT Cheetah formulation (IROS 2018): state
``x = [θ, p, ω, v, g] ∈ R¹³``, one condensed convex QP, friction pyramids,
horizon 6 × 50 ms. The humanoid extras are *not* a nonlinear program:

* Per-foot CoP-in-sole inequalities (Sleiman et al., TRO 2021 — the contact
  *surface*, not a 12 mm-shrunk patch; the gait already insets the ZMP
  *reference*).
* A scheduled contact mask and Bézier-posed airborne soles so the force plan
  does not brace on a foot the WBC has not planted.
* Optional RTI linearisation of the CoM along the last primal (still a QP).

Galliker et al., Humanoids 2022 is whole-body *nonlinear* MPC (ocs2, torques
over the horizon). That is not this class. ``CentroidalNMPC`` remains a
compatibility alias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from loka.control.qp import QP, Sparsity
from loka.control.robot import GRAVITY

STATE_DIM = 13
#: Four sole sites per foot, matching ``CONTACT_SITES`` in ``robot.py``.
SITES_PER_FOOT = 4
#: G1 sole half-length / half-width from the contact sites about the foot
#: centre (heel −85 mm, toe +85 mm, y ±25–30 mm). See ``g1_29dof.xml``.
SOLE_HALF_LENGTH = 0.085
SOLE_HALF_WIDTH = 0.030


@dataclass
class MPCConfig:
    #: 6 x 50 ms = 0.3 s lookahead. Galliker's HZD-as-terminal-cost lesson:
    #: a good gait reference is what makes a short horizon work, not a 2 s NLP.
    horizon: int = 6
    dt: float = 0.05
    friction_mu: float = 0.5
    #: Per-contact normal force limits [N]. A positive minimum keeps every
    #: contact loaded, which is what makes a two-foot stand stiff.
    fz_min: float = 5.0
    fz_max: float = 400.0
    #: Diagonal state cost over [roll, pitch, yaw, x, y, z, ωx, ωy, ωz, vx, vy, vz].
    weight_orientation: Sequence[float] = (500.0, 500.0, 300.0)
    weight_position: Sequence[float] = (120.0, 120.0, 1000.0)
    weight_angular_velocity: Sequence[float] = (10.0, 10.0, 10.0)
    weight_linear_velocity: Sequence[float] = (60.0, 60.0, 60.0)
    #: Force-effort weight. Must stay tiny: the forces are O(100 N) so their
    #: squares swamp the tracking terms unless this is several orders of
    #: magnitude below the state weights. It exists only to pick a sensible
    #: (evenly shared) point out of the redundant eight-contact null space.
    weight_force: float = 1e-6
    #: Numeric shrink of the CoP box from the physical sole edge [m].
    #: Sleiman et al. (TRO 2021) keep CoP inside the *contact surface*, not a
    #: shrunken patch: the gait already insets the ZMP *reference* by
    #: ``ZMP_INSET``, which is what leaves spare travel. A 12 mm hard box
    #: here stole that spare and the lateral orbit died by step 4.
    cop_margin: float = 0.002

    def state_weights(self) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(self.weight_orientation, dtype=float),
                np.asarray(self.weight_position, dtype=float),
                np.asarray(self.weight_angular_velocity, dtype=float),
                np.asarray(self.weight_linear_velocity, dtype=float),
                [0.0],
            ]
        )


@dataclass
class CentroidalState:
    rpy: np.ndarray
    com: np.ndarray
    angular_velocity: np.ndarray
    com_velocity: np.ndarray
    inertia: np.ndarray  # about the CoM, world frame
    contact_pos: np.ndarray  # (nc, 3) world


@dataclass
class CentroidalReference:
    com: np.ndarray
    yaw: float = 0.0
    com_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))


def _skew(vec: np.ndarray) -> np.ndarray:
    x, y, z = vec
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


class ConvexMPC:
    """Di Carlo convex SRBD force QP; returns the first contact-force block.

    Same ``solve(state, reference, schedule, contact_pos_seq) -> (nc, 3)``
    contract the WBC already consumes. Standing omits the schedule and holds
    the measured two-foot set.
    """

    def __init__(self, config: MPCConfig, num_contacts: int, mass: float) -> None:
        if num_contacts % SITES_PER_FOOT != 0:
            raise ValueError(
                f"ConvexMPC needs {SITES_PER_FOOT} sites/foot, got {num_contacts}"
            )
        self.config = config
        self.num_contacts = num_contacts
        self.num_feet = num_contacts // SITES_PER_FOOT
        self.mass = mass
        self.force_dim = 3 * num_contacts
        self.num_vars = self.force_dim * config.horizon
        self._n_friction = 5 * num_contacts * config.horizon
        self._n_cop = 4 * self.num_feet * config.horizon
        self._constraint, self._lower, self._upper = self._build_constraints()
        self._qp = QP(
            "mpc",
            hessian_pattern=Sparsity.upper_triangular(self.num_vars),
            constraint_pattern=Sparsity(self._constraint != 0.0),
        )
        self.last_forces = self._gravity_share()
        self._last_horizon = np.tile(self.last_forces, (config.horizon, 1, 1))
        self.cost = 0.0

    # -- setup ------------------------------------------------------------

    def _gravity_share(self) -> np.ndarray:
        """Even split of body weight across all contacts, used before the
        first solve and as the fallback if a solve fails."""
        forces = np.zeros((self.num_contacts, 3))
        forces[:, 2] = self.mass * GRAVITY / self.num_contacts
        return forces

    def _build_constraints(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Friction pyramids, normal-force bounds, and per-foot CoP boxes.

        Friction numbers depend only on μ. CoP coefficients depend on the
        current sole geometry and are overwritten every solve; the *pattern*
        (four ``fz`` columns per foot, per horizon step) is pinned here so
        OSQP can refactor once.
        """
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
        lower_block = np.array([-np.inf, -np.inf, -np.inf, -np.inf, cfg.fz_min])
        upper_block = np.array([0.0, 0.0, 0.0, 0.0, cfg.fz_max])

        rows_per_contact = block.shape[0]
        num_blocks = self.num_contacts * cfg.horizon
        n_rows = self._n_friction + self._n_cop
        constraint = np.zeros((n_rows, self.num_vars))
        for b in range(num_blocks):
            r = b * rows_per_contact
            c = b * 3
            constraint[r : r + rows_per_contact, c : c + 3] = block
        lower = np.concatenate(
            [
                np.tile(lower_block, num_blocks),
                np.full(self._n_cop, -np.inf),
            ]
        )
        upper = np.concatenate(
            [
                np.tile(upper_block, num_blocks),
                np.zeros(self._n_cop),
            ]
        )
        # Structural nonzeros for the CoP box: |Σ fz o| ≤ half Σ fz, four
        # inequalities × feet × horizon, each touching that foot's four fz.
        for k in range(cfg.horizon):
            for foot in range(self.num_feet):
                for side in range(4):
                    row = self._n_friction + 4 * (k * self.num_feet + foot) + side
                    for j in range(SITES_PER_FOOT):
                        col = k * self.force_dim + 3 * (foot * SITES_PER_FOOT + j) + 2
                        constraint[row, col] = 1.0
        return constraint, lower, upper

    def refresh(self) -> None:
        """Re-read config values that are baked into the constraint block.

        The cost is rebuilt from ``self.config`` on every solve, so weights
        need no refresh at all; only ``friction_mu`` and the force limits land
        here. CoP geometry is filled in ``solve``.
        """
        self._constraint, self._lower, self._upper = self._build_constraints()

    # -- dynamics ---------------------------------------------------------

    def _state_matrix(self, yaw: float) -> np.ndarray:
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        # Rz(ψ)ᵀ maps world angular velocity to roll-pitch-yaw rates under the
        # usual small roll/pitch assumption.
        rz_t = np.array([[cos_y, sin_y, 0.0], [-sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]])

        a_c = np.zeros((STATE_DIM, STATE_DIM))
        a_c[0:3, 6:9] = rz_t
        a_c[3:6, 9:12] = np.eye(3)
        a_c[11, 12] = 1.0  # x[12] carries -g
        return np.eye(STATE_DIM) + a_c * self.config.dt

    def _input_matrix(
        self, contact_pos: np.ndarray, com: np.ndarray, inertia_inv: np.ndarray
    ) -> np.ndarray:
        """Force-to-state map for one horizon step.

        Depends on the contact geometry, which moves as the robot steps, so
        this is rebuilt per horizon step while the state matrix is not.
        """
        b_c = np.zeros((STATE_DIM, self.force_dim))
        offsets = np.asarray(contact_pos, dtype=float) - np.asarray(com, dtype=float)
        inv_mass = 1.0 / self.mass
        for i in range(self.num_contacts):
            b_c[6:9, 3 * i : 3 * i + 3] = inertia_inv @ _skew(offsets[i])
            b_c[9:12, 3 * i : 3 * i + 3] = inv_mass * np.eye(3)
        return b_c * self.config.dt

    def _condense(
        self, a_d: np.ndarray, b_seq: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Roll the horizon into ``X = A_qp x₀ + B_qp U``.

        ``A`` is held constant over the horizon (it depends only on yaw), so its
        powers are accumulated once; ``B`` varies per step with the planned
        contact positions and the RTI CoM rollout.
        """
        n = self.config.horizon
        a_qp = np.zeros((STATE_DIM * n, STATE_DIM))
        b_qp = np.zeros((STATE_DIM * n, self.num_vars))

        powers = [np.eye(STATE_DIM)]
        for _ in range(n):
            powers.append(powers[-1] @ a_d)
        for k in range(n):
            a_qp[STATE_DIM * k : STATE_DIM * (k + 1)] = powers[k + 1]
            for j in range(k + 1):
                b_qp[
                    STATE_DIM * k : STATE_DIM * (k + 1),
                    self.force_dim * j : self.force_dim * (j + 1),
                ] = powers[k - j] @ b_seq[j]
        return a_qp, b_qp

    def _rollout_com(self, state: CentroidalState) -> list[np.ndarray]:
        """Predicted CoM along last horizon's forces (RTI linearisation point).

        The previous convex SRBD used ``com + v_ref * k dt``, i.e. the
        *reference* rather than the dynamics. Sequential QP linearises the
        centroidal translational dynamics about the last primal.
        """
        dt = self.config.dt
        com = np.asarray(state.com, dtype=float).copy()
        vel = np.asarray(state.com_velocity, dtype=float).copy()
        seq = [com.copy()]
        gravity = np.array([0.0, 0.0, -GRAVITY])
        for k in range(self.config.horizon - 1):
            force = self._last_horizon[k].sum(axis=0)
            acc = force / self.mass + gravity
            vel = vel + acc * dt
            com = com + vel * dt
            seq.append(com.copy())
        return seq

    # -- sole CoP ---------------------------------------------------------

    def _fill_cop(
        self,
        positions: np.ndarray,
        yaw: float,
    ) -> None:
        """Write per-foot CoP-in-sole inequalities into the constraint block.

        ``|Σ fz o_axis| ≤ (half − margin) Σ fz`` in the heading frame, four
        linear inequalities per foot per horizon step. ``fz ≥ 0`` at the four
        corners already keeps CoP in the hull; the tiny numeric margin is only
        so OSQP does not sit on a singular edge. The gait ZMP inset is the
        *reference*; this box is the physical sole (Sleiman).
        """
        cfg = self.config
        c, s = np.cos(yaw), np.sin(yaw)
        # Columns are world (x, y); rows are (forward, left).
        rotate = np.array([[c, s], [-s, c]])
        half_f = SOLE_HALF_LENGTH
        half_l = max(1e-3, SOLE_HALF_WIDTH - float(cfg.cop_margin))
        n_fric = self._n_friction
        self._constraint[n_fric:] = 0.0
        for k in range(cfg.horizon):
            for foot in range(self.num_feet):
                sl = slice(foot * SITES_PER_FOOT, (foot + 1) * SITES_PER_FOOT)
                pts = np.asarray(positions[k, sl, :2], dtype=float)
                center = pts.mean(axis=0)
                local = (pts - center) @ rotate.T
                # Four rows: +forward, −forward, +left, −left.
                coeffs = np.stack(
                    [
                        local[:, 0] - half_f,
                        -local[:, 0] - half_f,
                        local[:, 1] - half_l,
                        -local[:, 1] - half_l,
                    ],
                    axis=0,
                )
                base = n_fric + 4 * (k * self.num_feet + foot)
                for side in range(4):
                    row = base + side
                    for j in range(SITES_PER_FOOT):
                        col = k * self.force_dim + 3 * (foot * SITES_PER_FOOT + j) + 2
                        self._constraint[row, col] = float(coeffs[side, j])

    # -- solve ------------------------------------------------------------

    def _schedule(
        self, contact_mask: np.ndarray | None, schedule: np.ndarray | None
    ) -> np.ndarray:
        """Per-horizon-step contact flags, ``(horizon, nc)`` bool."""
        n = self.config.horizon
        if schedule is not None:
            return np.asarray(schedule, dtype=bool).reshape(n, self.num_contacts)
        if contact_mask is None:
            return np.ones((n, self.num_contacts), dtype=bool)
        return np.tile(np.asarray(contact_mask, dtype=bool), (n, 1))

    def _apply_schedule(self, schedule: np.ndarray) -> None:
        """Zero the normal-force bounds of contacts that are airborne then."""
        cfg = self.config
        for k in range(cfg.horizon):
            for i in range(self.num_contacts):
                row = 5 * (k * self.num_contacts + i) + 4
                loaded = bool(schedule[k, i])
                self._lower[row] = cfg.fz_min if loaded else 0.0
                self._upper[row] = cfg.fz_max if loaded else 0.0

    def solve(
        self,
        state: CentroidalState,
        reference: CentroidalReference,
        contact_mask: np.ndarray | None = None,
        schedule: np.ndarray | None = None,
        contact_pos_seq: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the desired contact forces now, shape ``(nc, 3)``.

        ``schedule`` is ``(horizon, nc)`` planned contact flags and
        ``contact_pos_seq`` the matching ``(horizon, nc, 3)`` contact positions
        (Bézier pose while a foot is airborne, foothold once it is scheduled
        down). Omit both to hold the measured contact set across the horizon,
        which is what standing wants.
        """
        cfg = self.config
        n = cfg.horizon
        plan = self._schedule(contact_mask, schedule)
        self._apply_schedule(plan)

        x0 = np.concatenate(
            [
                state.rpy,
                state.com,
                state.angular_velocity,
                state.com_velocity,
                [-GRAVITY],
            ]
        )

        # The reference travels: holding a fixed CoM target while asking for a
        # non-zero CoM velocity is self-contradictory, and the position term
        # wins, which is how the MPC ended up braking every step.
        velocity = np.asarray(reference.com_velocity, dtype=float)
        com_ref = np.asarray(reference.com, dtype=float)
        x_ref = np.concatenate(
            [
                np.concatenate(
                    [
                        [0.0, 0.0, reference.yaw],
                        com_ref + velocity * (k + 1) * cfg.dt,
                        np.zeros(3),
                        velocity,
                        [-GRAVITY],
                    ]
                )
                for k in range(n)
            ]
        )

        a_d = self._state_matrix(float(state.rpy[2]))
        inertia_inv = np.linalg.inv(state.inertia)
        if contact_pos_seq is None:
            positions = np.tile(state.contact_pos, (n, 1, 1))
        else:
            positions = np.asarray(contact_pos_seq, dtype=float).reshape(
                n, self.num_contacts, 3
            )
        com_seq = self._rollout_com(state)
        b_seq = [
            self._input_matrix(positions[k], com_seq[k], inertia_inv) for k in range(n)
        ]
        self._fill_cop(positions, float(state.rpy[2]))
        a_qp, b_qp = self._condense(a_d, b_seq)

        state_w = np.tile(cfg.state_weights(), n)
        # Height cost 5× the plan in xy is what standing wants. Over a
        # scheduled walk it is scaled by how many contacts that horizon
        # step actually has, so single support does not spend force holding
        # a height the remaining sole cannot produce.
        for k in range(n):
            state_w[STATE_DIM * k + 5] *= float(plan[k].mean())
        drift = a_qp @ x0 - x_ref

        weighted_b = b_qp * state_w[:, None]
        hessian = 2.0 * (b_qp.T @ weighted_b)
        hessian[np.diag_indices_from(hessian)] += 2.0 * cfg.weight_force
        gradient = 2.0 * (weighted_b.T @ drift)

        solution = self._qp.solve(
            hessian, gradient, self._constraint, self._lower, self._upper
        )
        horizon_forces = solution.reshape(n, self.num_contacts, 3)
        forces = horizon_forces[0]

        if not np.all(np.isfinite(forces)):
            forces = self._gravity_share()
            horizon_forces = np.tile(forces, (n, 1, 1))
        self.last_forces = forces
        self._last_horizon = horizon_forces
        residual = drift + b_qp @ solution
        self.cost = float(residual @ (state_w * residual))
        return forces


#: Compatibility alias. This class is not Sleiman/Galliker NMPC.
CentroidalNMPC = ConvexMPC
