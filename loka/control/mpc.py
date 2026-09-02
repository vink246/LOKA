"""Convex single-rigid-body MPC over ground-reaction forces.

This is the standard SRBD formulation used by the MIT Cheetah convex MPC
(Di Carlo et al., IROS 2018) and by most humanoid MPC+WBC stacks: collapse the
robot to one rigid body, keep the contact points fixed over the horizon, and
optimise the contact forces that drive the centroidal state to its reference.

The state is ``x = [θ, p, ω, v, g] ∈ R¹³`` where ``θ`` is the base
roll-pitch-yaw, ``p`` / ``v`` are the CoM position / velocity, ``ω`` is the
world-frame angular velocity, and the trailing gravity constant makes the
dynamics affine-free (a plain linear system), which is what keeps the whole
problem a single convex QP.

Only the first force block of the solution is used; the horizon exists so the
controller anticipates where the centroidal state is heading rather than
reacting to it.

Walking feeds the horizon a *schedule*: which feet are planted at each step and
where they are planned to be. A 0.3 s horizon spans most of a step, so pinning
the measured contact set across all of it -- which is what a standing
controller can get away with -- would have the force plan bracing against feet
that are in the air.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from loka.control.qp import QP, Sparsity
from loka.control.robot import GRAVITY

STATE_DIM = 13


@dataclass
class MPCConfig:
    #: 6 x 50 ms = 0.3 s lookahead. Longer horizons buy nothing for standing
    #: and the condensed QP cost grows roughly cubically in the horizon.
    horizon: int = 6
    dt: float = 0.05
    friction_mu: float = 0.5
    #: Per-contact normal force limits [N]. A positive minimum keeps every
    #: contact loaded, which is what makes a two-foot stand stiff.
    fz_min: float = 5.0
    fz_max: float = 400.0
    #: Diagonal state cost over [roll, pitch, yaw, x, y, z, ωx, ωy, ωz, vx, vy, vz].
    weight_orientation: Sequence[float] = (500.0, 500.0, 300.0)
    weight_position: Sequence[float] = (200.0, 200.0, 1000.0)
    weight_angular_velocity: Sequence[float] = (10.0, 10.0, 10.0)
    weight_linear_velocity: Sequence[float] = (60.0, 60.0, 60.0)
    #: Force-effort weight. Must stay tiny: the forces are O(100 N) so their
    #: squares swamp the tracking terms unless this is several orders of
    #: magnitude below the state weights. It exists only to pick a sensible
    #: (evenly shared) point out of the redundant eight-contact null space.
    weight_force: float = 1e-6

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
    """Solves for a horizon of contact forces; returns the first block."""

    def __init__(self, config: MPCConfig, num_contacts: int, mass: float) -> None:
        self.config = config
        self.num_contacts = num_contacts
        self.mass = mass
        self.force_dim = 3 * num_contacts
        self.num_vars = self.force_dim * config.horizon
        # The friction constraints never change, so OSQP keeps them verbatim
        # from setup; only the (dense) cost is refreshed each solve.
        self._constraint, self._lower, self._upper = self._build_constraints()
        # The friction block is re-uploaded each solve so that mu and the force
        # limits can be retuned live; its pattern never changes, so this costs
        # a fancy-index copy of a very sparse matrix.
        self._qp = QP(
            "mpc",
            hessian_pattern=Sparsity.upper_triangular(self.num_vars),
            constraint_pattern=Sparsity(self._constraint != 0.0),
        )
        self.last_forces = self._gravity_share()
        self.cost = 0.0

    # -- setup ------------------------------------------------------------

    def _gravity_share(self) -> np.ndarray:
        """Even split of body weight across all contacts, used before the
        first solve and as the fallback if a solve fails."""
        forces = np.zeros((self.num_contacts, 3))
        forces[:, 2] = self.mass * GRAVITY / self.num_contacts
        return forces

    def _build_constraints(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Linearised friction pyramid plus normal-force bounds.

        Constant across solves: the pattern and the numbers only depend on μ
        and the force limits.
        """
        cfg = self.config
        mu = cfg.friction_mu
        # |fx| ≤ μ fz, |fy| ≤ μ fz, fz_min ≤ fz ≤ fz_max
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
        constraint = np.zeros((rows_per_contact * num_blocks, self.num_vars))
        for b in range(num_blocks):
            r = b * rows_per_contact
            c = b * 3
            constraint[r : r + rows_per_contact, c : c + 3] = block
        lower = np.tile(lower_block, num_blocks)
        upper = np.tile(upper_block, num_blocks)
        return constraint, lower, upper

    def refresh(self) -> None:
        """Re-read config values that are baked into the constraint block.

        The cost is rebuilt from ``self.config`` on every solve, so weights
        need no refresh at all; only ``friction_mu`` and the force limits land
        here.
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
        for i in range(self.num_contacts):
            b_c[6:9, 3 * i : 3 * i + 3] = inertia_inv @ _skew(offsets[i])
            b_c[9:12, 3 * i : 3 * i + 3] = np.eye(3) / self.mass
        return b_c * self.config.dt

    def _condense(
        self, a_d: np.ndarray, b_seq: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Roll the horizon into ``X = A_qp x₀ + B_qp U``.

        ``A`` is held constant over the horizon (it depends only on yaw), so its
        powers are accumulated once; ``B`` varies per step with the planned
        contact positions.
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
        ``contact_pos_seq`` the matching ``(horizon, nc, 3)`` contact positions.
        Omit both to hold the measured contact set across the horizon, which is
        what standing wants.
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
            b_seq = [self._input_matrix(state.contact_pos, state.com, inertia_inv)] * n
        else:
            positions = np.asarray(contact_pos_seq, dtype=float).reshape(
                n, self.num_contacts, 3
            )
            b_seq = [
                self._input_matrix(
                    positions[k], state.com + velocity * k * cfg.dt, inertia_inv
                )
                for k in range(n)
            ]
        a_qp, b_qp = self._condense(a_d, b_seq)

        state_w = np.tile(cfg.state_weights(), n)
        drift = a_qp @ x0 - x_ref

        weighted_b = b_qp * state_w[:, None]
        hessian = 2.0 * (b_qp.T @ weighted_b)
        hessian[np.diag_indices_from(hessian)] += 2.0 * cfg.weight_force
        gradient = 2.0 * (weighted_b.T @ drift)

        solution = self._qp.solve(
            hessian, gradient, self._constraint, self._lower, self._upper
        )
        forces = solution[: self.force_dim].reshape(self.num_contacts, 3)

        if not np.all(np.isfinite(forces)):
            forces = self._gravity_share()
        self.last_forces = forces
        residual = drift + b_qp @ solution
        self.cost = float(residual @ (state_w * residual))
        return forces
