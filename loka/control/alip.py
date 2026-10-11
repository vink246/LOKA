"""ALIP step-to-step model for the G1.

The state in each plane is ``(x, L)``: horizontal CoM offset from the stance
contact, and the matching component of angular momentum about that contact.
``L`` is the Gong & Grizzle ALIP coordinate (it reduces to an H-LIP when
centroidal angular momentum is zero). The step-to-step map, double-support
drift and deadbeat/LQR stepping law are Xiong & Ames, T-RO 2022.

Sign convention, stance yaw frame (``forward``, ``left``):

* sagittal: ``x = (c - p) · forward``, ``L = L_world · left`` (``+L_y``)
* lateral:  ``x = (c - p) · left``,     ``L = -L_world · forward`` (``-L_x``)

so that both planes obey ``x_dot = L / (m H)`` and ``L_dot = m g (x - d)``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.linalg import solve_discrete_are

GRAVITY = 9.81

#: Per-step yaw cap [rad]. Kajita's HRP-2L turned about 0.34 rad/step; 0.25
#: is the starting value and is what ``walk_bench`` sizes ALIP deadlines to.
ALIP_TURN_CAP = 0.25

#: Fraction of double support over which the CoP moves from the old contact
#: to the new one; it then stays on the new contact. Fitted to closed-loop
#: G1 steps under the convex MPC, whose 0.3 s horizon loads the new foot
#: early. 1.0 is a linear transfer over the whole double support.
DS_TRANSFER_FRACTION = 0.8


def lip_matrix(t: float, mass: float, height: float) -> np.ndarray:
    """State transition of ``[x, L]`` over ``t`` seconds with CoP at the contact."""
    h = max(float(height), 0.05)
    m = max(float(mass), 1e-3)
    lam = np.sqrt(GRAVITY / h)
    ch = np.cosh(lam * t)
    sh = np.sinh(lam * t)
    return np.array(
        [[ch, sh / (m * h * lam)], [m * h * lam * sh, ch]], dtype=float
    )


def propagate(state: np.ndarray, t: float, mass: float, height: float, cop: float = 0.0) -> np.ndarray:
    """Advance ``[x, L]`` with a constant CoP offset ``cop`` from the contact."""
    x0, L0 = float(state[0]), float(state[1])
    d = float(cop)
    y = lip_matrix(t, mass, height) @ np.array([x0 - d, L0])
    return np.array([y[0] + d, y[1]])


def double_support_matrix(t_ds: float, mass: float, height: float) -> np.ndarray:
    """Xiong & Ames DSP: ``L`` constant, ``x`` drifts at ``L / (m H)``."""
    h = max(float(height), 0.05)
    m = max(float(mass), 1e-3)
    return np.array([[1.0, float(t_ds) / (m * h)], [0.0, 1.0]], dtype=float)


def _transfer_flow(t: float, mass: float, height: float) -> np.ndarray:
    """``expm`` of ``[x, L, d, d_dot]`` with the CoP ``d`` moving at a constant rate."""
    # Quantised so the 500 Hz caller hits the cache: 1 ms, 1 g, 1 mm.
    return _transfer_flow_cached(
        round(max(float(t), 0.0), 3), round(float(mass), 3), round(float(height), 3)
    ).copy()


@lru_cache(maxsize=4096)
def _transfer_flow_cached(t: float, mass: float, height: float) -> np.ndarray:
    from scipy.linalg import expm

    h = max(float(height), 0.05)
    m = max(float(mass), 1e-3)
    f = np.zeros((4, 4))
    f[0, 1] = 1.0 / (m * h)
    f[1, 0] = m * GRAVITY
    f[1, 2] = -m * GRAVITY
    f[2, 3] = 1.0
    return expm(f * max(float(t), 0.0))


def double_support_transfer(
    state: np.ndarray, u_prev: float, t_in_ds: float, t_ds: float, mass: float, height: float
) -> np.ndarray:
    """``[x, L]`` at the end of double support, CoP moving old foot -> new foot.

    ``state`` is relative to the new stance contact. The old contact sits at
    ``-u_prev``; the CoP starts there and reaches the new contact at ``t_ds``.
    Measured on the G1: the Xiong & Ames hold (``L`` constant) misses the
    velocity reversal in a 0.2 s double support by 0.2-0.4 m/s.
    """
    if t_ds <= 1e-4:
        return np.asarray(state, dtype=float).reshape(2).copy()
    t_move = max(DS_TRANSFER_FRACTION * float(t_ds), 1e-4)
    t_in = float(t_in_ds)
    z = np.asarray(state, dtype=float).reshape(2).copy()
    if t_in < t_move:
        rate = float(u_prev) / t_move
        d_now = -float(u_prev) + rate * t_in
        aug = np.array([z[0], z[1], d_now, rate])
        z = (_transfer_flow(t_move - t_in, mass, height) @ aug)[:2]
        t_in = t_move
    return lip_matrix(float(t_ds) - t_in, mass, height) @ z


def step_to_step(mass: float, height: float, t_ss: float, t_ds: float) -> tuple[np.ndarray, np.ndarray]:
    """Pre-impact map ``X+ = A X + B u``.

    Order inside the step: impact ``x := x - u``, then DSP with the CoP
    moving from the old contact to the new one, then SSP.
    ``u`` is the next contact's coordinate relative to the current one.
    """
    a_ss = lip_matrix(t_ss, mass, height)
    if t_ds <= 1e-4:
        return a_ss, -(a_ss @ np.array([1.0, 0.0]))
    t_move = max(DS_TRANSFER_FRACTION * float(t_ds), 1e-4)
    phi = _transfer_flow(t_move, mass, height)
    rest = lip_matrix(float(t_ds) - t_move, mass, height)
    p = rest @ phi[:2, :2]
    b_ds = rest @ (-phi[:2, :2][:, 0] - phi[:2, 2] + phi[:2, 3] / t_move)
    return a_ss @ p, a_ss @ b_ds


def deadbeat_gain(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row gain placing both closed-loop eigenvalues at 0 (Ackermann)."""
    b = np.asarray(b, dtype=float).reshape(2)
    ctr = np.column_stack([b, a @ b])
    # Ackermann's K is for ``u = -K e``. The stepping law uses ``u = K_law e``,
    # so the closed loop is ``A + B K_law`` and ``K_law = -K``.
    k = np.array([0.0, 1.0]) @ np.linalg.inv(ctr) @ (a @ a)
    return -k


def state_weight(mass: float, height: float) -> np.ndarray:
    """LQR state weight on ``(x [m], L / (m H) [m/s])``.

    With ``Q = I`` the momentum term is ~(m H)^2 heavier than position, the
    step cost is negligible, and every ``capture_gain`` below 1 is deadbeat.
    """
    mh = max(float(mass), 1e-3) * max(float(height), 0.05)
    return np.diag([1.0, 1.0 / (mh * mh)])


def lqr_gain(
    a: np.ndarray, b: np.ndarray, r_weight: float, q: np.ndarray | None = None
) -> np.ndarray:
    """Row gain ``K`` in ``u = -K e`` is returned as the law ``u = K_law e`` with ``K_law = -K``."""
    bcol = np.asarray(b, dtype=float).reshape(2, 1)
    r = np.array([[max(float(r_weight), 1e-8)]])
    qmat = np.eye(2) if q is None else np.asarray(q, dtype=float)
    p = solve_discrete_are(a, bcol, qmat, r)
    k = np.linalg.solve(r + bcol.T @ p @ bcol, bcol.T @ p @ a)
    return -k.reshape(2)


def stepping_gain(
    a: np.ndarray, b: np.ndarray, capture_gain: float, q: np.ndarray | None = None
) -> np.ndarray:
    """``u = u* + K (X - X*)``. ``capture_gain`` 1 is deadbeat; below that, DLQR.

    ``R`` grows as the gain falls, so a small ``capture_gain`` barely moves
    the foot off the nominal orbit. Pass ``q = state_weight(m, H)``.
    """
    g = float(np.clip(capture_gain, 0.0, 1.0))
    if g >= 0.999:
        return deadbeat_gain(a, b)
    if g <= 1e-6:
        return np.zeros(2)
    return lqr_gain(a, b, (1.0 - g) / g, q)


def p1_orbit(a: np.ndarray, b: np.ndarray, step_length: float) -> np.ndarray:
    """Sagittal fixed point for a constant step ``step_length``."""
    b = np.asarray(b, dtype=float).reshape(2)
    return np.linalg.solve(np.eye(2) - a, b * float(step_length))


def p2_orbit(
    a: np.ndarray, b: np.ndarray, u_left_stance: float, u_right_stance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Lateral two-step orbit.

    ``u_left_stance`` is the step taken while the left foot is in stance
    (it places the right foot). Returns ``(X_left, X_right)`` pre-impact.
    """
    b = np.asarray(b, dtype=float).reshape(2)
    # X_R = A X_L + B u_L,  X_L = A X_R + B u_R
    # (I - A^2) X_L = A B u_L + B u_R
    rhs = a @ b * float(u_left_stance) + b * float(u_right_stance)
    x_left = np.linalg.solve(np.eye(2) - a @ a, rhs)
    x_right = a @ x_left + b * float(u_left_stance)
    return x_left, x_right


def lateral_step(stance_leg: int, width: float, v_lat: float, t_step: float) -> float:
    """Nominal lateral step. Left stance places the right foot at ``-width``."""
    sign = -1.0 if int(stance_leg) == 0 else 1.0
    return float(v_lat) * float(t_step) + sign * float(width)


def plane_from_world(
    com_xy: np.ndarray,
    contact_xy: np.ndarray,
    angmom_world: np.ndarray,
    forward: np.ndarray,
    left: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """World measurement to ``(sagittal [x, L], lateral [x, L])``."""
    delta = np.asarray(com_xy, dtype=float).reshape(2) - np.asarray(contact_xy, dtype=float).reshape(2)
    # L_world is 3D. Horizontal axes of the stance frame live in xy; the
    # vertical component of L does not enter either plane.
    fwd3 = np.array([forward[0], forward[1], 0.0])
    left3 = np.array([left[0], left[1], 0.0])
    L = np.asarray(angmom_world, dtype=float).reshape(3)
    sag = np.array([float(np.dot(delta, forward)), float(np.dot(L, left3))])
    lat = np.array([float(np.dot(delta, left)), float(-np.dot(L, fwd3))])
    return sag, lat


def predicted_preimpact(
    state: np.ndarray,
    *,
    t_in_step: float,
    t_ss: float,
    t_ds: float,
    mass: float,
    height: float,
    cop: float = 0.0,
    u_prev: float = 0.0,
) -> np.ndarray:
    """Pre-impact ``[x, L]`` from a measurement taken ``t_in_step`` into the step.

    Double support runs first (CoP moving from the old contact, ``-u_prev``,
    to this one), then single support with the CoP ``cop`` from the contact.
    """
    if t_in_step < t_ds:
        at_ss = double_support_transfer(state, u_prev, t_in_step, t_ds, mass, height)
        return propagate(at_ss, t_ss, mass, height, cop)
    elapsed_ss = t_in_step - t_ds
    remain_ss = max(t_ss - elapsed_ss, 0.0)
    return propagate(state, remain_ss, mass, height, cop)


class CopEstimator:
    """Effective CoP offset per plane from ``L_dot = m g (x - d)``.

    The point-foot model has ``d = 0``. On the G1 the force plan, the flat
    sole and the swing leg's reaction all move the effective pivot, and the
    stepping law over-steps unless the prediction knows where it is.
    """

    def __init__(self, time_constant: float = 0.04, limit: float = 0.06) -> None:
        self.time_constant = float(time_constant)
        self.limit = float(limit)
        self.reset()

    def reset(self) -> None:
        self.d = np.zeros(2)
        self._prev: tuple[float, np.ndarray] | None = None

    def update(self, t: float, x: np.ndarray, L: np.ndarray, mass: float) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(2)
        L = np.asarray(L, dtype=float).reshape(2)
        if self._prev is not None:
            t0, L0 = self._prev
            dt = float(t) - t0
            if dt > 1e-6:
                raw = x - (L - L0) / (dt * max(float(mass), 1e-3) * GRAVITY)
                raw = np.clip(raw, -self.limit, self.limit)
                alpha = min(1.0, dt / max(self.time_constant, dt))
                self.d = self.d + alpha * (raw - self.d)
        self._prev = (float(t), L.copy())
        return self.d.copy()


class StepMapEstimator:
    """Online correction to the ALIP step-to-step map, one plane.

    Works in scaled coordinates ``X = (x [m], L / (m H) [m/s])``. The plant
    map is taken as ``X+ = (A_m + dA) X + (B_m + dB) u + c s`` where
    ``(A_m, B_m)`` is the ALIP model and ``s`` is +1 sagittally and the
    stance side laterally. Recursive least squares with forgetting fits
    ``(dA, dB, c)`` from each touchdown; with no data they are zero and the
    law is the pure model.

    The model is exact in single support. The convex MPC reshapes double
    support and moves the sagittal pivot, which is what ``dA, dB, c`` absorb.
    """

    def __init__(
        self,
        forgetting: float = 0.95,
        prior_std: tuple[float, float, float, float] = (0.3, 0.3, 0.3, 0.05),
        outlier: float = 0.6,
    ) -> None:
        self.forgetting = float(forgetting)
        self.p0 = np.diag(np.square(np.asarray(prior_std, dtype=float)))
        self.outlier = float(outlier)
        self.reset()

    def reset(self) -> None:
        self.theta = np.zeros((4, 2))
        self.p = self.p0.copy()
        self.updates = 0

    def update(
        self,
        x_k: np.ndarray,
        u_k: float,
        s_k: float,
        x_next: np.ndarray,
        a_model: np.ndarray,
        b_model: np.ndarray,
    ) -> bool:
        phi = np.array([float(x_k[0]), float(x_k[1]), float(u_k), float(s_k)])
        residual = (
            np.asarray(x_next, dtype=float)
            - a_model @ np.asarray(x_k, dtype=float)
            - np.asarray(b_model, dtype=float) * float(u_k)
        )
        innovation = residual - self.theta.T @ phi
        if not np.all(np.isfinite(innovation)) or np.max(np.abs(innovation)) > self.outlier:
            return False
        lam = self.forgetting
        pphi = self.p @ phi
        gain = pphi / (lam + float(phi @ pphi))
        self.theta = self.theta + np.outer(gain, innovation)
        self.p = (self.p - np.outer(gain, pphi)) / lam
        # Bounded covariance: forgetting on a steady gait is not persistently
        # exciting, so without this the estimate winds up and jumps.
        scale = float(np.trace(self.p) / np.trace(self.p0))
        if scale > 1.0:
            self.p = self.p / scale
        self.updates += 1
        return True

    def corrected(self, a_model: np.ndarray, b_model: np.ndarray):
        """``(A, B, c)`` of the identified map."""
        a = np.asarray(a_model, dtype=float) + self.theta[:2, :].T
        b = np.asarray(b_model, dtype=float).reshape(2) + self.theta[2, :]
        c = self.theta[3, :].copy()
        return a, b, c


def scaled_model(a: np.ndarray, b: np.ndarray, mass: float, height: float):
    """``(A, B)`` in ``(x, L / (m H))`` coordinates."""
    mh = max(float(mass), 1e-3) * max(float(height), 0.05)
    s = np.diag([1.0, 1.0 / mh])
    s_inv = np.diag([1.0, mh])
    return s @ a @ s_inv, s @ np.asarray(b, dtype=float).reshape(2)


def affine_p1(a: np.ndarray, b: np.ndarray, c: np.ndarray, u_star: float) -> np.ndarray:
    """Fixed point of ``X+ = A X + B u* + c``."""
    return np.linalg.solve(np.eye(2) - a, b * float(u_star) + c)


def affine_p2(a, b, c, u_left: float, u_right: float):
    """Two-step fixed point with ``+c`` after left stance and ``-c`` after right."""
    rhs = a @ (b * float(u_left) + c) + b * float(u_right) - c
    x_left = np.linalg.solve(np.eye(2) - a @ a, rhs)
    x_right = a @ x_left + b * float(u_left) + c
    return x_left, x_right


class OrbitCache:
    """Gains and orbits for one ``(m, H, T_ss, T_ds)``. Rebuilt when they change."""

    def __init__(self) -> None:
        self.key: tuple | None = None
        self._p1: dict = {}
        self._p2: dict = {}
        self.a = np.eye(2)
        self.b = np.zeros(2)
        self.k = np.zeros(2)

    def refresh(
        self, mass: float, height: float, t_ss: float, t_ds: float, capture_gain: float
    ) -> None:
        # The gait latch slews period and duty every tick. Quantise so the
        # Riccati solve and expm run when the step changes, not at 500 Hz.
        key = (
            round(float(mass), 1),
            round(float(height) / 0.005) * 0.005,
            round(float(t_ss) / 0.005) * 0.005,
            round(float(t_ds) / 0.005) * 0.005,
            round(float(capture_gain), 3),
        )
        if key == self.key:
            return
        m, h, tss, tds, g = key
        self.a, self.b = step_to_step(m, h, max(tss, 1e-3), tds)
        self.q = state_weight(m, h)
        self.k = stepping_gain(self.a, self.b, g, self.q)
        self._p2 = {}
        self._p1 = {}
        self.key = key

    def _p1_cached(self, u_star: float) -> np.ndarray:
        key = round(float(u_star), 4)
        if key not in self._p1:
            self._p1[key] = p1_orbit(self.a, self.b, key)
        return self._p1[key]

    def _p2_cached(self, u_l: float, u_r: float):
        key = (round(float(u_l), 4), round(float(u_r), 4))
        if key not in self._p2:
            self._p2[key] = p2_orbit(self.a, self.b, key[0], key[1])
        return self._p2[key]

    def sagittal_step(self, x_pre: np.ndarray, v_des: float, t_step: float) -> float:
        u_star = float(v_des) * float(t_step)
        x_star = self._p1_cached(u_star)
        return u_star + float(self.k @ (np.asarray(x_pre, dtype=float) - x_star))

    def lateral_step(
        self,
        x_pre: np.ndarray,
        stance_leg: int,
        width: float,
        v_lat: float,
        t_step: float,
    ) -> float:
        u_l = lateral_step(0, width, v_lat, t_step)
        u_r = lateral_step(1, width, v_lat, t_step)
        x_l, x_r = self._p2_cached(u_l, u_r)
        if int(stance_leg) == 0:
            u_star, x_star = u_l, x_l
        else:
            u_star, x_star = u_r, x_r
        return u_star + float(self.k @ (np.asarray(x_pre, dtype=float) - x_star))

    def closed_loop(self) -> np.ndarray:
        return self.a + np.outer(self.b, self.k)
