"""Receding-horizon position MPC for torso tracking.

This module runs an actual finite-horizon MPC loop:
1) Build state error from the latest estimate.
2) Roll out linearized torso-error dynamics over a horizon.
3) Minimize trajectory cost over that horizon.
4) Apply only the first action (receding horizon).

The control output remains joint positions (q_ref), while actuator PD handles
torque generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import numpy as np

from loka.nodes.actuator_manager import quat_to_rpy
from loka.nodes.state_estimator import RobotState

NUM_MOTORS = 29
PLANNER_OPTIONS = ("Convex QP", "Predictive Sampling", "iLQG")


@dataclass
class TorsoTarget:
    """Desired torso / base pose in world frame."""

    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


@dataclass
class PositionMPCWeights:
    w_z: float = 50.0
    w_xy: float = 10.0
    w_theta: float = 40.0
    w_q: float = 0.5

    @classmethod
    def from_dict(cls, d: Mapping) -> "PositionMPCWeights":
        return cls(
            w_z=float(d.get("w_z", 50.0)),
            w_xy=float(d.get("w_xy", d.get("w_p", 10.0))),
            w_theta=float(d.get("w_theta", 40.0)),
            w_q=float(d.get("w_q", 0.5)),
        )


@dataclass
class PositionMPCResult:
    q_ref: np.ndarray
    cost: float
    torso_error: np.ndarray  # [e_x, e_y, e_z, e_roll, e_pitch]
    planner: str = "Convex QP"


def _sensitivity_jacobian() -> np.ndarray:
    """Map from reduced joint action to torso pose channels.

    u = [hip_pitch, ankle_pitch, hip_roll, ankle_roll, knee, waist_pitch, waist_roll]
    pose channels = [ex, ey, ez, eroll, epitch]
    """
    J = np.zeros((5, 7), dtype=np.float64)
    # hip_p / ank_p -> forward (x) and pitch
    J[0, 0] = 0.35
    J[0, 1] = 0.25
    J[4, 0] = 2.0
    J[4, 1] = 2.5
    # hip_r / ank_r -> lateral (y) and roll
    J[1, 2] = 0.35
    J[3, 2] = 2.0
    J[3, 3] = 2.0
    # knee -> height (more flexion lowers z)
    J[2, 4] = -0.5
    # waist pitch / roll
    J[4, 5] = 2.0
    J[3, 6] = 2.0
    return J


class TorsoPositionMPC:
    """Finite-horizon torso MPC returning position references."""

    def __init__(
        self,
        q_nominal: np.ndarray,
        weights: Optional[PositionMPCWeights] = None,
        *,
        planner: str = "Convex QP",
        max_delta: float = 0.4,
        z_track_band: float = 0.20,
        horizon_steps: int = 50,
        model_dt: float = 0.02,
        dyn_stiffness: float = 16.0,
        dyn_damping: float = 8.0,
        num_samples: int = 64,
        sample_std: float = 0.08,
        ilqg_iters: int = 8,
        seed: int = 0,
    ) -> None:
        self.q_nominal = np.asarray(q_nominal, dtype=np.float64).copy()
        self.weights = weights or PositionMPCWeights()
        self.planner = planner if planner in PLANNER_OPTIONS else "Convex QP"
        self.max_delta = float(max_delta)
        self.z_track_band = float(z_track_band)
        self.horizon_steps = max(4, int(horizon_steps))
        self.model_dt = max(1e-3, float(model_dt))
        self.dyn_stiffness = max(1e-3, float(dyn_stiffness))
        self.dyn_damping = max(1e-3, float(dyn_damping))
        self.num_samples = int(num_samples)
        self.sample_std = float(sample_std)
        self.ilqg_iters = int(ilqg_iters)
        self._rng = np.random.default_rng(seed)
        self._anchor_xy: Optional[Tuple[float, float]] = None
        self._anchor_yaw: float = 0.0
        self._J = _sensitivity_jacobian()
        self._Qf_scale = 4.0

    def set_planner(self, name: str) -> None:
        if name in PLANNER_OPTIONS:
            self.planner = name

    def update_weights(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self.weights, key):
                setattr(self.weights, key, float(value))

    def reset_anchor(self, state: RobotState) -> None:
        self._anchor_xy = (float(state.base_pos[0]), float(state.base_pos[1]))
        _r, _p, yaw = quat_to_rpy(state.base_quat)
        self._anchor_yaw = float(yaw)

    def target_from_height(self, z_ref: float) -> TorsoTarget:
        if self._anchor_xy is None:
            x, y = 0.0, 0.0
        else:
            x, y = self._anchor_xy
        return TorsoTarget(
            x=x, y=y, z=float(z_ref), roll=0.0, pitch=0.0, yaw=self._anchor_yaw
        )

    def _error_and_weights(self, state: RobotState, z_ref: float):
        target = self.target_from_height(z_ref)
        w = self.weights
        roll, pitch, _yaw = quat_to_rpy(state.base_quat)
        err_pose = np.array(
            [
                float(state.base_pos[0]) - target.x,
                float(state.base_pos[1]) - target.y,
                float(state.base_pos[2]) - target.z,
                float(roll) - target.roll,
                float(pitch) - target.pitch,
            ],
            dtype=np.float64,
        )
        track_z = abs(err_pose[2]) <= self.z_track_band
        w_diag = np.array(
            [
                w.w_xy,
                w.w_xy,
                w.w_z if track_z else 0.0,
                w.w_theta,
                w.w_theta,
            ],
            dtype=np.float64,
        )
        vel = np.array(
            [
                float(state.base_lin_vel[0]),
                float(state.base_lin_vel[1]),
                float(state.base_lin_vel[2]),
                float(state.base_ang_vel[0]),
                float(state.base_ang_vel[1]),
            ],
            dtype=np.float64,
        )
        x0 = np.concatenate([err_pose, vel], axis=0)
        return x0, err_pose, np.diag(w_diag), track_z

    def _linear_dynamics(self, use_waist: bool) -> Tuple[np.ndarray, np.ndarray]:
        """x[k+1] = A x[k] + B u[k], x=[pose_err(5), vel_err(5)]."""
        n_u = 7 if use_waist else 5
        dt = self.model_dt
        kp = self.dyn_stiffness
        kd = self.dyn_damping

        A = np.eye(10, dtype=np.float64)
        # e_{k+1} = e_k + dt * v_k
        A[0:5, 5:10] = dt * np.eye(5)
        # v_{k+1} = v_k + dt*(-kp*e_k - kd*v_k + kp*J*u_k)
        A[5:10, 0:5] = -dt * kp * np.eye(5)
        A[5:10, 5:10] = (1.0 - dt * kd) * np.eye(5)

        B = np.zeros((10, n_u), dtype=np.float64)
        B[5:10, :] = dt * kp * self._J[:, :n_u]
        return A, B

    def _sequence_cost(
        self,
        x0: np.ndarray,
        U: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        Qf: np.ndarray,
    ) -> float:
        x = x0.copy()
        total = 0.0
        for k in range(U.shape[0]):
            u = U[k]
            total += float(x[:5] @ Q @ x[:5] + u @ R @ u)
            x = A @ x + B @ u
        total += float(x[:5] @ Qf @ x[:5])
        return total

    def _apply_u(self, u: np.ndarray, *, use_waist: bool) -> np.ndarray:
        """Map reduced action u to 29-joint position reference."""
        u = np.clip(u, -self.max_delta, self.max_delta)
        d_hip_p, d_ank_p, d_hip_r, d_ank_r, d_knee, d_w_p, d_w_r = [
            float(v) for v in u
        ]
        q = self.q_nominal.copy()
        for base in (0, 6):
            q[base + 0] += d_hip_p
            q[base + 1] += d_hip_r if base == 0 else -d_hip_r
            q[base + 3] += d_knee
            q[base + 4] += d_ank_p
            q[base + 5] += d_ank_r if base == 0 else -d_ank_r
        if use_waist:
            q[14] += d_w_p  # waist_pitch
            q[13] += d_w_r  # waist_roll
        total_delta = q - self.q_nominal
        total_delta = np.clip(total_delta, -self.max_delta, self.max_delta)
        q = self.q_nominal + total_delta
        return q

    def _solve_lqr_sequence(
        self,
        x0: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        Qf: np.ndarray,
    ) -> np.ndarray:
        """Finite-horizon time-varying LQR sequence for linear model."""
        n_x = A.shape[0]
        n_u = B.shape[1]
        N = self.horizon_steps
        P = [np.zeros((n_x, n_x), dtype=np.float64) for _ in range(N + 1)]
        K = [np.zeros((n_u, n_x), dtype=np.float64) for _ in range(N)]
        P[N][0:5, 0:5] = Qf

        Qx = np.zeros((n_x, n_x), dtype=np.float64)
        Qx[0:5, 0:5] = Q

        for k in range(N - 1, -1, -1):
            BtPB = B.T @ P[k + 1] @ B
            S = R + BtPB + 1e-8 * np.eye(n_u)
            F = B.T @ P[k + 1] @ A
            try:
                K[k] = np.linalg.solve(S, F)
            except np.linalg.LinAlgError:
                K[k] = np.linalg.pinv(S) @ F
            Acl = A - B @ K[k]
            P[k] = Qx + K[k].T @ R @ K[k] + Acl.T @ P[k + 1] @ Acl

        U = np.zeros((N, n_u), dtype=np.float64)
        x = x0.copy()
        for k in range(N):
            u = -K[k] @ x
            U[k] = np.clip(u, -self.max_delta, self.max_delta)
            x = A @ x + B @ U[k]
        return U

    def _solve_sampling(
        self,
        x0: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        Qf: np.ndarray,
        U_seed: np.ndarray,
    ) -> np.ndarray:
        n_u = B.shape[1]
        best = U_seed.copy()
        best_c = self._sequence_cost(x0, best, A, B, Q, R, Qf)
        noise = self._rng.normal(
            0.0, self.sample_std, size=(self.num_samples, self.horizon_steps, n_u)
        )
        for i in range(self.num_samples):
            trial = np.clip(U_seed + noise[i], -self.max_delta, self.max_delta)
            c = self._sequence_cost(x0, trial, A, B, Q, R, Qf)
            if c < best_c:
                best_c = c
                best = trial
        return best

    def _solve_ilqg(
        self,
        x0: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        Qf: np.ndarray,
        U_seed: np.ndarray,
    ) -> np.ndarray:
        """Simple sequence-level iLQG refinement over finite horizon."""
        U = U_seed.copy()
        eps = 1e-3
        for _ in range(self.ilqg_iters):
            base = self._sequence_cost(x0, U, A, B, Q, R, Qf)
            grad = np.zeros_like(U)
            for k in range(self.horizon_steps):
                for j in range(U.shape[1]):
                    up = U.copy()
                    um = U.copy()
                    up[k, j] += eps
                    um[k, j] -= eps
                    cp = self._sequence_cost(
                        x0, np.clip(up, -self.max_delta, self.max_delta), A, B, Q, R, Qf
                    )
                    cm = self._sequence_cost(
                        x0, np.clip(um, -self.max_delta, self.max_delta), A, B, Q, R, Qf
                    )
                    grad[k, j] = (cp - cm) / (2.0 * eps)
            scale = 0.15
            accepted = False
            for _ in range(6):
                trial = np.clip(U - scale * grad, -self.max_delta, self.max_delta)
                c = self._sequence_cost(x0, trial, A, B, Q, R, Qf)
                if c <= base:
                    U = trial
                    accepted = True
                    break
                scale *= 0.5
            if not accepted:
                break
        return U

    def solve(
        self,
        state: RobotState,
        z_ref: float,
        *,
        use_waist: bool = False,
    ) -> PositionMPCResult:
        """Compute q_ref using receding-horizon trajectory optimization."""
        x0, err_pose, W, _track_z = self._error_and_weights(state, z_ref)
        A, B = self._linear_dynamics(use_waist=use_waist)
        Q = W
        Qf = self._Qf_scale * W
        R = max(self.weights.w_q, 1e-6) * np.eye(B.shape[1], dtype=np.float64)

        U_lqr = self._solve_lqr_sequence(x0, A, B, Q, R, Qf)
        if self.planner == "Predictive Sampling":
            U = self._solve_sampling(x0, A, B, Q, R, Qf, U_lqr)
        elif self.planner == "iLQG":
            U = self._solve_ilqg(x0, A, B, Q, R, Qf, U_lqr)
        else:
            U = U_lqr

        u0 = U[0]
        q = self._apply_u(u0, use_waist=use_waist)
        cost = self._sequence_cost(x0, U, A, B, Q, R, Qf)
        return PositionMPCResult(
            q_ref=q, cost=cost, torso_error=err_pose, planner=self.planner
        )
