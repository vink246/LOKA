"""Modular MPC planners (Convex QP, Predictive Sampling, iLQG)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Type

import numpy as np

from loka.nodes.cpg_generator import CPGOutput
from loka.nodes.state_estimator import RobotState
from loka.nodes.trajectory_planner import TorsoReference

NUM_MOTORS = 29
LEG_DIM = 12


@dataclass
class MPCWeights:
    w_z: float = 50.0
    w_p: float = 20.0
    w_u: float = 1.0
    w_theta: float = 10.0
    w_q: float = 5.0
    w_tau: float = 0.01

    @classmethod
    def from_dict(cls, d: Mapping) -> "MPCWeights":
        return cls(**{k: float(d[k]) for k in cls.__dataclass_fields__ if k in d})

    def as_dict(self) -> Dict[str, float]:
        return {
            "w_z": self.w_z,
            "w_p": self.w_p,
            "w_u": self.w_u,
            "w_theta": self.w_theta,
            "w_q": self.w_q,
            "w_tau": self.w_tau,
        }


@dataclass
class MPCConfig:
    horizon: int = 10
    dt: float = 0.02
    mass: float = 35.0
    gravity: float = 9.81
    weights: MPCWeights = field(default_factory=MPCWeights)
    num_samples: int = 64
    noise_std: float = 5.0
    ilqg_iters: int = 5
    ilqg_reg: float = 1e-4

    @classmethod
    def from_dict(cls, d: Mapping) -> "MPCConfig":
        w = MPCWeights.from_dict(d.get("weights", {}))
        ps = d.get("predictive_sampling", {})
        ilqg = d.get("ilqg", {})
        return cls(
            horizon=int(d.get("horizon", 10)),
            dt=float(d.get("dt", 0.02)),
            mass=float(d.get("mass", 35.0)),
            gravity=float(d.get("gravity", 9.81)),
            weights=w,
            num_samples=int(ps.get("num_samples", 64)),
            noise_std=float(ps.get("noise_std", 5.0)),
            ilqg_iters=int(ilqg.get("max_iters", 5)),
            ilqg_reg=float(ilqg.get("reg", 1e-4)),
        )


@dataclass
class MPCSolution:
    """MPC output for one control step."""

    tau: np.ndarray
    q_ref: np.ndarray
    cost: float = 0.0
    planner_name: str = ""


def quat_to_yaw(quat: np.ndarray) -> float:
    """Extract yaw from Unitree-style quaternion ``[w, x, y, z]``."""
    w, x, y, z = quat
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _stage_cost(
    state: RobotState,
    torso: TorsoReference,
    q_cpg: np.ndarray,
    tau: np.ndarray,
    u_ref: np.ndarray,
    weights: MPCWeights,
) -> float:
    """Scalar stage cost matching the plan cost formulation."""
    z_err = state.base_pos[2] - torso.position[2]
    p_err = state.base_pos - torso.position
    yaw = quat_to_yaw(state.base_quat)
    theta_err = yaw - torso.yaw
    # wrap
    theta_err = (theta_err + np.pi) % (2.0 * np.pi) - np.pi
    q_err = state.joint_pos - q_cpg
    u_err = tau - u_ref
    return float(
        weights.w_z * z_err**2
        + weights.w_p * np.dot(p_err, p_err)
        + weights.w_u * np.dot(u_err, u_err)
        + weights.w_theta * theta_err**2
        + weights.w_q * np.dot(q_err, q_err)
        + weights.w_tau * np.dot(tau, tau)
    )


def _pd_joint_torques(
    state: RobotState,
    q_des: np.ndarray,
    kp: float = 40.0,
    kd: float = 1.5,
) -> np.ndarray:
    """Simple joint-space PD used as a feed-forward baseline / u_ref.

    Only for **pure torque** leg mode. In hybrid PD mode the actuator bridge
    already applies kp/kd — do not add this again or the legs will jitter.
    """
    tau = kp * (q_des - state.joint_pos) - kd * state.joint_vel
    tau[12:] = 0.0  # upper body handled by actuator PD
    return tau


def _com_balance_correction(
    state: RobotState,
    torso: TorsoReference,
    mass: float,
    gravity: float,
    weights: MPCWeights,
) -> np.ndarray:
    """Map CoM tracking error into approximate ankle / hip pitch torques."""
    tau = np.zeros(NUM_MOTORS, dtype=np.float64)
    p_err = state.base_pos - torso.position
    v_err = state.base_lin_vel - torso.velocity
    # Virtual wrench on CoM (spring-damper). Scale gently — this is FF only.
    f = -0.1 * weights.w_p * p_err - 0.1 * weights.w_u * v_err
    hip_share = 0.5 * f[0]
    ankle_share = 0.25 * f[0]
    roll_share = 0.4 * f[1]
    knee_share = 0.3 * f[2]

    for base in (0, 6):  # left, right
        tau[base + 0] += hip_share
        tau[base + 1] += roll_share if base == 0 else -roll_share
        tau[base + 3] += knee_share
        tau[base + 4] += ankle_share
    return np.clip(tau, -20.0, 20.0)


class BaseMPCSolver(ABC):
    """Abstract MPC interface (MJPC-style swappable backends)."""

    name: str = "base"

    def __init__(self, config: MPCConfig) -> None:
        self.config = config

    def update_weights(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self.config.weights, key):
                setattr(self.config.weights, key, float(value))

    @abstractmethod
    def solve(
        self,
        state: RobotState,
        torso: TorsoReference,
        cpg: CPGOutput,
        *,
        include_joint_pd: bool = True,
    ) -> MPCSolution:
        """Compute optimal joint torques / position refs for one step.

        Parameters
        ----------
        include_joint_pd:
            If False, do not bake joint-space PD into ``tau`` (use when the
            actuator already runs hybrid PD — avoids double-PD jitter).
        """


class ConvexQP_MPC(BaseMPCSolver):
    """Single-rigid-body linear-quadratic tracker (fastest backend).

    Solves a small dense QP over the CoM + yaw error analytically
    (unconstrained LQ closed form) and maps the virtual wrench to joint
    torques, then blends with CPG joint PD.
    """

    name = "convex_qp"

    def solve(
        self,
        state: RobotState,
        torso: TorsoReference,
        cpg: CPGOutput,
        *,
        include_joint_pd: bool = True,
    ) -> MPCSolution:
        cfg = self.config
        w = cfg.weights
        u_ref = (
            _pd_joint_torques(state, cpg.q_cpg)
            if include_joint_pd
            else np.zeros(NUM_MOTORS, dtype=np.float64)
        )
        tau_bal = _com_balance_correction(state, torso, cfg.mass, cfg.gravity, w)

        wu = max(w.w_u, 1e-6)
        wt = max(w.w_tau, 1e-9)
        alpha = wu / (wu + wt) if include_joint_pd else 0.0
        tau = alpha * u_ref + tau_bal

        yaw = quat_to_yaw(state.base_quat)
        yaw_err = (yaw - torso.yaw + np.pi) % (2.0 * np.pi) - np.pi
        tau[12] += -w.w_theta * yaw_err

        cost = _stage_cost(state, torso, cpg.q_cpg, tau, u_ref, w)
        return MPCSolution(tau=tau, q_ref=cpg.q_cpg.copy(), cost=cost, planner_name=self.name)


class PredictiveSampling_MPC(BaseMPCSolver):
    """Monte Carlo predictive sampling over torque-perturbed rollouts."""

    name = "predictive_sampling"

    def __init__(self, config: MPCConfig, seed: int = 0) -> None:
        super().__init__(config)
        self._rng = np.random.default_rng(seed)

    def solve(
        self,
        state: RobotState,
        torso: TorsoReference,
        cpg: CPGOutput,
        *,
        include_joint_pd: bool = True,
    ) -> MPCSolution:
        cfg = self.config
        w = cfg.weights
        u_ref = (
            _pd_joint_torques(state, cpg.q_cpg)
            if include_joint_pd
            else np.zeros(NUM_MOTORS, dtype=np.float64)
        )
        tau_bal = _com_balance_correction(state, torso, cfg.mass, cfg.gravity, w)
        nominal = u_ref + tau_bal

        best_tau = nominal
        best_cost = np.inf

        noise = self._rng.normal(
            0.0, cfg.noise_std, size=(cfg.num_samples, LEG_DIM)
        )
        for i in range(cfg.num_samples):
            tau = nominal.copy()
            tau[:LEG_DIM] = nominal[:LEG_DIM] + noise[i]
            cost = _stage_cost(state, torso, cpg.q_cpg, tau, u_ref, w)
            pos = state.base_pos.copy()
            vel = state.base_lin_vel.copy()
            for _ in range(cfg.horizon):
                acc = -w.w_p * (pos - torso.position) / max(cfg.mass, 1e-3)
                acc[0] += 0.01 * (tau[0] + tau[6])
                pos = pos + vel * cfg.dt
                vel = vel + acc * cfg.dt
                cost += w.w_p * np.sum((pos - torso.position) ** 2)
                cost += w.w_z * (pos[2] - torso.position[2]) ** 2
            cost += w.w_tau * float(np.dot(tau, tau))
            if cost < best_cost:
                best_cost = cost
                best_tau = tau

        return MPCSolution(
            tau=best_tau,
            q_ref=cpg.q_cpg.copy(),
            cost=float(best_cost),
            planner_name=self.name,
        )


class iLQG_MPC(BaseMPCSolver):
    """Iterative LQR / DDP-style refinement around the CPG nominal."""

    name = "ilqg"
    _TAU_CLIP = 80.0  # Nm — soft safety for G1 hip/knee class motors

    def solve(
        self,
        state: RobotState,
        torso: TorsoReference,
        cpg: CPGOutput,
        *,
        include_joint_pd: bool = True,
    ) -> MPCSolution:
        cfg = self.config
        w = cfg.weights
        u_nom = _com_balance_correction(state, torso, cfg.mass, cfg.gravity, w)
        if include_joint_pd:
            u_nom = u_nom + _pd_joint_torques(state, cpg.q_cpg)
        u = u_nom.copy()
        u_ref = u_nom.copy()

        eps = 1e-2
        hess = 2.0 * (max(w.w_u, 0.0) + max(w.w_tau, 0.0)) + cfg.ilqg_reg
        for _ in range(cfg.ilqg_iters):
            base_cost = _stage_cost(state, torso, cpg.q_cpg, u, u_ref, w)
            grad = np.zeros(LEG_DIM, dtype=np.float64)
            for j in range(LEG_DIM):
                u_pert = u.copy()
                u_pert[j] += eps
                c_plus = _stage_cost(state, torso, cpg.q_cpg, u_pert, u_ref, w)
                u_pert[j] = u[j] - eps
                c_minus = _stage_cost(state, torso, cpg.q_cpg, u_pert, u_ref, w)
                grad[j] = (c_plus - c_minus) / (2.0 * eps)

            step = grad / max(hess, 1e-6)
            step_scale = 1.0
            accepted = False
            for _bt in range(6):
                u_trial = u.copy()
                u_trial[:LEG_DIM] = np.clip(
                    u[:LEG_DIM] - step_scale * step,
                    -self._TAU_CLIP,
                    self._TAU_CLIP,
                )
                new_cost = _stage_cost(state, torso, cpg.q_cpg, u_trial, u_ref, w)
                if new_cost <= base_cost:
                    u = u_trial
                    accepted = True
                    break
                step_scale *= 0.5
            if not accepted:
                break

        u[:LEG_DIM] = np.clip(u[:LEG_DIM], -self._TAU_CLIP, self._TAU_CLIP)
        cost = _stage_cost(state, torso, cpg.q_cpg, u, u_ref, w)
        return MPCSolution(tau=u, q_ref=cpg.q_cpg.copy(), cost=cost, planner_name=self.name)


PLANNER_REGISTRY: Dict[str, Type[BaseMPCSolver]] = {
    "convex_qp": ConvexQP_MPC,
    "predictive_sampling": PredictiveSampling_MPC,
    "ilqg": iLQG_MPC,
    # UI display names
    "Convex QP": ConvexQP_MPC,
    "Predictive Sampling": PredictiveSampling_MPC,
    "iLQG": iLQG_MPC,
}


class MPCPlanner:
    """Facade that owns the active :class:`BaseMPCSolver` backend."""

    def __init__(self, config: MPCConfig, planner_name: str = "convex_qp") -> None:
        self.config = config
        self._solver: BaseMPCSolver = self._make(planner_name)

    def _make(self, name: str) -> BaseMPCSolver:
        key = name.strip()
        cls = PLANNER_REGISTRY.get(key) or PLANNER_REGISTRY.get(key.lower().replace(" ", "_"))
        if cls is None:
            raise ValueError(f"Unknown MPC planner '{name}'. Choose from {list(PLANNER_REGISTRY)}")
        return cls(self.config)

    @property
    def active_name(self) -> str:
        return self._solver.name

    def set_planner(self, name: str) -> None:
        if name == self._solver.name or PLANNER_REGISTRY.get(name) is type(self._solver):
            # Still rebuild if display name maps to same class — keep simple.
            pass
        self._solver = self._make(name)

    def update_weights(self, **kwargs) -> None:
        self._solver.update_weights(**kwargs)

    def solve(
        self,
        state: RobotState,
        torso: TorsoReference,
        cpg: CPGOutput,
        *,
        include_joint_pd: bool = True,
    ) -> MPCSolution:
        return self._solver.solve(
            state, torso, cpg, include_joint_pd=include_joint_pd
        )
