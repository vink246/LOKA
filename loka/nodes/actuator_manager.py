"""Mixed actuation manager for Unitree G1 (29 DoF)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from loka.utils.dds_interface import DDSInterface

NUM_MOTORS = 29
LEG_INDICES = tuple(range(0, 12))
WAIST_INDICES = tuple(range(12, 15))
ARM_INDICES = tuple(range(15, 29))

# Unitree G1 low-level example gains (stable for position hold).
DEFAULT_KP = np.array(
    [
        60, 60, 60, 100, 40, 40,
        60, 60, 60, 100, 40, 40,
        60, 40, 40,
        40, 40, 40, 40, 40, 40, 40,
        40, 40, 40, 40, 40, 40, 40,
    ],
    dtype=np.float64,
)
DEFAULT_KD = np.array(
    [
        1, 1, 1, 2, 1, 1,
        1, 1, 1, 2, 1, 1,
        1, 1, 1,
        1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1,
    ],
    dtype=np.float64,
)


@dataclass
class ActuatorConfig:
    """Joint modes and PD parameters for mixed control."""

    motor_mode: int = 1  # Unitree HG: 1 = enable
    default_q: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_MOTORS, dtype=np.float64)
    )
    kp: np.ndarray = field(default_factory=lambda: DEFAULT_KP.copy())
    kd: np.ndarray = field(default_factory=lambda: DEFAULT_KD.copy())
    freeze_arms: bool = True
    freeze_waist: bool = True
    enable_leg_torque: bool = False

    @classmethod
    def from_dicts(
        cls,
        g1: Mapping,
        ctrl: Mapping,
        default_q: Optional[Sequence[float]] = None,
    ) -> "ActuatorConfig":
        act = ctrl.get("actuator", {})
        q = np.array(
            default_q
            if default_q is not None
            else g1.get("default_joint_pos", [0.0] * NUM_MOTORS),
            dtype=np.float64,
        )
        if q.shape != (NUM_MOTORS,):
            raise ValueError(f"default_q must have shape ({NUM_MOTORS},)")

        kp = DEFAULT_KP.copy()
        kd = DEFAULT_KD.copy()
        # Optional uniform overrides from YAML (legacy scalar fields).
        leg_pd = g1.get("leg_pd", {})
        upper_pd = g1.get("upper_body_pd", {})
        if "kp" in leg_pd:
            kp[list(LEG_INDICES)] = float(leg_pd["kp"])
        if "kd" in leg_pd:
            kd[list(LEG_INDICES)] = float(leg_pd["kd"])
        if "kp" in upper_pd:
            kp[12:] = float(upper_pd["kp"])
        if "kd" in upper_pd:
            kd[12:] = float(upper_pd["kd"])
        if "kp" in g1:
            kp = np.array(g1["kp"], dtype=np.float64)
        if "kd" in g1:
            kd = np.array(g1["kd"], dtype=np.float64)

        return cls(
            motor_mode=int(g1.get("motor_mode", 1)),
            default_q=q,
            kp=kp,
            kd=kd,
            freeze_arms=bool(act.get("freeze_arms", True)),
            freeze_waist=bool(act.get("freeze_waist", True)),
            enable_leg_torque=bool(act.get("enable_leg_torque", False)),
        )


@dataclass
class MotorCommand:
    """Fully specified 29-motor LowCmd payload fields."""

    mode: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau: np.ndarray


class SoftStarter:
    """Interpolate joint targets from ``q_start`` to ``q_goal`` over ``duration``."""

    def __init__(self, duration: float = 2.0) -> None:
        self.duration = max(float(duration), 1e-3)
        self._t0: Optional[float] = None
        self._q0 = np.zeros(NUM_MOTORS, dtype=np.float64)
        self._q1 = np.zeros(NUM_MOTORS, dtype=np.float64)
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def begin(self, t: float, q_start: Sequence[float], q_goal: Sequence[float]) -> None:
        self._t0 = t
        self._q0 = np.asarray(q_start, dtype=np.float64).copy()
        self._q1 = np.asarray(q_goal, dtype=np.float64).copy()
        self._active = True

    def cancel(self) -> None:
        self._active = False

    def evaluate(self, t: float) -> np.ndarray:
        if not self._active or self._t0 is None:
            return self._q1.copy()
        ratio = float(np.clip((t - self._t0) / self.duration, 0.0, 1.0))
        # Smoothstep to avoid jerk at the endpoints.
        s = ratio * ratio * (3.0 - 2.0 * ratio)
        q = (1.0 - s) * self._q0 + s * self._q1
        if ratio >= 1.0:
            self._active = False
        return q


def quat_to_rpy(quat: np.ndarray) -> np.ndarray:
    """Convert Unitree ``[w, x, y, z]`` quaternion to roll, pitch, yaw."""
    w, x, y, z = [float(v) for v in quat]
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)

    sinp = 2.0 * (w * y - z * x)
    sinp = float(np.clip(sinp, -1.0, 1.0))
    pitch = np.arcsin(sinp)

    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)
    return np.array([roll, pitch, yaw], dtype=np.float64)


class ActuatorManager:
    """Map MPC / CPG outputs onto mixed torque and position motor commands.

    Legs (0–11)
        Torque mode (``enable_leg_torque``): ``Kp=Kd=0``, ``tau = tau_mpc``.
        Hybrid mode: PD about ``q_ref`` plus feed-forward ``tau`` only
        (``tau`` must NOT already contain joint PD — that causes double-PD jitter).

    Upper body (12–28)
        Locked in position/PD mode toward ``default_q`` (or live refs).
    """

    def __init__(self, config: ActuatorConfig, dds: Optional["DDSInterface"] = None) -> None:
        self.config = config
        self.dds = dds
        self.leg_mask = np.zeros(NUM_MOTORS, dtype=bool)
        self.waist_mask = np.zeros(NUM_MOTORS, dtype=bool)
        self.arm_mask = np.zeros(NUM_MOTORS, dtype=bool)
        self.leg_mask[list(LEG_INDICES)] = True
        self.waist_mask[list(WAIST_INDICES)] = True
        self.arm_mask[list(ARM_INDICES)] = True

    def set_freeze_arms(self, value: bool) -> None:
        self.config.freeze_arms = bool(value)

    def set_freeze_waist(self, value: bool) -> None:
        self.config.freeze_waist = bool(value)

    def set_enable_leg_torque(self, value: bool) -> None:
        self.config.enable_leg_torque = bool(value)

    def build_command(
        self,
        *,
        tau_ff: Sequence[float],
        q_ref: Optional[Sequence[float]] = None,
        dq_ref: Optional[Sequence[float]] = None,
    ) -> MotorCommand:
        """Compose mixed-mode motor command.

        Parameters
        ----------
        tau_ff:
            Feed-forward torque only (gravity / balance / MPC). Do **not** pass
            joint-space PD here when hybrid mode is enabled — the bridge already
            applies ``kp/kd``.
        """
        cfg = self.config
        tau = np.asarray(tau_ff, dtype=np.float64)
        if tau.shape != (NUM_MOTORS,):
            raise ValueError(f"tau_ff must have shape ({NUM_MOTORS},)")

        q = cfg.default_q.copy() if q_ref is None else np.asarray(q_ref, dtype=np.float64)
        dq = np.zeros(NUM_MOTORS) if dq_ref is None else np.asarray(dq_ref, dtype=np.float64)
        if q.shape != (NUM_MOTORS,) or dq.shape != (NUM_MOTORS,):
            raise ValueError("q_ref and dq_ref must have shape (29,)")

        mode = np.full(NUM_MOTORS, cfg.motor_mode, dtype=np.int32)
        kp = cfg.kp.copy()
        kd = cfg.kd.copy()
        q_cmd = q.copy()
        dq_cmd = dq.copy()
        tau_cmd = tau.copy()

        if cfg.enable_leg_torque:
            kp[self.leg_mask] = 0.0
            kd[self.leg_mask] = 0.0
            q_cmd[self.leg_mask] = 0.0
            dq_cmd[self.leg_mask] = 0.0
        # else: keep per-joint kp/kd; tau_cmd is feed-forward only

        tau_cmd[self.waist_mask] = 0.0
        tau_cmd[self.arm_mask] = 0.0
        if cfg.freeze_waist:
            q_cmd[self.waist_mask] = cfg.default_q[self.waist_mask]
            dq_cmd[self.waist_mask] = 0.0
        if cfg.freeze_arms:
            q_cmd[self.arm_mask] = cfg.default_q[self.arm_mask]
            dq_cmd[self.arm_mask] = 0.0

        return MotorCommand(mode=mode, q=q_cmd, dq=dq_cmd, kp=kp, kd=kd, tau=tau_cmd)

    def send(
        self,
        *,
        tau_ff: Sequence[float],
        q_ref: Optional[Sequence[float]] = None,
        dq_ref: Optional[Sequence[float]] = None,
    ) -> MotorCommand:
        """Build command and publish via DDS if attached."""
        cmd = self.build_command(tau_ff=tau_ff, q_ref=q_ref, dq_ref=dq_ref)
        if self.dds is not None:
            self.dds.publish_lowcmd(
                mode=cmd.mode,
                q=cmd.q,
                dq=cmd.dq,
                kp=cmd.kp,
                kd=cmd.kd,
                tau=cmd.tau,
            )
        return cmd

    def emergency_stop(self) -> None:
        """Zero all torques and PD gains."""
        if self.dds is not None:
            self.dds.publish_zero(motor_mode=self.config.motor_mode)
