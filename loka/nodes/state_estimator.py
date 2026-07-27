"""Ground-truth and abstract state estimation for the Unitree G1."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from loka.utils.dds_interface import DDSInterface

NUM_MOTORS = 29


@dataclass
class RobotState:
    """Clean robot state snapshot exposed to the controller loop."""

    base_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    base_lin_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_ang_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    joint_pos: np.ndarray = field(default_factory=lambda: np.zeros(NUM_MOTORS))
    joint_vel: np.ndarray = field(default_factory=lambda: np.zeros(NUM_MOTORS))
    foot_contacts: np.ndarray = field(default_factory=lambda: np.ones(2))
    timestamp: float = 0.0

    def copy(self) -> "RobotState":
        return RobotState(
            base_pos=self.base_pos.copy(),
            base_quat=self.base_quat.copy(),
            base_lin_vel=self.base_lin_vel.copy(),
            base_ang_vel=self.base_ang_vel.copy(),
            joint_pos=self.joint_pos.copy(),
            joint_vel=self.joint_vel.copy(),
            foot_contacts=self.foot_contacts.copy(),
            timestamp=self.timestamp,
        )


class BaseStateEstimator(ABC):
    """Abstract state estimator interface."""

    @abstractmethod
    def get_state(self) -> RobotState:
        """Return a thread-safe snapshot of the latest robot state."""

    @abstractmethod
    def start(self) -> None:
        """Begin receiving sensor / DDS updates."""

    @abstractmethod
    def stop(self) -> None:
        """Stop updates (best-effort)."""


class GroundTruthStateEstimator(BaseStateEstimator):
    """DDS ground-truth estimator using SportModeState + LowState.

    - ``rt/sportmodestate`` → base position and linear velocity (sim GT).
    - ``rt/lowstate`` (unitree_hg) → 29 joint positions / velocities and IMU
      (quaternion + body angular velocity). SportModeState IMU is used when
      populated; otherwise LowState IMU fills orientation / ang-vel.
    """

    def __init__(self, dds: "DDSInterface") -> None:
        self._dds = dds
        self._lock = threading.Lock()
        self._state = RobotState()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._dds.subscribe_sportmodestate(self._on_sport)
        self._dds.subscribe_lowstate(self._on_lowstate)
        self._started = True

    def stop(self) -> None:
        self._started = False

    def get_state(self) -> RobotState:
        with self._lock:
            return self._state.copy()

    def _on_sport(self, msg) -> None:
        with self._lock:
            self._state.base_pos = np.array(msg.position, dtype=np.float64)
            self._state.base_lin_vel = np.array(msg.velocity, dtype=np.float64)
            # Prefer sport IMU when non-trivial; sim often leaves it empty.
            quat = np.array(msg.imu_state.quaternion, dtype=np.float64)
            if np.linalg.norm(quat) > 0.5:
                self._state.base_quat = quat
                self._state.base_ang_vel = np.array(
                    msg.imu_state.gyroscope, dtype=np.float64
                )
            # Continuous contact proxy from foot_force when available (quad layout;
            # use front-left / front-right slots as left / right for G1 sim).
            try:
                ff = np.array(msg.foot_force, dtype=np.float64)
                if ff.shape[0] >= 2:
                    self._state.foot_contacts = (ff[:2] > 10.0).astype(np.float64)
            except Exception:
                pass

    def _on_lowstate(self, msg) -> None:
        q = np.zeros(NUM_MOTORS, dtype=np.float64)
        dq = np.zeros(NUM_MOTORS, dtype=np.float64)
        for i in range(NUM_MOTORS):
            q[i] = float(msg.motor_state[i].q)
            dq[i] = float(msg.motor_state[i].dq)

        quat = np.array(msg.imu_state.quaternion, dtype=np.float64)
        gyro = np.array(msg.imu_state.gyroscope, dtype=np.float64)

        with self._lock:
            self._state.joint_pos = q
            self._state.joint_vel = dq
            if np.linalg.norm(quat) > 0.5:
                self._state.base_quat = quat
                self._state.base_ang_vel = gyro
            try:
                self._state.timestamp = float(msg.tick) * 1e-3
            except Exception:
                pass
