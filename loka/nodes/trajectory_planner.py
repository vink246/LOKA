"""3D time-parameterized torso / CoM path generator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional

import numpy as np


class PathType(str, Enum):
    STRAIGHT = "straight"
    CIRCLE = "circle"
    SINE = "sine"


@dataclass
class TorsoTrajectoryParams:
    """Parameters for the torso reference path ``p_ref(t)``."""

    path_type: PathType = PathType.STRAIGHT
    v_ref: float = 0.3
    y_offset: float = 0.0
    z_target: float = 0.78
    theta_ref: float = 0.0
    path_radius: float = 1.0
    omega: float = 0.5
    sine_amplitude: float = 0.2

    @classmethod
    def from_dict(cls, d: Mapping) -> "TorsoTrajectoryParams":
        raw = str(d.get("path_type", "straight")).lower()
        try:
            path_type = PathType(raw)
        except ValueError:
            path_type = PathType.STRAIGHT
        return cls(
            path_type=path_type,
            v_ref=float(d.get("v_ref", 0.3)),
            y_offset=float(d.get("y_offset", 0.0)),
            z_target=float(d.get("z_target", 0.78)),
            theta_ref=float(d.get("theta_ref", 0.0)),
            path_radius=float(d.get("path_radius", 1.0)),
            omega=float(d.get("omega", 0.5)),
            sine_amplitude=float(d.get("sine_amplitude", 0.2)),
        )


@dataclass
class TorsoReference:
    """Torso pose / velocity reference at a single time."""

    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    yaw: float
    yaw_rate: float


class TorsoTrajectoryPlanner:
    """Time-parameterized torso path: straight, circular arc, or harmonic sine."""

    def __init__(self, params: Optional[TorsoTrajectoryParams] = None) -> None:
        self.params = params or TorsoTrajectoryParams()
        self._origin = np.zeros(3, dtype=np.float64)
        self._t0 = 0.0
        self._yaw0 = 0.0
        self._initialized = False

    def reset(self, t: float, origin: np.ndarray, yaw: float = 0.0) -> None:
        """Anchor the path at the current robot pose."""
        self._t0 = t
        self._origin = np.asarray(origin, dtype=np.float64).copy()
        self._yaw0 = float(yaw)
        self._initialized = True

    def update_params(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if key == "path_type":
                self.params.path_type = PathType(str(value).lower())
            elif hasattr(self.params, key):
                setattr(self.params, key, value)

    def evaluate(self, t: float) -> TorsoReference:
        """Evaluate ``p_ref(t) = [x_ref, y_ref, z_ref]`` and yaw."""
        p = self.params
        if not self._initialized:
            return TorsoReference(
                position=np.array([0.0, p.y_offset, p.z_target]),
                velocity=np.zeros(3),
                yaw=p.theta_ref,
                yaw_rate=0.0,
            )

        dt = max(t - self._t0, 0.0)
        z = p.z_target
        vz = 0.0

        if p.path_type == PathType.CIRCLE:
            R = max(abs(p.path_radius), 1e-3)
            # Arc-length speed → angular rate about circle center.
            omega_path = p.v_ref / R
            ang = self._yaw0 + omega_path * dt
            cx = self._origin[0]
            cy = self._origin[1] - R  # center to the left of start heading +x
            x = cx + R * np.sin(ang - self._yaw0)
            y = cy + R * (1.0 - np.cos(ang - self._yaw0)) + p.y_offset
            # Rotate start frame if yaw0 != 0
            c0, s0 = np.cos(self._yaw0), np.sin(self._yaw0)
            xr = c0 * (x - self._origin[0]) - s0 * (y - self._origin[1]) + self._origin[0]
            yr = s0 * (x - self._origin[0]) + c0 * (y - self._origin[1]) + self._origin[1]
            vx = p.v_ref * np.cos(ang)
            vy = p.v_ref * np.sin(ang)
            yaw = ang
            yaw_rate = omega_path
            pos = np.array([xr, yr, z], dtype=np.float64)
            vel = np.array([vx, vy, vz], dtype=np.float64)
        elif p.path_type == PathType.SINE:
            # Forward along body x, lateral harmonic in y.
            s = p.v_ref * dt
            c0, s0 = np.cos(self._yaw0), np.sin(self._yaw0)
            y_local = p.y_offset + p.sine_amplitude * np.sin(p.omega * dt)
            vy_local = p.sine_amplitude * p.omega * np.cos(p.omega * dt)
            x = self._origin[0] + c0 * s - s0 * y_local
            y = self._origin[1] + s0 * s + c0 * y_local
            vx = c0 * p.v_ref - s0 * vy_local
            vy = s0 * p.v_ref + c0 * vy_local
            yaw = self._yaw0 + p.theta_ref
            yaw_rate = 0.0
            pos = np.array([x, y, z], dtype=np.float64)
            vel = np.array([vx, vy, vz], dtype=np.float64)
        else:
            # Straight line along heading (yaw0 + theta_ref).
            yaw = self._yaw0 + p.theta_ref
            c, s = np.cos(yaw), np.sin(yaw)
            dist = p.v_ref * dt
            x = self._origin[0] + c * dist - s * p.y_offset
            y = self._origin[1] + s * dist + c * p.y_offset
            vx = c * p.v_ref
            vy = s * p.v_ref
            yaw_rate = 0.0
            pos = np.array([x, y, z], dtype=np.float64)
            vel = np.array([vx, vy, vz], dtype=np.float64)

        return TorsoReference(position=pos, velocity=vel, yaw=yaw, yaw_rate=yaw_rate)
