"""Simple position-space stand / torso hold (no torque MPC).

The actuator PD loop converts ``q_ref`` into motor torques. This module only
chooses joint *positions* that keep the torso upright at a target height.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from loka.nodes.actuator_manager import quat_to_rpy
from loka.nodes.state_estimator import RobotState


@dataclass
class StandGains:
    """Joint-space corrections from torso attitude / height error."""

    k_hip_pitch: float = 0.6
    k_ankle_pitch: float = 0.8
    k_hip_roll: float = 0.4
    k_ankle_roll: float = 0.4
    k_knee_height: float = 0.5
    max_joint_delta: float = 0.35

    @classmethod
    def from_dict(cls, d: Mapping) -> "StandGains":
        return cls(
            k_hip_pitch=float(d.get("k_hip_pitch", 0.6)),
            k_ankle_pitch=float(d.get("k_ankle_pitch", 0.8)),
            k_hip_roll=float(d.get("k_hip_roll", 0.4)),
            k_ankle_roll=float(d.get("k_ankle_roll", 0.4)),
            k_knee_height=float(d.get("k_knee_height", 0.5)),
            max_joint_delta=float(
                d.get("max_joint_delta", d.get("max_correction", 0.35))
            ),
        )


class TorsoStandController:
    """Hold a standing joint pose and nudge it to keep the torso upright."""

    def __init__(
        self,
        q_stand: np.ndarray,
        z_target: float = 0.793,
        gains: Optional[StandGains] = None,
    ) -> None:
        self.q_stand = np.asarray(q_stand, dtype=np.float64).copy()
        self.z_target = float(z_target)
        self.gains = gains or StandGains()

    def update_z_target(self, z: float) -> None:
        self.z_target = float(z)

    def compute_q_ref(self, state: RobotState) -> np.ndarray:
        """Return joint position targets for the actuator PD controller."""
        q = self.q_stand.copy()
        g = self.gains
        lim = g.max_joint_delta

        roll, pitch, _yaw = quat_to_rpy(state.base_quat)

        d_hip_p = float(np.clip(-g.k_hip_pitch * pitch, -lim, lim))
        d_ank_p = float(np.clip(-g.k_ankle_pitch * pitch, -lim, lim))
        d_hip_r = float(np.clip(-g.k_hip_roll * roll, -lim, lim))
        d_ank_r = float(np.clip(-g.k_ankle_roll * roll, -lim, lim))

        z_err = float(state.base_pos[2]) - self.z_target
        d_knee = float(np.clip(g.k_knee_height * z_err, -lim, lim))

        for base in (0, 6):
            q[base + 0] += d_hip_p
            q[base + 1] += d_hip_r if base == 0 else -d_hip_r
            q[base + 3] += d_knee
            q[base + 4] += d_ank_p
            q[base + 5] += d_ank_r if base == 0 else -d_ank_r

        return q
