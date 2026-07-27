"""Analytical CPG phase generator and 6-DoF leg inverse kinematics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Tuple

import numpy as np

NUM_MOTORS = 29
LEFT_LEG = slice(0, 6)
RIGHT_LEG = slice(6, 12)


@dataclass
class CPGParams:
    """Analytical CPG gait parameters."""

    stride_period: float = 0.8
    sweep_amplitude: float = 0.0
    clearance: float = 0.0
    duty_factor: float = 0.6
    phase_left: float = 0.0
    phase_right: float = 0.5
    stance_x: float = 0.0
    stance_y_left: float = 0.1185
    stance_y_right: float = -0.1185
    stance_z: float = -0.743

    @classmethod
    def from_dict(cls, d: Mapping) -> "CPGParams":
        return cls(
            stride_period=float(d.get("stride_period", 0.8)),
            sweep_amplitude=float(d.get("sweep_amplitude", 0.0)),
            clearance=float(d.get("clearance", 0.0)),
            duty_factor=float(d.get("duty_factor", 0.6)),
            phase_left=float(d.get("phase_left", 0.0)),
            phase_right=float(d.get("phase_right", 0.5)),
            stance_x=float(d.get("stance_x", 0.0)),
            stance_y_left=float(d.get("stance_y_left", 0.1185)),
            stance_y_right=float(d.get("stance_y_right", -0.1185)),
            stance_z=float(d.get("stance_z", -0.743)),
        )

    @property
    def is_stand_still(self) -> bool:
        """True when the CPG should hold a static double-support pose."""
        return abs(self.sweep_amplitude) < 1e-4 and abs(self.clearance) < 1e-4


@dataclass
class LegKinematics:
    """Approximate serial-chain lengths for analytical leg IK (MuJoCo-calibrated)."""

    hip_lateral_offset: float = 0.064452
    hip_vertical_offset: float = 0.1027
    thigh_length: float = 0.341
    shank_length: float = 0.317
    foot_height: float = 0.04
    hip_pitch_bias: float = 0.0

    @classmethod
    def from_dict(cls, d: Mapping) -> "LegKinematics":
        return cls(**{k: float(d[k]) for k in cls.__dataclass_fields__ if k in d})


@dataclass
class CPGOutput:
    """CPG evaluation result for one control step."""

    q_cpg: np.ndarray
    foot_pos_left: np.ndarray
    foot_pos_right: np.ndarray
    foot_contacts: np.ndarray
    phase_left: float
    phase_right: float


def _cycloid_swing(phi_swing: float, a_s: float, h: float) -> Tuple[float, float]:
    """Cycloid swing trajectory with zero acceleration at liftoff / touchdown."""
    x = a_s * (phi_swing - (1.0 / (2.0 * np.pi)) * np.sin(2.0 * np.pi * phi_swing)) - (
        a_s / 2.0
    )
    z = h * (1.0 - np.cos(2.0 * np.pi * phi_swing)) / 2.0
    return float(x), float(z)


def _stance_retract(phi_stance: float, a_s: float) -> Tuple[float, float]:
    """Linear sweep retraction during stance; clearance stays zero."""
    x = (a_s / 2.0) - a_s * phi_stance
    return float(x), 0.0


def foot_cartesian(
    phi: float,
    params: CPGParams,
    *,
    y_stance: float,
) -> Tuple[np.ndarray, bool]:
    """Map local phase ``phi ∈ [0, 1)`` to pelvis-frame foot position."""
    if params.is_stand_still:
        pos = np.array(
            [params.stance_x, y_stance, params.stance_z],
            dtype=np.float64,
        )
        return pos, True

    d = float(np.clip(params.duty_factor, 1e-3, 1.0 - 1e-3))
    swing_frac = 1.0 - d
    if phi < swing_frac:
        phi_swing = phi / swing_frac
        dx, dz = _cycloid_swing(phi_swing, params.sweep_amplitude, params.clearance)
        in_stance = False
    else:
        phi_stance = (phi - swing_frac) / d
        dx, dz = _stance_retract(phi_stance, params.sweep_amplitude)
        in_stance = True

    pos = np.array(
        [params.stance_x + dx, y_stance, params.stance_z + dz],
        dtype=np.float64,
    )
    return pos, in_stance


def leg_ik(
    foot_pos: np.ndarray,
    kin: LegKinematics,
    *,
    side: str,
) -> np.ndarray:
    """Analytical 6-DoF inverse kinematics for one G1 leg.

    Uses a numerically stable biped decomposition:
    - ``hip_yaw = 0`` (feet stay parallel; avoids atan2 singularity when x≈0)
    - ``hip_roll`` absorbs lateral foot placement in the frontal plane
    - ``hip_pitch`` + ``knee`` solve the 2-link sagittal reach
    - ankles keep the sole approximately flat

    Note: the real G1 chain has fixed link offsets/quaternions, so this IK is
    approximate. Prefer ``default_q`` for standing; use IK for gait deltas.
    """
    sign = 1.0 if side == "left" else -1.0
    hip = np.array(
        [0.0, sign * kin.hip_lateral_offset, -kin.hip_vertical_offset],
        dtype=np.float64,
    )
    d = np.asarray(foot_pos, dtype=np.float64) - hip
    d[2] += kin.foot_height
    x, y, z = float(d[0]), float(d[1]), float(d[2])

    # Yaw held at 0 — atan2(y, x) is singular for nearly-vertical stance legs.
    hip_yaw = 0.0
    hip_roll = float(np.clip(np.arctan2(y, -z + 1e-8), -0.6, 0.6))

    # Rotate about +X by -hip_roll to bring the foot into the sagittal plane.
    cr, sr = np.cos(hip_roll), np.sin(hip_roll)
    x2 = x
    z2 = -sr * y + cr * z

    L1, L2 = kin.thigh_length, kin.shank_length
    reach = float(np.clip(np.hypot(x2, z2), 1e-4, L1 + L2 - 1e-4))
    cos_knee = float(np.clip((L1**2 + L2**2 - reach**2) / (2.0 * L1 * L2), -1.0, 1.0))
    knee = np.pi - float(np.arccos(cos_knee))

    alpha = float(np.arctan2(x2, -z2))
    beta = float(
        np.arccos(
            np.clip((L1**2 + reach**2 - L2**2) / (2.0 * L1 * reach), -1.0, 1.0)
        )
    )
    hip_pitch = alpha - beta - kin.hip_pitch_bias
    ankle_pitch = -(hip_pitch + knee)
    ankle_roll = -hip_roll

    # Right-leg roll joints use a mirrored range / axis convention.
    if side == "right":
        hip_roll = -hip_roll
        ankle_roll = -ankle_roll

    return np.array(
        [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll],
        dtype=np.float64,
    )


@dataclass
class CPGGenerator:
    """Analytical CPG that exposes ``q_CPG(t)`` and foot contact states."""

    params: CPGParams = field(default_factory=CPGParams)
    kinematics: LegKinematics = field(default_factory=LegKinematics)
    default_q: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_MOTORS, dtype=np.float64)
    )
    _t0: float = 0.0
    _running: bool = False

    def start(self, t: float) -> None:
        self._t0 = t
        self._running = True

    def stop(self) -> None:
        self._running = False

    def reset(self, t: float) -> None:
        self._t0 = t

    def update_params(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)

    def evaluate(self, t: float) -> CPGOutput:
        """Evaluate CPG phase, foot Cartesian targets, and joint-space refs.

        Stand-still (``A_S≈0`` and ``H≈0``) returns the calibrated ``default_q``
        in double support — do **not** run approximate IK, which folds the legs.
        """
        q = self.default_q.copy()
        foot_l = np.array(
            [self.params.stance_x, self.params.stance_y_left, self.params.stance_z]
        )
        foot_r = np.array(
            [self.params.stance_x, self.params.stance_y_right, self.params.stance_z]
        )

        if not self._running:
            return CPGOutput(
                q_cpg=q,
                foot_pos_left=foot_l,
                foot_pos_right=foot_r,
                foot_contacts=np.array([1.0, 1.0]),
                phase_left=0.0,
                phase_right=0.0,
            )

        # Static double-support hold (stand under own weight).
        if self.params.is_stand_still:
            return CPGOutput(
                q_cpg=q,
                foot_pos_left=foot_l,
                foot_pos_right=foot_r,
                foot_contacts=np.array([1.0, 1.0]),
                phase_left=0.0,
                phase_right=0.0,
            )

        T = max(float(self.params.stride_period), 1e-3)
        tau = (t - self._t0) / T
        phi_l = (tau + self.params.phase_left) % 1.0
        phi_r = (tau + self.params.phase_right) % 1.0

        foot_l, stance_l = foot_cartesian(
            phi_l, self.params, y_stance=self.params.stance_y_left
        )
        foot_r, stance_r = foot_cartesian(
            phi_r, self.params, y_stance=self.params.stance_y_right
        )

        q[LEFT_LEG] = leg_ik(foot_l, self.kinematics, side="left")
        q[RIGHT_LEG] = leg_ik(foot_r, self.kinematics, side="right")

        contacts = np.array(
            [1.0 if stance_l else 0.0, 1.0 if stance_r else 0.0],
            dtype=np.float64,
        )
        return CPGOutput(
            q_cpg=q,
            foot_pos_left=foot_l,
            foot_pos_right=foot_r,
            foot_contacts=contacts,
            phase_left=phi_l,
            phase_right=phi_r,
        )
