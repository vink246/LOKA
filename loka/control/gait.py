"""Gait clock, Raibert footholds, and Bezier swing trajectories.

Phase B walking geometry stays classical (out of the LLM vocabulary).
The orchestrator only tunes high-level ``gait.*`` policy knobs; foothold XY and
per-tick contact bits are computed here.

Footstep placement is Raibert in the heading frame about a planned midline:
sagittal feedforward + velocity feedback, fixed ``±stance_width/2`` lateral.
Swing arcs are cubic Bézier in XY with a raised mid-swing Z — tracked in the
WBC via foot-center Jacobians (no separate analytical foot IK).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

GRAVITY = 9.81

# Mode codes for typed Task_Targets (floats) and string aliases.
MODE_STAND = 0.0
MODE_WALK = 1.0
MODE_TREAD = 2.0
MODE_LIMP = 3.0

MODE_NAME_TO_CODE = {
    "stand": MODE_STAND,
    "walk": MODE_WALK,
    "tread": MODE_TREAD,
    "limp": MODE_LIMP,
}
MODE_CODE_TO_NAME = {v: k for k, v in MODE_NAME_TO_CODE.items()}


class LegPhase(str, Enum):
    STANCE = "stance"
    SWING = "swing"


@dataclass
class GaitConfig:
    """High-level gait policy (LLM / operator facing)."""

    mode: float = MODE_STAND  # see MODE_* codes
    speed: float = 0.0  # commanded forward speed in heading frame [m/s]
    heading: float = 0.0  # world-frame travel direction [rad]
    step_period: float = 0.50  # full cycle period [s]
    duty_factor: float = 0.82  # short SS for lateral stability; swing ~0.09 s
    step_length_max: float = 0.08  # clamp on foothold offset from CoM [m]
    swing_height: float = 0.04  # peak clearance above ground [m]
    stance_width: float = 0.26  # slightly wide for SS balance while swinging
    capture_gain: float = 1.0  # scales 1/ω0 DCM velocity feedback (1 = full CP)
    #: Mild CoM bias toward the stance foot during single support [m].
    #: Large snaps tip the plant; ~2 cm helps when the swing foot unloads.
    ss_lateral_bias: float = 0.018
    #: Soften first steps after stand→walk.
    walk_accel: float = 0.08  # m/s^2 ramp on commanded speed
    #: Fraction of swing during which the foothold may be retargeted (DCM).
    foothold_retarget_s: float = 0.30


# Clamps applied when LLM / operator sets gait.* Task_Targets.
GAIT_LIMITS: dict[str, tuple[float, float]] = {
    "gait.mode": (0.0, 3.0),
    "gait.speed": (0.0, 0.6),
    "gait.heading": (-np.pi, np.pi),
    "gait.step_period": (0.30, 0.80),
    "gait.duty_factor": (0.55, 0.92),
    "gait.step_length_max": (0.06, 0.35),
    "gait.swing_height": (0.02, 0.12),
    "gait.stance_width": (0.10, 0.30),
    "gait.capture_gain": (0.0, 2.0),
    "gait.walk_accel": (0.05, 1.0),
    "gait.ss_lateral_bias": (0.0, 0.05),
    "gait.foothold_retarget_s": (0.05, 0.60),
}

GAIT_PARAMETER_NAMES = frozenset(GAIT_LIMITS.keys())


def parse_gait_mode(value) -> float:
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in MODE_NAME_TO_CODE:
            raise ValueError(f"Unknown gait.mode {value!r}")
        return MODE_NAME_TO_CODE[key]
    code = float(value)
    return float(np.clip(round(code), 0, 3))


def gait_mode_name(code: float) -> str:
    return MODE_CODE_TO_NAME.get(float(round(code)), "stand")


def _cubic_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, s: float) -> np.ndarray:
    u = 1.0 - s
    return u**3 * p0 + 3 * u**2 * s * p1 + 3 * u * s**2 * p2 + s**3 * p3


def _bezier_swing_pos(
    start: np.ndarray,
    end: np.ndarray,
    s: float,
    *,
    swing_height: float,
    ground_z: float,
) -> np.ndarray:
    """Foot-center position along a raised cubic Bézier, ``s ∈ [0, 1]``."""
    s = float(np.clip(s, 0.0, 1.0))
    mid = 0.5 * (start + end)
    # Control points lift mid-swing; endpoints stay near ground.
    p0 = start.copy()
    p3 = end.copy()
    p1 = start + 0.35 * (end - start)
    p2 = end - 0.35 * (end - start)
    p1[2] = ground_z + swing_height
    p2[2] = ground_z + swing_height
    mid_xy = _cubic_bezier(p0, p1, p2, p3, s)
    # Extra Z bump so clearance peaks near s=0.5 even if XY controls sag.
    z_bump = 4.0 * s * (1.0 - s) * swing_height
    mid_xy[2] = max(float(mid_xy[2]), ground_z + z_bump)
    return mid_xy


@dataclass
class SwingState:
    active: bool = False
    leg: int = 0  # 0=left, 1=right
    start: np.ndarray = field(default_factory=lambda: np.zeros(3))
    foothold: np.ndarray = field(default_factory=lambda: np.zeros(3))
    des_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    s: float = 0.0


@dataclass
class GaitOutput:
    """One control-tick gait decision."""

    contact_mask: np.ndarray  # (8,) bool planned
    com_ref_xy: np.ndarray  # (2,) world
    com_vel_cmd: np.ndarray  # (2,) world
    swing: SwingState
    phase: float
    left_phase: LegPhase
    right_phase: LegPhase


class GaitScheduler:
    """Periodic bipedal gait with CP footholds and Bézier swing."""

    def __init__(self, config: GaitConfig | None = None) -> None:
        self.config = config or GaitConfig()
        self.phase = 0.0
        self._cmd_speed = 0.0
        self._swing = SwingState()
        self._foothold = np.zeros((2, 2))  # per-leg xy target
        self._com_ref_xy = np.zeros(2)
        self._mid_xy = np.zeros(2)  # sagittal midline (no SS lateral sway)
        self._lat_anchor = 0.0  # heading-lateral CoM midline (world scalar)
        self._initialized = False

    def reset(self) -> None:
        self.phase = 0.0
        self._cmd_speed = 0.0
        self._swing = SwingState()
        self._com_ref_xy = np.zeros(2)
        self._mid_xy = np.zeros(2)
        self._lat_anchor = 0.0
        self._initialized = False

    def set_config(self, config: GaitConfig) -> None:
        self.config = config

    def snapshot(self) -> dict[str, float]:
        cfg = self.config
        return {
            "gait.mode": float(cfg.mode),
            "gait.speed": float(cfg.speed),
            "gait.heading": float(cfg.heading),
            "gait.step_period": float(cfg.step_period),
            "gait.duty_factor": float(cfg.duty_factor),
            "gait.step_length_max": float(cfg.step_length_max),
            "gait.swing_height": float(cfg.swing_height),
            "gait.stance_width": float(cfg.stance_width),
            "gait.capture_gain": float(cfg.capture_gain),
            "gait.walk_accel": float(cfg.walk_accel),
            "gait.ss_lateral_bias": float(cfg.ss_lateral_bias),
            "gait.foothold_retarget_s": float(cfg.foothold_retarget_s),
        }

    def apply_updates(self, updates: dict) -> dict[str, float]:
        """Clamp and apply ``gait.*`` Task_Targets; return applied floats."""
        cfg = self.config
        applied: dict[str, float] = {}
        for raw_name, raw_value in updates.items():
            name = str(raw_name).strip()
            if name not in GAIT_PARAMETER_NAMES:
                continue
            field = name.split(".", 1)[1]
            try:
                if field == "mode":
                    value = parse_gait_mode(raw_value)
                else:
                    value = float(raw_value)
            except (TypeError, ValueError):
                print(f"     * [WARN] Bad gait parameter '{name}' (ignored)")
                continue
            lo, hi = GAIT_LIMITS[name]
            value = float(np.clip(value, lo, hi))
            setattr(cfg, field, value)
            applied[name] = value
            if field == "mode" and value == MODE_STAND:
                self.reset()
        return applied

    def _leg_in_swing(self, phase: float, leg: int, duty: float) -> bool:
        """Double-support first: each leg swings for (1-duty) of the cycle, 180° out of phase."""
        # Left swings in [0, 1-duty); right in [0.5, 0.5+(1-duty)) mod 1.
        swing_len = 1.0 - duty
        if leg == 0:
            return phase < swing_len
        p = (phase - 0.5) % 1.0
        return p < swing_len

    def _swing_progress(self, phase: float, leg: int, duty: float) -> float:
        swing_len = max(1e-3, 1.0 - duty)
        if leg == 0:
            if phase >= swing_len:
                return 1.0
            return float(phase / swing_len)
        p = (phase - 0.5) % 1.0
        if p >= swing_len:
            return 1.0
        return float(p / swing_len)

    def _velocity_command(self, dt: float, com_xy: np.ndarray | None = None) -> np.ndarray:
        cfg = self.config
        target = float(cfg.speed)
        if cfg.mode == MODE_TREAD:
            target = min(target, 0.15)
        if cfg.mode == MODE_LIMP:
            target = min(target, 0.20)
        # Brake commanded speed when the feet are ahead of the torso (torso lag).
        # Prevents the MPC velocity task from yanking the base past the support.
        if com_xy is not None and self._initialized:
            c, s = np.cos(cfg.heading), np.sin(cfg.heading)
            forward = np.array([c, s])
            feet_lead = float(np.dot(self._mid_xy - com_xy, forward))
            if feet_lead > 0.03:
                brake = float(np.clip(1.0 - (feet_lead - 0.03) / 0.08, 0.15, 1.0))
                target *= brake
            # Also brake if the torso has raced ahead of the foot midline.
            torso_lead = float(np.dot(com_xy - self._mid_xy, forward))
            if torso_lead > 0.04:
                brake = float(np.clip(1.0 - (torso_lead - 0.04) / 0.06, 0.20, 1.0))
                target *= brake
        # Ramp commanded speed for smooth stand→walk.
        max_step = float(cfg.walk_accel) * dt
        if self._cmd_speed < target:
            self._cmd_speed = min(target, self._cmd_speed + max_step)
        else:
            # Allow faster deceleration than acceleration when braking.
            decel = max(max_step, 2.5 * float(cfg.walk_accel) * dt)
            self._cmd_speed = max(target, self._cmd_speed - decel)
        c, s = np.cos(cfg.heading), np.sin(cfg.heading)
        return np.array([self._cmd_speed * c, self._cmd_speed * s])

    def _capture_foothold(
        self,
        *,
        com_xy: np.ndarray,
        com_vel_xy: np.ndarray,
        v_cmd: np.ndarray,
        leg: int,
        period: float,
        duty: float,
        height: float,
    ) -> np.ndarray:
        """DCM / capture-point foothold with Raibert feedforward.

        Instantaneous DCM ``ξ = p + v/ω₀`` (Pratt capture point). Foot placement
        regulates ξ toward the constant-velocity target on the walk midline,
        which steps under a racing torso and lengthens stride when lagging.
        """
        cfg = self.config
        c, s = np.cos(cfg.heading), np.sin(cfg.heading)
        forward = np.array([c, s])
        left = np.array([-s, c])
        mid = self._mid_xy

        h = max(0.45, float(height))
        omega0 = float(np.sqrt(GRAVITY / h))
        stance_t = max(0.08, duty * period)
        # Regulate over ~one step (swing + half stance) so the next touchdown
        # catches the DCM before the following SS tip.
        T = max(0.12, (1.0 - duty) * period + 0.5 * stance_t)

        xi = com_xy + com_vel_xy / omega0
        # Desired DCM rides above the midline at the commanded speed offset.
        xi_des = mid + v_cmd / omega0
        # Englsberger DCM foot placement: u = (e^{ωT} ξ − ξ_d) / (e^{ωT} − 1).
        e_wt = float(np.exp(omega0 * T))
        foot_dcm = (e_wt * xi - xi_des) / max(e_wt - 1.0, 1e-3)

        # Mild Raibert feedforward so strides remain visible at low speed.
        v_fwd = float(np.dot(v_cmd, forward))
        v_err = com_vel_xy - v_cmd
        raibert_fwd = 0.50 * stance_t * v_fwd
        # Extra CP feedback scaled by capture_gain (1 ⇒ full 1/ω₀ on vel error).
        cp_fb = (cfg.capture_gain / omega0) * float(np.dot(v_err, forward))

        fwd_pos = float(np.dot(foot_dcm, forward)) + raibert_fwd + cp_fb
        # Keep foothold near the CoM / DCM — clamp step length.
        max_step = min(cfg.step_length_max, 0.04 + 0.25 * abs(self._cmd_speed))
        com_fwd = float(np.dot(com_xy, forward))
        fwd_pos = float(np.clip(fwd_pos, com_fwd - max_step, com_fwd + max_step))

        # Hard gates: feet racing ahead → step under CoM; torso racing → step out.
        lag = float(np.dot(mid - com_xy, forward))  # >0 feet ahead of torso
        if lag > 0.03:
            fwd_pos = min(fwd_pos, com_fwd - 0.6 * (lag - 0.01))
        elif lag < -0.03:
            fwd_pos = max(fwd_pos, com_fwd + min(max_step, -lag))

        lat_hip = 0.5 * cfg.stance_width * (1.0 if leg == 0 else -1.0)
        # Lateral: hip width + DCM lateral correction (clamped).
        lat_dcm = float(np.dot(foot_dcm, left)) - self._lat_anchor
        lat_off = lat_hip + float(np.clip(0.35 * lat_dcm, -0.04, 0.04))
        lat_off += 0.015 * float(np.dot(v_err, left))
        return fwd_pos * forward + (self._lat_anchor + lat_off) * left

    def _update_com_ref(
        self,
        *,
        dt: float,
        com_xy: np.ndarray,
        feet_xy: np.ndarray,
        v_cmd: np.ndarray,
        stance_side: float,
    ) -> np.ndarray:
        """Support-tied midline CoM reference with mild SS stance bias."""
        cfg = self.config
        c, s = np.cos(cfg.heading), np.sin(cfg.heading)
        forward = np.array([c, s])
        left = np.array([-s, c])
        del v_cmd  # reserved for anticipatory LIPM feedforward

        if not self._initialized:
            mid = feet_xy.mean(axis=0)
            self._mid_xy = mid.copy()
            self._com_ref_xy = mid.copy()
            self._lat_anchor = float(np.dot(mid, left))
            # Keep configured stance_width (may be wider than the standing pose
            # so first steps open the base for SS balance).
            duty = float(np.clip(cfg.duty_factor, 0.55, 0.92))
            swing_len = 1.0 - duty
            self.phase = 0.5 * (swing_len + 0.5)  # mid first DS window
            self._initialized = True

        # Sagittal: blend foot support and actual CoM so the reference cannot
        # sit on racing feet while the torso falls behind.
        foot_mid = feet_xy.mean(axis=0)
        # Small anticipatory lead only — large lead yanks the torso past the feet.
        lead = float(np.clip(0.12 * self._cmd_speed, 0.0, 0.03))
        lag = float(np.dot(foot_mid - com_xy, forward))
        if lag > 0.03:
            # Pull reference back toward the CoM so MPC/WBC push the torso up.
            blend = float(np.clip((lag - 0.03) / 0.08, 0.0, 0.8))
            support = (1.0 - blend) * foot_mid + blend * com_xy
            lead = 0.0
        elif lag < -0.03:
            # Torso ahead of feet: pin reference to the feet (don't chase the tip).
            support = foot_mid
            lead = 0.0
        else:
            support = foot_mid
        target_mid = support + lead * forward
        tau_fwd = 0.08
        alpha_fwd = 1.0 - float(np.exp(-dt / tau_fwd))
        self._mid_xy = self._mid_xy + alpha_fwd * (target_mid - self._mid_xy)
        mid_lat = float(np.dot(self._mid_xy, left))
        self._mid_xy = self._mid_xy + (self._lat_anchor - mid_lat) * left

        # Lateral: hold the walk midline, plus a small bias toward the stance
        # foot in SS. Full stance-foot snaps exceed CoP authority (~0.4 m/s²).
        bias = float(np.clip(cfg.ss_lateral_bias, 0.0, 0.05))
        target_lat = self._lat_anchor + bias * float(stance_side)
        tau = 0.08
        alpha = 1.0 - float(np.exp(-dt / tau))
        cur_lat = float(np.dot(self._com_ref_xy, left))
        new_lat = cur_lat + alpha * (target_lat - cur_lat)

        self._com_ref_xy = self._mid_xy + (new_lat - self._lat_anchor) * left
        return self._com_ref_xy.copy()

    def step(
        self,
        *,
        dt: float,
        com: np.ndarray,
        com_vel: np.ndarray,
        foot_centers: np.ndarray,
        ground_z: float,
        height: float,
        yaw: float,
        measured_mask: np.ndarray,
    ) -> GaitOutput:
        cfg = self.config
        com_xy = np.asarray(com[:2], dtype=float)
        com_vel_xy = np.asarray(com_vel[:2], dtype=float)
        feet = np.asarray(foot_centers, dtype=float).reshape(2, 3)
        feet_xy = feet[:, :2]

        if cfg.mode == MODE_STAND or (
            float(cfg.speed) <= 1e-4 and self._cmd_speed <= 1e-4
        ):
            self._cmd_speed = 0.0
            self._initialized = False
            mask = np.asarray(measured_mask, dtype=bool).copy()
            if not mask.any():
                mask[:] = True
            return GaitOutput(
                contact_mask=mask,
                com_ref_xy=feet_xy.mean(axis=0),
                com_vel_cmd=np.zeros(2),
                swing=SwingState(active=False),
                phase=self.phase,
                left_phase=LegPhase.STANCE,
                right_phase=LegPhase.STANCE,
            )

        v_cmd = self._velocity_command(dt, com_xy=com_xy)
        duty = float(np.clip(cfg.duty_factor, 0.55, 0.92))
        period = max(0.25, float(cfg.step_period))

        # Advance clock before CoM/swing so lateral shift matches this tick's support.
        if not self._initialized:
            # Prime anchors; phase set inside _update_com_ref on first call.
            pass
        else:
            self.phase = (self.phase + dt / period) % 1.0

        left_swing = self._leg_in_swing(self.phase, 0, duty) if self._initialized else False
        right_swing = self._leg_in_swing(self.phase, 1, duty) if self._initialized else False
        if left_swing and right_swing:
            right_swing = False

        if left_swing and not right_swing:
            stance_side = -1.0  # right foot planted
        elif right_swing and not left_swing:
            stance_side = 1.0  # left foot planted
        else:
            stance_side = 0.0

        com_ref_xy = self._update_com_ref(
            dt=dt,
            com_xy=com_xy,
            feet_xy=feet_xy,
            v_cmd=v_cmd,
            stance_side=stance_side,
        )
        # First tick just initialized: recompute swing with the DS start phase.
        if left_swing is False and right_swing is False and float(cfg.mode) >= MODE_WALK:
            left_swing = self._leg_in_swing(self.phase, 0, duty)
            right_swing = self._leg_in_swing(self.phase, 1, duty)
            if left_swing and right_swing:
                right_swing = False

        mask = np.ones(8, dtype=bool)
        if left_swing:
            mask[0:4] = False
        if right_swing:
            mask[4:8] = False

        swing = SwingState(active=False)
        for leg, is_swing in ((0, left_swing), (1, right_swing)):
            if not is_swing:
                continue
            s = self._swing_progress(self.phase, leg, duty)
            retarget = float(np.clip(cfg.foothold_retarget_s, 0.05, 0.60))
            # DCM retarget early in swing so a racing CoM still gets a catch step.
            new_swing = (not self._swing.active) or self._swing.leg != leg
            if new_swing or s < retarget:
                target_xy = self._capture_foothold(
                    com_xy=com_xy,
                    com_vel_xy=com_vel_xy,
                    v_cmd=v_cmd,
                    leg=leg,
                    period=period,
                    duty=duty,
                    height=height,
                )
                self._foothold[leg] = target_xy
                foothold = np.array([target_xy[0], target_xy[1], ground_z])
                start = feet[leg].copy() if new_swing else self._swing.start.copy()
                self._swing = SwingState(
                    active=True,
                    leg=leg,
                    start=start,
                    foothold=foothold,
                    des_pos=feet[leg].copy() if new_swing else self._swing.des_pos.copy(),
                    s=s,
                )
            else:
                self._swing.s = s
                self._swing.foothold[2] = ground_z
            self._swing.des_pos = _bezier_swing_pos(
                self._swing.start,
                self._swing.foothold,
                self._swing.s,
                swing_height=cfg.swing_height,
                ground_z=ground_z,
            )
            swing = self._swing
            break
        else:
            self._swing = SwingState(active=False)

        return GaitOutput(
            contact_mask=mask,
            com_ref_xy=com_ref_xy,
            com_vel_cmd=v_cmd,
            swing=swing,
            phase=self.phase,
            left_phase=LegPhase.SWING if left_swing else LegPhase.STANCE,
            right_phase=LegPhase.SWING if right_swing else LegPhase.STANCE,
        )
