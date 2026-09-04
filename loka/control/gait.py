"""Gait clock, footstep plan, DCM reference trajectory and swing arcs.

Walking geometry stays classical and outside the LLM vocabulary. The
orchestrator only tunes the high-level ``gait.*`` policy knobs declared in
:data:`GAIT_LIMITS`; footstep coordinates and per-tick contact flags are
decided here. If the LLM sets only ``gait.speed``, :func:`gait_schedule_for_speed`
fills period, duty and stance width so 0.10 and 0.50 m/s share one controller.

The plan *leads* the robot rather than following it:

* **Footsteps** form a chain anchored at the current support foot. Each link
  advances by ``speed × T_step`` along the heading and crosses ``stance_width``
  to the other side, so commanded progress is built into the geometry.
* The **CoM reference** is the DCM trajectory implied by that footstep sequence.
  Each foothold is where the foot lands; the ZMP sits *inside* the sole, inset
  from the centre, so the orbit leaves CoP travel for tracking. Within a step
  the ZMP slides across double support onto that inset rather than jumping:

      ξ(τ)      = p(τ) + (ξ_eos,k − p(τ)) · e^{ω₀(τ − T_step)}   within step k
      ξ_eos,k−1 = p_k + (ξ_eos,k − p_k) · e^{−ω₀ T_step}         backward
      ċ_ref     = ω₀ (ξ_ref − c_ref)                             CoM from DCM

  The backward recursion is Englsberger's DCM planner; the forward CoM
  integration is a stable first-order filter, so the position reference is
  continuous even when the ZMP reference moves sideways.

Anchoring the chain to the support foot is what keeps the plan *coherent*
without letting measurement set the pace. Every link is exactly one commanded
stride long, so the plan's velocity is the commanded velocity by construction,
whatever the robot is doing; but the chain starts where the robot actually
stands, so the footholds are always reachable and the CoM reference always sits
near the body. Position error and speed error are thereby decoupled: the plan
absorbs an overshoot in position and still commands the right speed.

Measured state enters in exactly two bounded places: a clamped DCM correction
on the next foothold, and a saturation on the CoM tracking error handed to the
controller. Both can reject disturbances; neither can cancel the feedforward.
That distinction is the whole fix -- the previous version derived the CoM
reference from the *measured* feet while placing the feet relative to the
*measured* CoM, a loop containing no term that commanded forward progress at
all, which is why it could only shuffle.
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

#: Footsteps planned ahead of the current one. The terminal condition assumes
#: the robot halts on the last planned step, and its influence on the current
#: step decays by ``e^{-ω₀ T_step}`` (~0.27) per step, so five is already far
#: beyond what the current step can feel.
PLAN_HORIZON = 5

#: Saturation on the CoM tracking error handed to the controller [m]. Bounds
#: what the WBC is asked for without touching the plan itself: the target is
#: pulled *toward* the robot, never past the plan.
MAX_COM_REF_ERROR = 0.06

#: Ceiling on the DCM foothold correction [m]. Large enough to absorb a real
#: push, small enough that it cannot invert the nominal stride.
MAX_DCM_CORRECTION = 0.20

#: Length of the opening double-support, in step durations.
#:
#: LIPM acceleration is away from the ZMP, so loading the first *stance* foot
#: throws CoM onto the swing side and first lift-off has the wrong sign. The
#: opening ZMP sits on the upcoming swing-foot inset (APA). A midpoint ZMP
#: never loads the orbit; a stance-foot ZMP reverses it.
OPENING_TRANSFER_STEPS = 1.0

#: Inset of the planned ZMP from each foot centre toward the midline [m].
#: The G1 sole is 60 mm wide; parking ZMP at the centre spent 87% of that
#: on the nominal orbit and left ~0.05 m/s² of rejectable accel. 12 mm in
#: leaves the outer 18 mm for tracking.
ZMP_INSET = 0.012

#: How close the swing-foot *centre* must be to the ground before the gait
#: clock may leave single support [m]. Contact sites trigger at 5 mm, but a
#: pitched sole can scuff a toe while the centre is still 15 mm up -- which
#: is how a 40 mm miss used to count as a landing. 8 mm is just above the
#: sensor and well below a hovering foot.
TOUCHDOWN_HEIGHT = 0.008

#: Height that distinguishes a real swing from a scuff [m]. The seated
#: test is 8 mm; a 9 mm hop that never tracked the Bézier must not count
#: as airborne, or the clock early-plants at ``s = 0.75`` and cuts off the
#: lagged Cartesian lift. Isolation tests keep z = 0 and never reach this.
SWING_CLEARED_HEIGHT = 0.020

#: Longest extra single-support, past ``T_step``, spent waiting for that
#: contact [s]. The right swing was landing 30–40 mm high at clock timeout;
#: ~0.1 s of PD on the Cartesian task closes that, and 0.2 s is a hard cap
#: so a falling robot cannot freeze the scheduler.
MAX_TOUCHDOWN_HOLD = 0.20

#: Earliest the clock may commit a step, as a fraction of the swing window.
#: Same test as the late hold (centre down *and* a measured sole contact),
#: but only after the foot has actually been airborne this swing — otherwise
#: planner-isolation tests, which keep z = 0, would skip the last quarter of
#: every Bézier.
EARLY_PLANT_S = 0.75

#: How close the swing-foot centre must be to the planned foothold [m] before
#: a real (airborne) swing counts as planted. The G1 sole is 60 mm wide; if
#: the planned point is farther than that, it is no longer on the real sole
#: and the next ZMP is a lie. Isolation tests never leave the ground, so
#: they skip this gate and keep the nominal cadence.
TOUCHDOWN_RADIUS = 0.030

#: Extra horizontal speed the Cartesian swing task can spend catching a
#: moving foothold [m/s]. Capture that outruns this in the remaining swing
#: is not a kinematically feasible plant (Galliker et al. 2022: the foot
#: pose at contact enters through kinematics, not as an unbounded DCM nudge).
SWING_CATCHUP_SPEED = 0.45

#: Minimum lateral gap between consecutive footholds [m], so a lateral
#: correction can never plant one foot on top of the other. The G1 sole is
#: 0.06 m wide with its inner edge 0.0885 m off centre.
MIN_FOOT_SEPARATION = 0.16


class LegPhase(str, Enum):
    STANCE = "stance"
    SWING = "swing"


@dataclass
class GaitConfig:
    """High-level gait policy (LLM / operator facing)."""

    mode: float = MODE_STAND  # see MODE_* codes
    speed: float = 0.0  # commanded travel speed in the heading frame [m/s]
    heading: float = 0.0  # world-frame travel direction [rad]
    #: Full two-step cycle: each leg swings once, so a step lasts half of this.
    step_period: float = 0.70
    #: Fraction of the cycle each leg spends in stance. Must exceed 0.5 or the
    #: two swing windows would overlap and leave the robot airborne.
    duty_factor: float = 0.65
    step_length_max: float = 0.30  # cap on stride between consecutive footholds [m]
    swing_height: float = 0.045  # peak clearance above the foot line [m]
    stance_width: float = 0.24  # lateral foot separation [m]; nominal G1 is 0.237
    #: DCM foothold feedback gain. 1.0 is deadbeat capture-point placement;
    #: below 1 trades disturbance rejection for a smoother nominal stride.
    capture_gain: float = 0.6
    walk_accel: float = 0.5  # ramp on the commanded speed [m/s²]
    #: Fraction of swing during which the foothold may still be retargeted.
    foothold_retarget_s: float = 0.70


@dataclass(frozen=True)
class GaitKnob:
    """One gait policy scalar, with the band it is clamped into.

    The same declaration serves the operator's slider and the orchestrator's
    prompt, so the two cannot drift apart on either the range or the meaning.
    """

    name: str  # "gait.speed"
    summary: str
    low: float
    high: float

    @property
    def field(self) -> str:
        return self.name.split(".", 1)[1]


GAIT_KNOBS: tuple[GaitKnob, ...] = (
    GaitKnob("gait.mode", "0=stand 1=walk 2=tread (march in place) 3=limp", 0.0, 3.0),
    GaitKnob("gait.speed", "Travel speed along the heading [m/s]", 0.0, 0.80),
    GaitKnob("gait.heading", "World-frame travel direction [rad]", -np.pi, np.pi),
    GaitKnob("gait.step_period",
             "Full two-step cycle [s]; one step is half of this", 0.40, 1.20),
    GaitKnob("gait.duty_factor",
             "Stance fraction per leg. Must stay above 0.5 or both feet leave "
             "the ground at once", 0.55, 0.85),
    GaitKnob("gait.step_length_max",
             "Cap on the stride between consecutive footholds [m]", 0.05, 0.45),
    GaitKnob("gait.swing_height", "Peak swing clearance above the foot line [m]",
             0.02, 0.15),
    GaitKnob("gait.stance_width", "Lateral foot separation [m]; nominal G1 is 0.237",
             0.18, 0.34),
    GaitKnob("gait.capture_gain",
             "DCM foothold feedback. 1.0 is deadbeat capture-point placement; "
             "below 1 trades disturbance rejection for a smoother stride",
             0.0, 1.5),
    GaitKnob("gait.walk_accel", "Ramp on the commanded speed [m/s^2]", 0.05, 2.0),
    GaitKnob("gait.foothold_retarget_s",
             "Fraction of swing during which the foothold may still move",
             0.05, 0.95),
)

#: Clamps applied when the LLM / operator sets gait.* Task_Targets.
GAIT_LIMITS: dict[str, tuple[float, float]] = {
    knob.name: (knob.low, knob.high) for knob in GAIT_KNOBS
}

GAIT_PARAMETER_NAMES = frozenset(GAIT_LIMITS.keys())

#: Cadence / width knobs the speed schedule may fill when the LLM omitted them.
_SCHEDULED_FIELDS = frozenset({"step_period", "duty_factor", "stance_width"})


def gait_schedule_for_speed(speed: float) -> dict[str, float]:
    """Period, duty and stance width that keep a DCM orbit on a 60 mm sole.

    Closed-loop 0.20 m/s with isolation's duty 0.65 falls at ~3 s: CoP sits
    on the sole edge, then sagittal ``land_f`` grows. Pinning duty ~0.80
    (more double-support, shorter single-support exponential) is a bounded
    walk. High speed needs the opposite — enough swing time to finish a
    longer stride — so duty falls and the cycle shortens toward 0.50 m/s.
    Stance stays near the G1's 0.237 m; widening it grew the orbit. The LLM
    may override any of these; this is what runs when it only sets
    ``gait.speed``.
    """
    t = float(np.clip(abs(float(speed)) / 0.50, 0.0, 1.0))
    period = 0.80 * (1.0 - t) + 0.64 * t
    duty = 0.84 * (1.0 - t) + 0.74 * t
    width = 0.22 * (1.0 - t) + 0.24 * t
    return {
        "gait.step_period": float(np.clip(period, *GAIT_LIMITS["gait.step_period"])),
        "gait.duty_factor": float(np.clip(duty, *GAIT_LIMITS["gait.duty_factor"])),
        "gait.stance_width": float(np.clip(width, *GAIT_LIMITS["gait.stance_width"])),
    }


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


def heading_frame(heading: float) -> tuple[np.ndarray, np.ndarray]:
    """Unit ``(forward, left)`` vectors of the travel frame, world xy."""
    c, s = np.cos(heading), np.sin(heading)
    return np.array([c, s]), np.array([-s, c])


def swing_reference(
    start: np.ndarray, end: np.ndarray, s: float, *, swing_height: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Swing-arc position and its first two ``d/ds`` derivatives.

    Horizontal motion follows a smoothstep, whose slope vanishes at both ends,
    so the foot neither jerks off the ground at lift-off nor scuffs forward as
    it lands. Vertical motion is a half sine, which peaks at mid-swing and
    arrives with a downward velocity that seats the contact.

    Both components are analytic, so the caller gets velocity *and* acceleration
    feedforward for free. That matters more than the tracking gains: asked to
    synthesise a 0.25 s arc from position error alone, any sane gain saturates,
    and the task degenerates into bang-bang.
    """
    s = float(np.clip(s, 0.0, 1.0))
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    shape = s * s * (3.0 - 2.0 * s)
    dshape = 6.0 * s * (1.0 - s)
    ddshape = 6.0 - 12.0 * s
    lift = np.sin(np.pi * s)
    dlift = np.pi * np.cos(np.pi * s)
    ddlift = -np.pi * np.pi * np.sin(np.pi * s)

    delta = end - start
    pos = start + delta * shape
    vel = delta * dshape
    acc = delta * ddshape
    pos[2] += swing_height * lift
    vel[2] += swing_height * dlift
    acc[2] += swing_height * ddlift
    return pos, vel, acc


@dataclass
class Footstep:
    """One planned footfall: which leg, where, and whether it may still move."""

    leg: int  # 0=left, 1=right, -1 for the initial both-feet step
    pos: np.ndarray  # (2,) world xy, nominal plus any correction
    #: Where the walk line alone puts this step. Corrections are always applied
    #: as a bounded offset from *this*, never from the previous foothold, or a
    #: lateral bias would compound step over step into a sideways drift.
    nominal: np.ndarray | None = None
    #: Accumulated offset from ``nominal``. Held on the footstep rather than
    #: recomputed, because the nominal is regenerated from the walk line every
    #: tick and would otherwise erase the correction the instant the foothold
    #: is committed -- planting the foot behind the capture point.
    correction: np.ndarray = field(default_factory=lambda: np.zeros(2))
    frozen: bool = False
    #: The opening step of a walk, where both feet stay planted while the
    #: opening ZMP slides onto the upcoming swing foot (APA).
    initial: bool = False


@dataclass
class SwingState:
    active: bool = False
    leg: int = 0  # 0=left, 1=right
    start: np.ndarray = field(default_factory=lambda: np.zeros(3))
    foothold: np.ndarray = field(default_factory=lambda: np.zeros(3))
    des_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    des_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    des_acc: np.ndarray = field(default_factory=lambda: np.zeros(3))
    s: float = 0.0


@dataclass
class GaitOutput:
    """One control-tick gait decision."""

    contact_mask: np.ndarray  # (8,) bool, planned
    com_ref_xy: np.ndarray  # (2,) world
    com_vel_ref: np.ndarray  # (2,) world
    dcm_ref: np.ndarray  # (2,) world
    swing: SwingState
    phase: float  # position within the two-step cycle, [0, 1)
    left_phase: LegPhase
    right_phase: LegPhase
    walking: bool


class GaitScheduler:
    """Periodic bipedal gait: footstep plan, DCM reference and swing arcs."""

    def __init__(self, config: GaitConfig | None = None) -> None:
        self.config = config or GaitConfig()
        #: ``step_period`` / ``duty_factor`` / ``stance_width`` the operator or
        #: LLM set explicitly. The speed schedule fills the rest.
        self._user_gait_fields: set[str] = set()
        self.reset()

    # -- lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        self._step = 0  # index of the current support step
        self._tau = 0.0  # elapsed time within the current step [s]
        self._cmd_speed = 0.0  # ramped commanded speed [m/s]
        self._steps: dict[int, Footstep] = {}
        self._first_leg = 0  # leg that supports step 1
        self._com_ref = np.zeros(2)
        self._com_vel_ref = np.zeros(2)
        self._dcm_ref = np.zeros(2)
        self._swing = SwingState()
        self._initial_feet = np.zeros((2, 2))
        self._opening_zmp = np.zeros(2)
        self._active = False
        self._touchdown_hold = 0.0
        self._swing_cleared = False

    def set_config(self, config: GaitConfig) -> None:
        self.config = config

    @property
    def cmd_speed(self) -> float:
        """Ramped commanded speed [m/s]."""
        return float(self._cmd_speed)

    @property
    def walking(self) -> bool:
        """True while the gait clock owns the contact schedule."""
        return self._active

    @property
    def step_index(self) -> int:
        """Support steps completed since the walk began."""
        return int(self._step)

    def wants_walk(self) -> bool:
        cfg = self.config
        return float(cfg.mode) >= MODE_WALK and (
            float(cfg.speed) > 1e-4 or self._cmd_speed > 1e-4 or self._active
        )

    # -- policy surface ----------------------------------------------------

    def snapshot(self) -> dict[str, float]:
        cfg = self.config
        return {name: float(getattr(cfg, name.split(".", 1)[1])) for name in GAIT_LIMITS}

    def apply_updates(self, updates: dict) -> dict[str, float]:
        """Clamp and apply ``gait.*`` Task_Targets; return the applied floats."""
        cfg = self.config
        applied: dict[str, float] = {}
        for raw_name, raw_value in updates.items():
            name = str(raw_name).strip()
            if name not in GAIT_PARAMETER_NAMES:
                continue
            field_name = name.split(".", 1)[1]
            try:
                if field_name == "mode":
                    value = parse_gait_mode(raw_value)
                else:
                    value = float(raw_value)
            except (TypeError, ValueError):
                print(f"     * [WARN] Bad gait parameter '{name}' (ignored)")
                continue
            lo, hi = GAIT_LIMITS[name]
            value = float(np.clip(value, lo, hi))
            setattr(cfg, field_name, value)
            applied[name] = value
            if field_name in _SCHEDULED_FIELDS:
                self._user_gait_fields.add(field_name)
            if field_name == "mode" and value == MODE_STAND:
                self._user_gait_fields.clear()
                self.reset()
        applied.update(self.apply_speed_schedule())
        return applied

    def apply_speed_schedule(self) -> dict[str, float]:
        """Fill cadence/width from ``gait.speed`` unless the user pinned them."""
        cfg = self.config
        if float(cfg.mode) < MODE_WALK or abs(float(cfg.speed)) < 1e-4:
            return {}
        applied: dict[str, float] = {}
        for name, value in gait_schedule_for_speed(cfg.speed).items():
            field_name = name.split(".", 1)[1]
            if field_name in self._user_gait_fields:
                continue
            setattr(cfg, field_name, value)
            applied[name] = value
        return applied

    # -- timing ------------------------------------------------------------

    def _durations(self) -> tuple[float, float, float]:
        """``(T_step, T_swing, T_ds)`` for the current policy [s].

        A step is half a cycle. Each leg swings for ``(1 - duty)`` of the cycle,
        leaving ``(duty - 0.5)`` of double support on either side of it.
        """
        cfg = self.config
        period = max(0.20, float(cfg.step_period))
        duty = float(np.clip(cfg.duty_factor, 0.55, 0.85))
        t_step = 0.5 * period
        t_swing = (1.0 - duty) * period
        return t_step, t_swing, max(0.0, t_step - t_swing)

    def _current_duration(self, t_step: float) -> float:
        """Duration of the step in progress [s].

        Only the opening double-support transfer differs; see
        :data:`OPENING_TRANSFER_STEPS`.
        """
        step = self._steps.get(self._step)
        if step is not None and step.initial:
            return t_step * OPENING_TRANSFER_STEPS
        return t_step

    def _omega(self, height: float) -> float:
        return float(np.sqrt(GRAVITY / max(0.30, float(height))))

    def _swing_leg(self) -> int:
        """Leg currently airborne, or -1 when both feet should be down."""
        landing = self._steps.get(self._step + 1)
        if landing is None or landing.leg < 0:
            return -1
        return int(landing.leg)

    def _swing_foot_down(
        self,
        feet: np.ndarray,
        ground_z: float,
        measured_mask: np.ndarray,
    ) -> bool:
        """True when the in-flight foot has actually planted.

        Height and measured contact both have to agree. Height alone would
        accept a sole that is close but unloaded; contact alone would accept
        a toe scuff 4 cm up if a site flickered. A swing that actually left
        the ground also has to finish near the planned foothold — otherwise
        the clock names a short scuff as stance and the next ZMP sits off
        the sole. Isolation tests keep z = 0, never set ``_swing_cleared``,
        and still tick on time.
        """
        leg = self._swing_leg()
        if leg < 0:
            return True
        height = float(feet[leg, 2] - ground_z)
        rows = slice(0, 4) if leg == 0 else slice(4, 8)
        planted = bool(np.asarray(measured_mask, dtype=bool).reshape(-1)[rows].any())
        if height > TOUCHDOWN_HEIGHT or not planted:
            return False
        if not self._swing_cleared:
            return True
        landing = self._steps.get(self._step + 1)
        if landing is None:
            return True
        return float(np.linalg.norm(feet[leg, :2] - landing.pos)) <= TOUCHDOWN_RADIUS

    @property
    def phase(self) -> float:
        """Position within the two-step cycle, for telemetry."""
        t_step, _, _ = self._durations()
        span = self._current_duration(t_step)
        half = 0.5 * (self._tau / span if span > 0 else 0.0)
        return float((half + 0.5 * (self._step % 2)) % 1.0)

    # -- commanded velocity ------------------------------------------------

    def _velocity_command(self, dt: float) -> np.ndarray:
        cfg = self.config
        target = float(cfg.speed)
        if cfg.mode == MODE_TREAD:
            target = 0.0
        elif cfg.mode == MODE_LIMP:
            target = min(target, 0.25)

        rate = float(cfg.walk_accel) * dt
        if self._cmd_speed < target:
            self._cmd_speed = min(target, self._cmd_speed + rate)
        else:
            # Decelerate faster than we accelerate; stopping is the safe way to
            # be wrong about how much authority is left.
            self._cmd_speed = max(target, self._cmd_speed - 2.0 * rate)

        forward, _ = heading_frame(float(cfg.heading))
        return self._cmd_speed * forward

    # -- footstep plan -----------------------------------------------------

    def _leg_of(self, index: int) -> int:
        """Which leg supports step ``index``."""
        if index <= 0:
            return -1
        return (self._first_leg + index - 1) % 2

    def _refresh_plan(self, *, t_step: float, v_cmd: np.ndarray) -> None:
        """Rebuild the unfrozen tail of the footstep chain.

        Each link is one commanded stride further along the heading and one
        ``stance_width`` across, starting from the foot the robot is actually
        standing on. The chain therefore commands the right *speed* regardless
        of where the robot has got to, while staying within reach of it.
        """
        _, left = heading_frame(float(self.config.heading))
        stride = v_cmd * t_step
        width = float(self.config.stance_width)

        previous = self._steps[self._step].pos
        for index in range(self._step + 1, self._step + PLAN_HORIZON + 1):
            existing = self._steps.get(index)
            side = 1.0 if self._leg_of(index) == 0 else -1.0
            nominal = previous + stride + side * width * left
            if existing is None:
                self._steps[index] = Footstep(
                    leg=self._leg_of(index), pos=nominal.copy(), nominal=nominal
                )
            elif not existing.frozen:
                existing.nominal = nominal
                existing.pos = nominal + existing.correction
            previous = self._steps[index].pos

        for index in list(self._steps):
            if index < self._step - 1:
                del self._steps[index]

    def _apply_dcm_correction(
        self, *, dcm: np.ndarray, s: float, omega: float, t_step: float
    ) -> None:
        """Nudge the in-flight foothold to absorb DCM error.

        The error is evaluated at *touchdown*, not now: the measured DCM is
        propagated over the remaining step time under the support foot, and
        compared with the planned end-of-step DCM. DCM dynamics are divergent,
        so an instantaneous error understates where the robot is actually going
        and lands the correction a quarter-cycle out of phase -- which injects
        a step-synchronous wobble instead of damping one.

        Bounded twice over: the correction is a clamped offset from the nominal
        foothold, and the result is projected back inside a reachable,
        non-crossing stride. Feedback may rescue a step; it may not redesign the
        gait.
        """
        cfg = self.config
        target = self._steps.get(self._step + 1)
        if target is None or target.frozen or target.nominal is None:
            return
        if s > float(np.clip(cfg.foothold_retarget_s, 0.05, 0.95)):
            target.frozen = True
            return

        support_zmp = self._zmp_of(self._steps[self._step])
        remaining = max(self._current_duration(t_step) - self._tau, 1e-3)
        predicted = support_zmp + (np.asarray(dcm) - support_zmp) * float(
            np.exp(omega * remaining)
        )
        goal = self._dcm_eos(omega=omega, t_step=t_step)

        correction = float(cfg.capture_gain) * (predicted - goal)
        norm = float(np.linalg.norm(correction))
        if norm > MAX_DCM_CORRECTION:
            correction *= MAX_DCM_CORRECTION / norm
        pos = target.nominal + correction
        # Late-swing only: remaining Bézier time is how far the target may
        # still travel. Early swing keeps the full capture (isolation tests
        # and a racing CoM need that). After mid-swing, a 95–200 mm yank is
        # how step 5 landed 21 mm short — Galliker: the foot pose at contact
        # is kinematic, not an unbounded DCM nudge.
        if s > 0.45:
            _, t_swing, _ = self._durations()
            budget = SWING_CATCHUP_SPEED * max((1.0 - s) * t_swing, 1e-3)
            delta = pos - np.asarray(target.pos, dtype=float)
            dnorm = float(np.linalg.norm(delta))
            if dnorm > budget:
                pos = np.asarray(target.pos, dtype=float) + delta * (budget / dnorm)

        support = self._steps.get(self._step)
        if support is not None and not support.initial:
            forward, left = heading_frame(float(cfg.heading))
            side = 1.0 if target.leg == 0 else -1.0
            rel = pos - support.pos
            # Subtract only the violation, so an inactive constraint leaves the
            # foothold exactly where the walk line put it.
            stride = float(np.dot(rel, forward))
            excess = stride - float(
                np.clip(stride, -cfg.step_length_max, cfg.step_length_max)
            )
            pos = pos - excess * forward
            gap = side * float(np.dot(rel, left))
            if gap < MIN_FOOT_SEPARATION:
                pos = pos + side * (MIN_FOOT_SEPARATION - gap) * left
        target.correction = pos - target.nominal
        target.pos = pos

    # -- DCM reference -----------------------------------------------------

    def _left_axis(self) -> np.ndarray:
        _, left = heading_frame(float(self.config.heading))
        return left

    def _zmp_of(self, step: Footstep) -> np.ndarray:
        """Inset ZMP for a foothold: inside the sole, toward the midline.

        ``step.pos`` is where the foot lands. The orbit's ZMP sits 12 mm in
        from the centre so CoP has spare travel (see :data:`ZMP_INSET`).
        """
        pos = np.asarray(step.pos, dtype=float).reshape(2)
        if step.leg < 0:
            return pos.copy()
        side = 1.0 if step.leg == 0 else -1.0
        return pos - side * ZMP_INSET * self._left_axis()

    def _zmp_now(self, *, t_step: float, t_ds: float) -> np.ndarray:
        """ZMP at the current instant: opening APA, then the stance inset."""
        step = self._steps.get(self._step)
        if step is None:
            return np.zeros(2)
        if step.initial:
            return self._opening_zmp.copy()
        return self._zmp_of(step)

    def _dcm_eos(self, *, omega: float, t_step: float) -> np.ndarray:
        """Planned DCM at the end of the current step.

        Recurses backward from the last planned footstep, where the robot is
        assumed to come to rest. Each step back attenuates by
        ``e^{-ω₀ T_step}``, so the terminal assumption is invisible from here.
        Future ZMPs are the inset footholds; the current step's DS slide is
        a local forward smoothing and does not enter the recursion.
        """
        last = self._step + PLAN_HORIZON
        eos = self._zmp_of(self._steps[last])
        decay = float(np.exp(-omega * t_step))
        for index in range(last, self._step, -1):
            zmp = self._zmp_of(self._steps[index])
            eos = zmp + (eos - zmp) * decay
        return eos

    def _dcm_at(self, *, omega: float, t_step: float, t_ds: float) -> np.ndarray:
        """DCM reference for the current instant."""
        eos = self._dcm_eos(omega=omega, t_step=t_step)
        span = self._current_duration(t_step)
        zmp = self._zmp_now(t_step=t_step, t_ds=t_ds)
        return zmp + (eos - zmp) * float(np.exp(omega * (self._tau - span)))

    def _integrate_com_ref(self, *, dt: float, omega: float, dcm_ref: np.ndarray) -> None:
        """``ċ = ω(ξ − c)``, on plan state only.

        Measurement is kept out of this integrator on purpose. It is a stable
        first-order filter of the DCM reference, which is itself pinned to the
        footstep plan, which is clamped to a reachable stride -- so the plan
        stays bounded without ever being steered by the robot's own error.
        """
        self._com_vel_ref = omega * (dcm_ref - self._com_ref)
        self._com_ref = self._com_ref + self._com_vel_ref * dt

    def _com_ref_output(self, com_xy: np.ndarray) -> np.ndarray:
        """Plan position with the tracking error saturated."""
        error = self._com_ref - com_xy
        norm = float(np.linalg.norm(error))
        if norm <= MAX_COM_REF_ERROR:
            return self._com_ref.copy()
        return com_xy + error * (MAX_COM_REF_ERROR / norm)

    # -- start / stop ------------------------------------------------------

    def _begin(self, *, feet_xy: np.ndarray, com_xy: np.ndarray) -> None:
        """Seed the plan from the current stance."""
        stance = 1  # right supports first, so the left steps out
        swing = 1 - stance
        self._com_ref = com_xy.copy()
        self._com_vel_ref = np.zeros(2)
        self._step = 0
        self._tau = 0.0
        self._swing = SwingState()
        self._initial_feet = feet_xy.copy()
        # Opening contact is still both feet. The DCM ZMP slides from the
        # midpoint onto the upcoming swing-foot inset (APA); the stored
        # footstep is the stance so the chain after it is coherent.
        left = self._left_axis()
        swing_inset = feet_xy[swing] - (1.0 if swing == 0 else -1.0) * ZMP_INSET * left
        self._opening_zmp = swing_inset
        self._steps = {
            0: Footstep(
                leg=stance, pos=feet_xy[stance].copy(), frozen=True, initial=True
            ),
            1: Footstep(leg=stance, pos=feet_xy[stance].copy(), frozen=True),
        }
        self._first_leg = stance
        self._active = True
        self._touchdown_hold = 0.0
        self._swing_cleared = False

    def _may_stop(self, *, t_step: float) -> bool:
        """True once halting would leave the feet square and level.

        Called only at a step boundary, so stopping always completes the step
        in progress rather than freezing a leg in mid-air.
        """
        if self._cmd_speed > 1e-3:
            return False
        forward, _ = heading_frame(float(self.config.heading))
        here = self._steps.get(self._step)
        nxt = self._steps.get(self._step + 1)
        if here is None or nxt is None:
            return False
        return abs(float(np.dot(nxt.pos - here.pos, forward))) < 0.02

    def _stand_output(self, measured_mask: np.ndarray, feet_xy: np.ndarray) -> GaitOutput:
        mask = np.asarray(measured_mask, dtype=bool).copy()
        if not mask.any():
            mask[:] = True
        mid = feet_xy.mean(axis=0)
        return GaitOutput(
            contact_mask=mask,
            com_ref_xy=mid,
            com_vel_ref=np.zeros(2),
            dcm_ref=mid,
            swing=SwingState(active=False),
            phase=0.0,
            left_phase=LegPhase.STANCE,
            right_phase=LegPhase.STANCE,
            walking=False,
        )

    # -- main entry point --------------------------------------------------

    def step(
        self,
        *,
        dt: float,
        com: np.ndarray,
        com_vel: np.ndarray,
        foot_centers: np.ndarray,
        ground_z: float,
        height: float,
        measured_mask: np.ndarray,
    ) -> GaitOutput:
        com_xy = np.asarray(com[:2], dtype=float)
        com_vel_xy = np.asarray(com_vel[:2], dtype=float)
        feet = np.asarray(foot_centers, dtype=float).reshape(2, 3)
        feet_xy = feet[:, :2]

        if not self.wants_walk():
            if self._active:
                self.reset()
            return self._stand_output(measured_mask, feet_xy)

        if not self._active:
            self._begin(feet_xy=feet_xy, com_xy=com_xy)

        t_step, t_swing, t_ds = self._durations()
        v_cmd = self._velocity_command(dt)
        omega = self._omega(height)

        # -- advance the clock ---------------------------------------------
        # The Bézier is scheduled to arrive at T_step, but the plant does not
        # always make it: the right swing was still 30–40 mm up when this
        # used to increment, and the next tick named that airborne foot as
        # stance. Hold the clock at the end of the step until the foot is
        # actually down (or the hold times out). The matching early path
        # commits once a *real* swing has seated, so a late Bézier does not
        # keep dragging a planted foot.
        span = self._current_duration(t_step)
        self._tau += dt
        if (
            self._swing_cleared
            and t_swing > 1e-6
            and self._tau >= t_ds + EARLY_PLANT_S * t_swing
            and self._tau < span
            and self._swing_foot_down(feet, ground_z, measured_mask)
        ):
            self._tau = span
        if self._tau >= span:
            support = self._steps.get(self._step)
            waiting = (
                support is not None
                and not support.initial
                and not self._swing_foot_down(feet, ground_z, measured_mask)
                and self._touchdown_hold < MAX_TOUCHDOWN_HOLD
            )
            if waiting:
                self._tau = span
                self._touchdown_hold += dt
            else:
                self._tau = max(0.0, self._tau - span)
                self._step += 1
                self._touchdown_hold = 0.0
                # Isolation keeps z = 0 and never clears, so the planned
                # chain is already the truth. A real swing is re-anchored
                # at the measured plant: a short landing must shorten the
                # next nominal, not leave a ZMP in the air.
                cleared = self._swing_cleared
                self._swing_cleared = False
                if self._may_stop(t_step=t_step):
                    self.reset()
                    return self._stand_output(measured_mask, feet_xy)
                support = self._steps.get(self._step)
                if support is not None:
                    if support.leg >= 0 and cleared:
                        support.pos = feet_xy[support.leg].copy()
                        if support.nominal is not None:
                            support.correction = support.pos - support.nominal
                    support.frozen = True
                self._swing = SwingState()
        else:
            self._touchdown_hold = 0.0

        self._refresh_plan(t_step=t_step, v_cmd=v_cmd)

        # -- swing window ---------------------------------------------------
        support = self._steps[self._step]
        in_swing = (not support.initial) and self._tau >= t_ds and t_swing > 1e-6
        s = float(np.clip((self._tau - t_ds) / t_swing, 0.0, 1.0)) if in_swing else 0.0

        # -- DCM reference, then bounded foothold feedback -------------------
        dcm = com_xy + com_vel_xy / omega
        if in_swing:
            self._apply_dcm_correction(dcm=dcm, s=s, omega=omega, t_step=t_step)
        dcm_ref = self._dcm_at(omega=omega, t_step=t_step, t_ds=t_ds)
        self._dcm_ref = dcm_ref
        self._integrate_com_ref(dt=dt, omega=omega, dcm_ref=dcm_ref)

        # -- contacts and swing trajectory ----------------------------------
        mask = np.ones(8, dtype=bool)
        swing = SwingState(active=False)
        left_phase = right_phase = LegPhase.STANCE
        if in_swing:
            leg = self._steps[self._step + 1].leg
            rows = slice(0, 4) if leg == 0 else slice(4, 8)
            mask[rows] = False
            if not self._swing.active or self._swing.leg != leg:
                # Lift off from where the foot actually is, not where the plan
                # thought it would be.
                self._swing = SwingState(active=True, leg=leg, start=feet[leg].copy())
                self._swing_cleared = False
            if float(feet[leg, 2] - ground_z) > SWING_CLEARED_HEIGHT:
                self._swing_cleared = True
            foothold = np.array(
                [
                    self._steps[self._step + 1].pos[0],
                    self._steps[self._step + 1].pos[1],
                    ground_z,
                ]
            )
            des_pos, des_vel_s, des_acc_s = swing_reference(
                self._swing.start,
                foothold,
                s,
                swing_height=float(self.config.swing_height),
            )
            span = max(t_swing, 1e-3)
            self._swing.foothold = foothold
            self._swing.des_pos = des_pos
            self._swing.des_vel = des_vel_s / span
            self._swing.des_acc = des_acc_s / (span * span)
            self._swing.s = s
            swing = self._swing
            if leg == 0:
                left_phase = LegPhase.SWING
            else:
                right_phase = LegPhase.SWING
        else:
            self._swing = SwingState(active=False)

        return GaitOutput(
            contact_mask=mask,
            com_ref_xy=self._com_ref_output(com_xy),
            com_vel_ref=self._com_vel_ref.copy(),
            dcm_ref=dcm_ref.copy(),
            swing=swing,
            phase=self.phase,
            left_phase=left_phase,
            right_phase=right_phase,
            walking=True,
        )

    # -- MPC preview -------------------------------------------------------

    def preview(self, *, horizon: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Planned contact flags and foot poses over an MPC horizon.

        Returns ``(contact, foot_pose)`` with shapes ``(horizon, 2)`` bool and
        ``(horizon, 2, 3)``. ``foot_pose[..., 2]`` is clearance above the foot
        line (zero when planted, the Bézier lift while airborne). The centroidal
        MPC plans 0.3 s ahead, which spans most of a step, so replicating the
        *current* contact set across the horizon -- as this used to -- misstates
        who is even touching the ground. Airborne soles follow the swing arc
        rather than sitting at the foothold the WBC has not planted yet.
        """
        contact = np.ones((horizon, 2), dtype=bool)
        foot_pose = np.zeros((horizon, 2, 3))
        t_step, t_swing, t_ds = self._durations()
        height = float(self.config.swing_height)

        def _xy3(xy: np.ndarray) -> np.ndarray:
            pose = np.zeros(3)
            pose[:2] = xy
            return pose

        here = self._steps.get(self._step)
        if not self._active or t_step <= 0.0 or here is None or here.initial:
            # Standing, or still in the opening transfer: both feet stay where
            # they are for longer than the horizon reaches.
            fallback = self._initial_feet if self._active else np.zeros((2, 2))
            foot_pose[:, :, :2] = fallback
            return contact, foot_pose

        for k in range(horizon):
            # While waiting on a late touchdown the plant is still in this
            # step's single support; do not tell the MPC the swing foot is
            # already the new stance. k = 0 stays here, later ticks proceed
            # as if we land on the next control step.
            if self._touchdown_hold > 0.0:
                ahead = (t_step - 1e-6) + k * dt
            else:
                ahead = self._tau + (k + 1) * dt
            index = self._step + int(ahead // t_step)
            tau = ahead % t_step
            support = self._steps.get(index)
            landing = self._steps.get(index + 1)
            previous = self._steps.get(index - 1)
            if support is None or landing is None:
                foot_pose[k] = foot_pose[k - 1] if k else 0.0
                continue

            if support.initial:
                # Opening double support: both feet are where they started.
                foot_pose[k, :, :2] = self._initial_feet
                continue

            swing_leg = landing.leg
            foot_pose[k, support.leg, :2] = support.pos
            if tau >= t_ds and t_swing > 1e-6:
                contact[k, swing_leg] = False
                if previous is None:
                    start_xy = self._initial_feet[swing_leg]
                elif previous.initial:
                    start_xy = self._initial_feet[swing_leg]
                else:
                    start_xy = previous.pos
                s = float(np.clip((tau - t_ds) / t_swing, 0.0, 1.0))
                pos, _, _ = swing_reference(
                    _xy3(start_xy), _xy3(landing.pos), s, swing_height=height
                )
                foot_pose[k, swing_leg] = pos
            elif previous is None:
                foot_pose[k, swing_leg, :2] = landing.pos
            elif previous.initial:
                # The opening step carries no foothold of its own.
                foot_pose[k, swing_leg, :2] = self._initial_feet[swing_leg]
            else:
                # Double support: the swing leg is still on its last foothold,
                # which is where it stood while supporting the previous step.
                foot_pose[k, swing_leg, :2] = previous.pos
        return contact, foot_pose
