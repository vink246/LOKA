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

from loka.control.qp import QP, Sparsity

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

#: Per-step slew on latched timing and geometry. A mid-swing edit of these
#: jumps the Bézier parameter; they move only at a step boundary.
SLEW_PERIOD = 0.04  # s, on the full two-step cycle
SLEW_DUTY = 0.02
SLEW_WIDTH = 0.015  # m
SLEW_SWING_HEIGHT = 0.01  # m

#: Turn authority. ``max_turn_per_step`` shrinks this as speed rises so a
#: fast walk cannot yaw the next foothold out from under the stance foot.
TURN_CAP_STAND = 0.12  # rad / step at 0 m/s. 0.16 made turn-in-place fall; 0.08 was the previous cap.
TURN_CAP_FAST = 0.12  # rad / step at 0.50 m/s and above
DEFAULT_TURN_RATE = 0.40  # rad/s

#: Heading error past which forward speed is cut, and the crawl it is cut to.
HEADING_ERR_FREE = 0.35  # rad
HEADING_ERR_CRAWL = 0.85  # rad
HEADING_CRAWL_SPEED = 0.10  # m/s

#: Cadence follows a lagged copy of the commanded speed (wb_humanoid_mpc
#: ProceduralMpcMotionManager). Measured speed may delay a band change. It
#: must never pull the planned stride down: the chain still uses ``_cmd_speed``.
SPEED_HYSTERESIS = False  # no failing speed-step on this plant; left off until a bench shows a gain
SCHED_UP_LAG = 0.05  # m/s; schedule may sit this far above the measured speed
SCHED_DOWN_LAG = 0.05  # m/s; drop cadence only once measured speed is this close
SCHED_MIN_DWELL_STEPS = 2

#: While accelerating, hold the ramp if the body is behind it. Never decrease
#: the ramp because of lag (that would chase a slow measurement).
RAMP_FREEZE = False  # G1 already holds; freezing the ramp did not earn a bench gain on its own
RAMP_FREEZE_LAG = 0.08  # m/s

#: After a large heading error, stay on the crawl until the heading and the
#: measured speed have both settled. Leaving early is the post-turn fall.
TURN_SETTLE = True
TURN_EXIT_ERR = 0.10  # rad
TURN_SETTLE_STEPS = 2

#: Pelvis yaw moves during double support and holds in single support, so one
#: planted sole is not asked to twist (wb_humanoid_mpc weights foot yaw moment
#: and hip-yaw far above the other leg joints).
YAW_IN_DS_ONLY = False  # ruled out: T1 (a +0.3 rad turn) fell at 6.3 s; see walking.md
YAW_DS_MIN = 0.05  # s; a shorter transfer keeps the full-step yaw ramp

#: Swing height profile. ``sine`` is the shipped half-sine. ``spline`` is the
#: wb_humanoid_mpc touchdown: small lift-off speed, soft landing, 1 mm below
#: the foot line. Kept on sine unless a bench shows the impact is the fall.
SWING_PROFILE = "sine"
SWING_LIFT_VZ = 0.05  # m/s at lift-off, spline profile
SWING_TOUCHDOWN_VZ = 0.10  # m/s downward at landing, spline profile
SWING_TOUCHDOWN_Z = -0.001  # m, spline profile
SWING_TIME_SCALE = 0.20  # s; shorter swings shrink height (spline profile)

#: Step duration as a decision beside the foothold (Khadiv et al., TRO 2020).
#: The 0.02 / 0.1 / 0.5 weight sweep walked S0 backwards and added falls
#: (temp_docs sweep, 2026-09-28), so this stays off.
STEP_TIMING = False
STEP_TIMING_WEIGHT = 0.1  # cost on (tau - tau_nom)^2 beside the foothold/offset costs
STEP_DS_MIN = 0.05  # s; shortest double support left after an early lift-off
STEP_SWING_RATE_MAX = 1.25  # fastest swing clock; v1 only ever shortens a step

#: Extra steps after a stop request before the feet-square test loosens.
STOP_RELAX_STEPS = 4


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
    duty_factor: float = 0.80
    step_length_max: float = 0.30  # cap on stride between consecutive footholds [m]
    swing_height: float = 0.045  # peak clearance above the foot line [m]
    stance_width: float = 0.24  # lateral foot separation [m]; nominal G1 is 0.237
    #: Fraction of full DCM-offset foothold placement. 1.0 is full (deadbeat):
    #: the end-of-step DCM offset matches the plan in one step. 0.7 is the
    #: measured bench gain (0 falls). It drifts sideways on the 30 s
    #: limit-cycle test; that bound is recorded, not used to pick a lower gain.
    capture_gain: float = 0.7
    walk_accel: float = 0.5  # ramp on the commanded speed [m/s²]
    #: Fraction of swing during which the foothold may still be retargeted.
    foothold_retarget_s: float = 0.70
    #: How fast the plan may yaw toward ``heading`` [rad/s]. The heading
    #: itself is a goal; the feet turn by at most one step's worth per step.
    turn_rate: float = DEFAULT_TURN_RATE


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


# Closed-loop 6 s straight walk at 0.25 m/s, one knob off the speed schedule
# at a time (2026-09-24). Outside this band the robot falls in a few seconds:
# speed ≥ 0.40, duty ≤ 0.74, period 0.45, swing ≥ 0.10, width 0.32, capture 0.
# Dashboard / yaml defaults that survived sit inside.
GAIT_KNOBS: tuple[GaitKnob, ...] = (
    GaitKnob("gait.mode", "0=stand 1=walk 2=tread (march in place) 3=limp", 0.0, 3.0),
    GaitKnob("gait.speed", "Travel speed along the heading [m/s]. Straight walk holds through 0.30",
             0.0, 0.30),
    GaitKnob("gait.heading",
             "Goal travel direction [rad], world. Rate-limited by turn_rate; "
             "a large error cuts speed to a crawl",
             -np.pi, np.pi),
    GaitKnob("gait.step_period",
             "Full two-step cycle [s]; one step is half of this. 0.45 falls",
             0.55, 1.10),
    GaitKnob("gait.duty_factor",
             "Stance fraction per leg. At 0.25 m/s, 0.74 falls and 0.78 holds",
             0.78, 0.85),
    GaitKnob("gait.step_length_max",
             "Cap on the stride between consecutive footholds [m]", 0.08, 0.40),
    GaitKnob("gait.swing_height",
             "Peak swing clearance [m]. 0.08 holds; 0.10 falls",
             0.02, 0.08),
    GaitKnob("gait.stance_width",
             "Lateral foot separation [m]; nominal G1 is 0.237. 0.32 falls",
             0.18, 0.28),
    GaitKnob("gait.capture_gain",
             "DCM foothold feedback, fraction of full placement. 0 falls; 0.7 default; 1.0 is deadbeat",
             0.30, 1.00),
    GaitKnob("gait.walk_accel", "Ramp on the commanded speed [m/s^2]", 0.15, 1.60),
    GaitKnob("gait.foothold_retarget_s",
             "Fraction of swing during which the foothold may still move",
             0.20, 0.90),
    GaitKnob("gait.turn_rate",
             "How fast the plan yaws toward gait.heading [rad/s]. Straight "
             "walk did not bind this; the per-step cap still applies",
             0.10, 0.80),
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


def wrap_angle(angle: float) -> float:
    """Shortest signed angle, in ``(-pi, pi]``."""
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def circ_mid(a: float, b: float) -> float:
    """Midpoint of the short arc from ``a`` to ``b``."""
    return float(a) + 0.5 * wrap_angle(float(b) - float(a))


def heading_frame(heading: float) -> tuple[np.ndarray, np.ndarray]:
    """Unit ``(forward, left)`` vectors of the travel frame, world xy."""
    c, s = np.cos(heading), np.sin(heading)
    return np.array([c, s]), np.array([-s, c])


def _side(leg: int) -> float:
    """+1 for the left foot, −1 for the right, along ``heading_frame`` left."""
    return 1.0 if int(leg) == 0 else -1.0


def max_turn_per_step(speed: float, t_step: float, turn_rate: float) -> float:
    """Largest yaw change one step may commit [rad]."""
    t = float(np.clip(abs(float(speed)) / 0.50, 0.0, 1.0))
    cap = TURN_CAP_STAND * (1.0 - t) + TURN_CAP_FAST * t
    return float(min(abs(float(turn_rate)) * max(float(t_step), 0.0), cap))


def speed_cap_for_heading_error(err: float, speed: float) -> float:
    """Cut forward speed while the heading error is still large."""
    gap = abs(float(err))
    speed = float(speed)
    if gap <= HEADING_ERR_FREE:
        return speed
    crawl = min(speed, HEADING_CRAWL_SPEED)
    if gap >= HEADING_ERR_CRAWL:
        return crawl
    u = (gap - HEADING_ERR_FREE) / (HEADING_ERR_CRAWL - HEADING_ERR_FREE)
    return speed * (1.0 - u) + crawl * u


def _slew(current: float, goal: float, limit: float) -> float:
    delta = float(goal) - float(current)
    if abs(delta) <= limit:
        return float(goal)
    return float(current) + np.sign(delta) * float(limit)


def _swing_lift(s: float, swing_height: float, t_swing: float | None, profile: str):
    """Vertical lift and its ``d/ds`` derivatives for one swing profile."""
    if profile != "spline":
        lift = np.sin(np.pi * s)
        return (
            swing_height * lift,
            swing_height * np.pi * np.cos(np.pi * s),
            swing_height * (-np.pi * np.pi * np.sin(np.pi * s)),
        )
    duration = 0.15 if t_swing is None else max(float(t_swing), 1e-3)
    height = float(swing_height) * min(1.0, duration / SWING_TIME_SCALE)
    # dz/ds = vz * dt/ds, and s runs 0→1 over the swing, so dt/ds = duration.
    v0 = SWING_LIFT_VZ * duration
    v1 = -SWING_TOUCHDOWN_VZ * duration
    z1 = SWING_TOUCHDOWN_Z
    # z = a6 s^6 + a5 s^5 + a4 s^4 + a3 s^3 + v0 s
    # z(0)=0, z'(0)=v0, z''(0)=0 are built in. Remaining rows are
    # z(1), z'(1), z''(1), z(0.5).
    rows = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [6.0, 5.0, 4.0, 3.0],
        [30.0, 20.0, 12.0, 6.0],
        [1.0 / 64.0, 1.0 / 32.0, 1.0 / 16.0, 1.0 / 8.0],
    ])
    rhs = np.array([z1 - v0, v1 - v0, 0.0, height - 0.5 * v0])
    a6, a5, a4, a3 = np.linalg.solve(rows, rhs)
    s2, s3, s4, s5, s6 = s * s, s ** 3, s ** 4, s ** 5, s ** 6
    z = a6 * s6 + a5 * s5 + a4 * s4 + a3 * s3 + v0 * s
    dz = 6 * a6 * s5 + 5 * a5 * s4 + 4 * a4 * s3 + 3 * a3 * s2 + v0
    ddz = 30 * a6 * s4 + 20 * a5 * s3 + 12 * a4 * s2 + 6 * a3 * s
    return float(z), float(dz), float(ddz)


def swing_reference(
    start: np.ndarray,
    end: np.ndarray,
    s: float,
    *,
    swing_height: float,
    t_swing: float | None = None,
    profile: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Swing-arc position and its first two ``d/ds`` derivatives.

    Horizontal motion follows a smoothstep, whose slope vanishes at both ends,
    so the foot neither jerks off the ground at lift-off nor scuffs forward as
    it lands. Vertical motion is a half sine (``SWING_PROFILE == "sine"``),
    which peaks at mid-swing and arrives with a downward velocity that seats
    the contact. ``"spline"`` is the softer wb_humanoid_mpc touchdown.

    Both components are analytic, so the caller gets velocity *and* acceleration
    feedforward for free. That matters more than the tracking gains: asked to
    synthesise a 0.25 s arc from position error alone, any sane gain saturates,
    and the task degenerates into bang-bang.
    """
    s = float(np.clip(s, 0.0, 1.0))
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    which = SWING_PROFILE if profile is None else profile

    shape = s * s * (3.0 - 2.0 * s)
    dshape = 6.0 * s * (1.0 - s)
    ddshape = 6.0 - 12.0 * s
    lift, dlift, ddlift = _swing_lift(s, swing_height, t_swing, which)

    delta = end - start
    pos = start + delta * shape
    vel = delta * dshape
    acc = delta * ddshape
    pos[2] += lift
    vel[2] += dlift
    acc[2] += ddlift
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
    yaw: float = 0.0
    nominal_yaw: float | None = None
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
    start_yaw: float = 0.0
    foothold_yaw: float = 0.0
    des_yaw: float = 0.0
    des_yaw_rate: float = 0.0


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
    yaw_ref: float = 0.0
    yaw_rate_ref: float = 0.0
    stance_yaw: float = 0.0


class GaitScheduler:
    """Periodic bipedal gait: footstep plan, DCM reference and swing arcs."""

    def __init__(self, config: GaitConfig | None = None) -> None:
        self.config = config or GaitConfig()
        #: ``step_period`` / ``duty_factor`` / ``stance_width`` the operator or
        #: LLM set explicitly. The speed schedule fills the rest.
        self._user_gait_fields: set[str] = set()
        self._use_speed_schedule = False
        self._step_qp = QP(
            "step_adjust",
            hessian_pattern=Sparsity.diagonal(5),
            constraint_pattern=Sparsity.dense(9, 5),
        )
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
        self._capture_hot = False
        self._stop_requested = False
        self._stop_extra = 0
        self._stand_yaw = 0.0
        self._initial_yaws = np.zeros(2)
        self._latch_valid = False
        self._t_step = 0.35
        self._t_swing = 0.25
        self._t_ds = 0.10
        self._width = 0.24
        self._swing_height = 0.045
        self._sched_speed = 0.0
        self._meas_speed = 0.0
        self._sched_dwell = 0
        self._cleared_once = False
        self._turning = False
        self._turn_settle_count = 0
        self._ds_rate = 1.0
        self._swing_rate = 1.0

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
        if self._stop_requested and self._active:
            return True
        if float(cfg.mode) < MODE_WALK:
            return False
        if float(cfg.mode) == MODE_TREAD:
            return True
        if float(cfg.speed) > 1e-4 or self._cmd_speed > 1e-4 or self._active:
            return True
        # Turn in place: a heading goal with no travel speed still steps.
        stance = self._stance_yaw()
        return abs(wrap_angle(float(cfg.heading) - stance)) > 0.05

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
                if self._active:
                    # Finish the step in the air. Resetting here drops the
                    # swing foot and the next tick names it stance.
                    self._stop_requested = True
                else:
                    self.reset()
        self._use_speed_schedule = float(self.config.mode) >= MODE_WALK
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

    def _config_durations(self, period: float, duty: float) -> tuple[float, float, float]:
        period = max(0.20, float(period))
        duty = float(np.clip(duty, 0.55, 0.85))
        t_step = 0.5 * period
        t_swing = (1.0 - duty) * period
        return t_step, t_swing, max(0.0, t_step - t_swing)

    def _goal_geometry(self) -> tuple[float, float, float, float]:
        """``(period, duty, width, swing_height)`` the latch slews toward.

        Unpinned cadence follows the *ramped* speed, so a speed command does
        not retune the step that is already in the air. Limp is a conservative
        preset and does not count as a user pin. Tread uses the crawl schedule.
        """
        cfg = self.config
        period = float(cfg.step_period)
        duty = float(cfg.duty_factor)
        width = float(cfg.stance_width)
        height = float(cfg.swing_height)
        if float(cfg.mode) == MODE_LIMP:
            return 0.90, 0.85, width, 0.035
        if float(cfg.mode) == MODE_TREAD:
            schedule_speed = 0.0
        elif SPEED_HYSTERESIS:
            schedule_speed = float(self._sched_speed)
        else:
            schedule_speed = float(self._cmd_speed)
        if TURN_SETTLE and self._turning:
            schedule_speed = min(schedule_speed, HEADING_CRAWL_SPEED)
        if self._use_speed_schedule and float(cfg.mode) >= MODE_WALK:
            # A turn in place has no ramped speed; it still needs the crawl
            # cadence. Duty 0.65 (the isolation default) falls in ~3 s.
            scheduled = gait_schedule_for_speed(schedule_speed)
            if "step_period" not in self._user_gait_fields:
                period = scheduled["gait.step_period"]
            if "duty_factor" not in self._user_gait_fields:
                duty = scheduled["gait.duty_factor"]
            if "stance_width" not in self._user_gait_fields:
                width = scheduled["gait.stance_width"]
        return period, duty, width, height

    def _latch_step_params(self, *, exact: bool) -> None:
        """Copy or slew timing and geometry. Call only at a step boundary."""
        period, duty, width, height = self._goal_geometry()
        if not exact and self._latch_valid:
            period = _slew(self._t_step * 2.0, period, SLEW_PERIOD)
            duty = _slew(1.0 - self._t_swing / max(2.0 * self._t_step, 1e-3), duty, SLEW_DUTY)
            width = _slew(self._width, width, SLEW_WIDTH)
            height = _slew(self._swing_height, height, SLEW_SWING_HEIGHT)
        self._t_step, self._t_swing, self._t_ds = self._config_durations(period, duty)
        self._width = float(width)
        self._swing_height = float(height)
        self._latch_valid = True

    def _durations(self) -> tuple[float, float, float]:
        """``(T_step, T_swing, T_ds)`` for the step in progress [s].

        A step is half a cycle. Each leg swings for ``(1 - duty)`` of the cycle,
        leaving ``(duty - 0.5)`` of double support on either side of it. While
        a walk is active these are the latched values, not the live goal.
        """
        if self._latch_valid:
            return self._t_step, self._t_swing, self._t_ds
        period, duty, _, _ = self._goal_geometry()
        return self._config_durations(period, duty)

    def _yaw_now(self, t_step: float) -> tuple[float, float]:
        """Pelvis yaw reference and rate across the current step.

        Smoothstep between the mid-yaw of the previous transfer and the
        mid-yaw of the upcoming one, so the torso leads the feet without a
        step in the reference. Rate is zero while waiting on a late plant.
        """
        span = self._current_duration(t_step)
        _, _, t_ds = self._durations()
        here = self._steps.get(self._step)
        use_ds = (
            YAW_IN_DS_ONLY
            and here is not None
            and not here.initial
            and t_ds >= YAW_DS_MIN
        )
        if use_ds:
            u = 0.0 if t_ds <= 0.0 else float(np.clip(self._tau / t_ds, 0.0, 1.0))
        else:
            u = 0.0 if span <= 0.0 else float(np.clip(self._tau / span, 0.0, 1.0))
        landing = self._steps.get(self._step + 1)
        prev = self._steps.get(self._step - 1)
        yaw_here = float(here.yaw) if here is not None else 0.0
        yaw_land = float(landing.yaw) if landing is not None else yaw_here
        yaw_prev = float(prev.yaw) if prev is not None else yaw_here
        a = circ_mid(yaw_prev, yaw_here)
        b = circ_mid(yaw_here, yaw_land)
        shape = u * u * (3.0 - 2.0 * u)
        dshape = 6.0 * u * (1.0 - u)
        yaw_ref = a + wrap_angle(b - a) * shape
        ramp = t_ds if use_ds else span
        if self._touchdown_hold > 0.0 or ramp <= 0.0:
            rate = 0.0
        else:
            rate = wrap_angle(b - a) * dshape / ramp
        return float(yaw_ref), float(rate)

    def _stance_yaw(self) -> float:
        step = self._steps.get(self._step) if self._steps else None
        if step is not None:
            return float(step.yaw)
        return 0.0

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

    def _update_turning(self, *, at_boundary: bool) -> None:
        """Hold the crawl after a large heading error until the body settles."""
        if not TURN_SETTLE or self._stop_requested:
            self._turning = False
            self._turn_settle_count = 0
            return
        err = abs(wrap_angle(float(self.config.heading) - self._stance_yaw()))
        if err > HEADING_ERR_FREE:
            self._turning = True
            self._turn_settle_count = 0
            return
        if not self._turning or not at_boundary:
            return
        settled = (
            err < TURN_EXIT_ERR
            and abs(self._meas_speed) <= HEADING_CRAWL_SPEED + 0.05
        )
        if settled:
            self._turn_settle_count += 1
            if self._turn_settle_count >= TURN_SETTLE_STEPS:
                self._turning = False
                self._turn_settle_count = 0
        else:
            self._turn_settle_count = 0

    def _update_sched_speed(self) -> None:
        """Move the cadence speed toward the command, gated on the measurement.

        An upward step stops at ``measured + SCHED_UP_LAG``, so a 0 → 0.5 m/s
        command cannot retune duty while the robot is still standing. A
        downward step waits until the measured speed has actually fallen.
        """
        if not SPEED_HYSTERESIS:
            self._sched_speed = float(self._cmd_speed)
            return
        target = float(self._cmd_speed)
        if self._turning:
            target = min(target, HEADING_CRAWL_SPEED)
            if self._sched_speed > HEADING_CRAWL_SPEED + 1e-4:
                self._sched_speed = HEADING_CRAWL_SPEED
                self._sched_dwell = 0
        immediate = self._stop_requested or float(self.config.mode) == MODE_TREAD
        if immediate and target < self._sched_speed - 1e-4:
            self._sched_speed = target
            self._sched_dwell = 0
            return
        if self._capture_hot:
            return
        self._sched_dwell += 1
        if self._sched_dwell < SCHED_MIN_DWELL_STEPS:
            return
        meas = float(self._meas_speed)
        new = self._sched_speed
        if target > self._sched_speed + 1e-4 and meas >= self._sched_speed - SCHED_UP_LAG:
            ceiling = max(self._sched_speed, meas + SCHED_UP_LAG)
            new = min(target, ceiling)
        elif target < self._sched_speed - 1e-4 and meas <= target + SCHED_DOWN_LAG:
            new = target
        if abs(new - self._sched_speed) > 1e-4:
            self._sched_speed = float(new)
            self._sched_dwell = 0

    def _velocity_command(self, dt: float) -> float:
        """Ramped scalar speed [m/s], already cut for heading error and limp."""
        cfg = self.config
        target = float(cfg.speed)
        if cfg.mode == MODE_TREAD or self._stop_requested:
            target = 0.0
        elif cfg.mode == MODE_LIMP:
            target = min(target, 0.15)
        if not self._stop_requested:
            err = wrap_angle(float(cfg.heading) - self._stance_yaw())
            target = speed_cap_for_heading_error(err, target)
        if TURN_SETTLE and self._turning:
            target = min(target, HEADING_CRAWL_SPEED)

        rate = float(cfg.walk_accel) * dt
        if self._cmd_speed < target:
            # Isolation never clears a swing (z stays 0), so the freeze stays
            # off and verify_gait's perfect tracker is unchanged.
            lagging = (
                RAMP_FREEZE
                and self._cleared_once
                and self._meas_speed < self._cmd_speed - RAMP_FREEZE_LAG
            )
            if not lagging:
                self._cmd_speed = min(target, self._cmd_speed + rate)
        else:
            # Decelerate faster than we accelerate; stopping is the safe way to
            # be wrong about how much authority is left.
            self._cmd_speed = max(target, self._cmd_speed - 2.0 * rate)
        return float(self._cmd_speed)

    # -- footstep plan -----------------------------------------------------

    def _leg_of(self, index: int) -> int:
        """Which leg supports step ``index``."""
        if index <= 0:
            return -1
        return (self._first_leg + index - 1) % 2

    def _heading_goal(self) -> float:
        """Yaw the chain walks toward. Frozen while a stop is in progress."""
        if self._stop_requested:
            return self._stance_yaw()
        return float(self.config.heading)

    def _refresh_plan(self, *, t_step: float, speed: float) -> None:
        """Rebuild the unfrozen tail of the footstep chain.

        The chain is a centre line. Each step advances ``speed × T_step``
        along the yaw *of that step* and yaws toward the heading goal by at
        most :func:`max_turn_per_step`. Feet sit ``stance_width / 2`` either
        side of the centre, so a heading change cannot plant the next foot
        behind the stance foot. At zero yaw this is the old
        ``previous + stride + side * width`` chain.
        """
        width = float(self._width if self._latch_valid else self.config.stance_width)
        goal = self._heading_goal()
        turn = max_turn_per_step(speed, t_step, float(self.config.turn_rate))
        support = self._steps[self._step]
        yaw = float(support.yaw)
        center = np.asarray(support.pos, dtype=float) - (
            _side(support.leg) * 0.5 * width * heading_frame(yaw)[1]
            if support.leg >= 0
            else 0.0
        )

        for index in range(self._step + 1, self._step + PLAN_HORIZON + 1):
            existing = self._steps.get(index)
            if existing is not None and existing.frozen:
                yaw = float(existing.yaw)
                if existing.leg >= 0:
                    center = np.asarray(existing.pos, dtype=float) - (
                        _side(existing.leg) * 0.5 * width * heading_frame(yaw)[1]
                    )
                continue
            dyaw = float(np.clip(wrap_angle(goal - yaw), -turn, turn))
            forward, left = heading_frame(yaw + 0.5 * dyaw)
            center = center + speed * t_step * forward
            yaw = yaw + dyaw
            leg = self._leg_of(index)
            nominal = center + _side(leg) * 0.5 * width * left
            if existing is None:
                self._steps[index] = Footstep(
                    leg=leg,
                    pos=nominal.copy(),
                    nominal=nominal.copy(),
                    yaw=yaw,
                    nominal_yaw=yaw,
                )
            else:
                existing.nominal = nominal
                existing.nominal_yaw = yaw
                existing.pos = nominal + existing.correction
                existing.yaw = yaw
            # The following step chains off where this foot actually is,
            # correction included. Leaving it on the geometric centre makes
            # the end-of-step DCM ignore the correction and the foothold
            # ratchets back onto the liftoff foot.
            center = np.asarray(self._steps[index].pos, dtype=float) - (
                _side(leg) * 0.5 * width * heading_frame(yaw)[1]
            )
            yaw = float(self._steps[index].yaw)

        for index in list(self._steps):
            if index < self._step - 1:
                del self._steps[index]

    def _tau_ahead(self, real_dt: float, *, t_ds: float, span: float) -> float:
        """Plan-time reached after ``real_dt`` seconds at the timing rates.

        Double support runs at ``_ds_rate``, the swing at ``_swing_rate``, and
        anything past this step at 1. Later steps are not shortened.
        """
        left = float(real_dt)
        tau = float(self._tau)
        while left > 1e-12:
            if tau < t_ds:
                rate = max(float(self._ds_rate), 1e-6)
                room = t_ds - tau
            elif tau < span:
                rate = max(float(self._swing_rate), 1e-6)
                room = span - tau
            else:
                return tau + left
            step = min(left, room / rate)
            tau += step * rate
            left -= step
        return tau

    def _step_adjust(
        self,
        *,
        dcm: np.ndarray,
        s: float,
        omega: float,
        t_step: float,
        t_swing: float,
        t_ds: float,
        in_swing: bool,
    ) -> None:
        """Foothold and step duration from one 5-variable DCM QP.

        With the step duration fixed, the solution is the offset foothold law
        in :meth:`_apply_dcm_correction`. Timing, when allowed, only shortens
        the step: double support first, then the swing clock, never below
        :data:`STEP_DS_MIN` of double support or slower than
        :data:`STEP_SWING_RATE_MAX`.
        """
        cfg = self.config
        target = self._steps.get(self._step + 1)
        support = self._steps.get(self._step)
        if (
            target is None
            or target.nominal is None
            or support is None
            or support.initial
        ):
            self._ds_rate = 1.0
            self._swing_rate = 1.0
            return
        if float(cfg.capture_gain) <= 1e-6:
            target.correction = np.zeros(2)
            target.pos = np.asarray(target.nominal, dtype=float).copy()
            self._ds_rate = 1.0
            self._swing_rate = 1.0
            return
        if in_swing and s > float(np.clip(cfg.foothold_retarget_s, 0.05, 0.95)):
            target.frozen = True
            return
        if target.frozen:
            return

        span = self._current_duration(t_step)
        t_nom = max(span - self._tau, 1e-3)
        ds_rem = max(t_ds - self._tau, 0.0)
        sw_rem = span - max(self._tau, t_ds)
        ds_floor = min(ds_rem, STEP_DS_MIN)
        t_min = ds_floor + sw_rem / STEP_SWING_RATE_MAX
        timing = (
            STEP_TIMING
            and self._cleared_once
            and not self._stop_requested
            and self._touchdown_hold == 0.0
        )
        if not timing:
            t_min = t_nom
        t_min = min(t_min, t_nom)

        zmp0 = self._zmp_of(support)
        d = np.asarray(dcm, dtype=float) - zmp0
        goal = self._dcm_eos(omega=omega, t_step=t_step)
        target_zmp = self._zmp_of(target)
        b_nom = goal - target_zmp
        inset = target_zmp - np.asarray(target.pos, dtype=float)
        u_nom = np.asarray(target.nominal, dtype=float) + inset
        tau_nom = float(np.exp(omega * t_nom))
        tau_min = float(np.exp(omega * t_min))

        k = float(np.clip(cfg.capture_gain, 0.0, 1.0))
        weight = np.maximum(
            np.array([1.0 - k, 1.0 - k, STEP_TIMING_WEIGHT, k, k], dtype=float),
            1e-3,
        )
        hessian = np.diag(weight)
        gradient = -weight * np.array(
            [u_nom[0], u_nom[1], tau_nom, b_nom[0], b_nom[1]], dtype=float
        )
        forward, left = heading_frame(float(support.yaw))
        side = 1.0 if target.leg == 0 else -1.0
        anchor = np.asarray(support.pos, dtype=float) + inset
        here = np.asarray(target.pos, dtype=float) + inset
        constraint = np.zeros((9, 5))
        lower = np.zeros(9)
        upper = np.zeros(9)
        # u + b - d * tau = zmp0
        constraint[0, 0] = 1.0
        constraint[0, 2] = -d[0]
        constraint[0, 3] = 1.0
        constraint[1, 1] = 1.0
        constraint[1, 2] = -d[1]
        constraint[1, 4] = 1.0
        lower[0] = upper[0] = zmp0[0]
        lower[1] = upper[1] = zmp0[1]
        # Correction box and stride, in the stance heading frame.
        for row, axis, centre, half in (
            (2, forward, u_nom, MAX_DCM_CORRECTION),
            (3, left, u_nom, MAX_DCM_CORRECTION),
            (4, forward, anchor, float(cfg.step_length_max)),
        ):
            constraint[row, 0] = axis[0]
            constraint[row, 1] = axis[1]
            mid = float(np.dot(axis, centre))
            lower[row] = mid - half
            upper[row] = mid + half
        # Lateral gap. side * left · (u - anchor) >= MIN_FOOT_SEPARATION.
        constraint[5, 0] = side * left[0]
        constraint[5, 1] = side * left[1]
        lower[5] = MIN_FOOT_SEPARATION + side * float(np.dot(left, anchor))
        upper[5] = 1e3
        budget = (
            SWING_CATCHUP_SPEED * max((1.0 - s) * t_swing, 1e-3)
            if in_swing and s > 0.45
            else 1e3
        )
        for row, axis in ((6, forward), (7, left)):
            constraint[row, 0] = axis[0]
            constraint[row, 1] = axis[1]
            mid = float(np.dot(axis, here))
            lower[row] = mid - budget
            upper[row] = mid + budget
        constraint[8, 2] = 1.0
        lower[8] = tau_min
        upper[8] = tau_nom

        solution = self._step_qp.solve(hessian, gradient, constraint, lower, upper)
        if self._step_qp.last_status != "solved":
            if in_swing:
                self._apply_dcm_correction(dcm=dcm, s=s, omega=omega, t_step=t_step)
            self._ds_rate = 1.0
            self._swing_rate = 1.0
            return

        pos = solution[:2] - inset
        target.correction = pos - np.asarray(target.nominal, dtype=float)
        target.pos = pos
        duration = float(np.clip(np.log(max(solution[2], 1.0)) / omega, t_min, t_nom))
        cut = t_nom - duration
        if cut <= ds_rem - ds_floor + 1e-9:
            self._ds_rate = (
                1.0 if ds_rem <= 1e-9 else ds_rem / max(ds_rem - cut, 1e-6)
            )
            self._swing_rate = 1.0
        else:
            self._ds_rate = 1.0 if ds_rem <= 1e-9 else ds_rem / max(ds_floor, 1e-6)
            self._swing_rate = float(np.clip(
                sw_rem / max(duration - ds_floor, 1e-6), 1.0, STEP_SWING_RATE_MAX
            ))
        self._ds_rate = max(self._ds_rate, 1.0)
        residual = constraint @ solution
        boxed = np.any(np.minimum(residual[2:8] - lower[2:8], upper[2:8] - residual[2:8]) < 1e-5)
        self._capture_hot = bool(boxed or (timing and solution[2] <= tau_min + 1e-9))

    def _apply_dcm_correction(
        self, *, dcm: np.ndarray, s: float, omega: float, t_step: float
    ) -> None:
        """Nudge the in-flight foothold to absorb DCM error.

        The error is evaluated at *touchdown*, not now: the measured DCM is
        propagated over the remaining step time under the support foot.
        Compared with the planned end-of-step DCM itself, that error moves
        with the correction, because the tail is chained off the corrected
        foothold: each tick would compute ``c = k (e - c_prev)`` and settle
        at ``k/(1+k)`` of the error (and oscillate for ``k >= 1``). The DCM
        *offset* past the target foothold does not move, so the error is
        measured against that. ``capture_gain`` is the fraction of full
        placement; 1.0 makes the offset match the plan in one step.

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
        # The tail is chained off target.pos, so goal moves with the correction.
        # The DCM offset past the target does not; measure the error against it.
        target_zmp = self._zmp_of(target)
        offset = goal - target_zmp
        inset = target_zmp - np.asarray(target.pos, dtype=float)
        want = predicted - offset - inset
        correction = float(cfg.capture_gain) * (want - target.nominal)
        norm = float(np.linalg.norm(correction))
        saturated = norm > MAX_DCM_CORRECTION
        if saturated:
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
            forward, left = heading_frame(float(support.yaw))
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
                saturated = True
            if abs(excess) > 1e-6:
                saturated = True
        self._capture_hot = bool(saturated)
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
        return pos - _side(step.leg) * ZMP_INSET * heading_frame(float(step.yaw))[1]

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

    def _begin(self, *, feet_xy: np.ndarray, com_xy: np.ndarray, foot_yaws: np.ndarray) -> None:
        """Seed the plan from the current stance."""
        stance = 1  # right supports first, so the left steps out
        swing = 1 - stance
        yaw = float(foot_yaws[stance])
        self._com_ref = com_xy.copy()
        self._com_vel_ref = np.zeros(2)
        self._step = 0
        self._tau = 0.0
        self._swing = SwingState()
        self._initial_feet = feet_xy.copy()
        self._initial_yaws = np.asarray(foot_yaws, dtype=float).reshape(2).copy()
        # Opening contact is still both feet. The DCM ZMP slides from the
        # midpoint onto the upcoming swing-foot inset (APA); the stored
        # footstep is the stance so the chain after it is coherent.
        left = heading_frame(yaw)[1]
        swing_inset = feet_xy[swing] - _side(swing) * ZMP_INSET * left
        self._opening_zmp = swing_inset
        self._steps = {
            0: Footstep(
                leg=stance, pos=feet_xy[stance].copy(), yaw=yaw,
                frozen=True, initial=True,
            ),
            1: Footstep(
                leg=stance, pos=feet_xy[stance].copy(), yaw=yaw, frozen=True,
            ),
        }
        self._first_leg = stance
        self._active = True
        self._touchdown_hold = 0.0
        self._swing_cleared = False
        self._stop_extra = 0
        self._latch_step_params(exact=True)

    def _may_stop(self, *, t_step: float) -> bool:
        """True once halting would leave the feet square and level.

        Called only at a step boundary, so stopping always completes the step
        in progress rather than freezing a leg in mid-air.
        """
        if float(self.config.mode) == MODE_TREAD and not self._stop_requested:
            return False
        if self._cmd_speed > 1e-3:
            return False
        here = self._steps.get(self._step)
        nxt = self._steps.get(self._step + 1)
        if here is None or nxt is None:
            return False
        if not self._stop_requested:
            if abs(wrap_angle(self._heading_goal() - float(here.yaw))) > 0.03:
                return False
        relaxed = self._stop_extra >= STOP_RELAX_STEPS
        stride_tol = 0.06 if relaxed else 0.02
        yaw_tol = 0.06 if relaxed else 0.03
        forward, _ = heading_frame(float(here.yaw))
        stride_ok = abs(float(np.dot(nxt.pos - here.pos, forward))) < stride_tol
        yaw_ok = abs(wrap_angle(float(nxt.yaw) - float(here.yaw))) < yaw_tol
        if stride_ok and yaw_ok:
            return True
        if self._stop_requested:
            self._stop_extra += 1
        return False

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
            yaw_ref=float(getattr(self, "_stand_yaw", 0.0)),
            yaw_rate_ref=0.0,
            stance_yaw=float(getattr(self, "_stand_yaw", 0.0)),
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
        foot_yaws: np.ndarray | None = None,
    ) -> GaitOutput:
        com_xy = np.asarray(com[:2], dtype=float)
        com_vel_xy = np.asarray(com_vel[:2], dtype=float)
        feet = np.asarray(foot_centers, dtype=float).reshape(2, 3)
        feet_xy = feet[:, :2]
        if foot_yaws is None:
            yaws = np.zeros(2)
        else:
            yaws = np.asarray(foot_yaws, dtype=float).reshape(2)
        self._stand_yaw = float(circ_mid(yaws[0], yaws[1]))
        forward_h, _ = heading_frame(self._stance_yaw())
        self._meas_speed = float(np.dot(com_vel_xy, forward_h))
        self._update_turning(at_boundary=False)

        if not self.wants_walk():
            if self._active:
                yaw = self._stance_yaw()
                self.reset()
                self._stand_yaw = yaw
            return self._stand_output(measured_mask, feet_xy)

        if not self._active:
            self._begin(feet_xy=feet_xy, com_xy=com_xy, foot_yaws=yaws)

        t_step, t_swing, t_ds = self._durations()
        speed = self._velocity_command(dt)
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
        # A saturated capture correction means the next step is late. Advance
        # the swing clock up to 1.5×, but never below a 0.2 s swing. With step
        # timing on, the QP's rates replace that speed-up.
        clock = dt
        if STEP_TIMING:
            clock *= self._ds_rate if self._tau < t_ds else self._swing_rate
        elif self._capture_hot and t_swing > 0.20:
            clock *= min(1.5, t_swing / 0.20)
        self._tau += clock
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
                    yaw = self._stance_yaw()
                    self.reset()
                    self._stand_yaw = yaw
                    return self._stand_output(measured_mask, feet_xy)
                if cleared:
                    self._cleared_once = True
                self._update_turning(at_boundary=True)
                self._update_sched_speed()
                self._latch_step_params(exact=False)
                t_step, t_swing, t_ds = self._durations()
                support = self._steps.get(self._step)
                if support is not None:
                    if support.leg >= 0 and cleared:
                        support.pos = feet_xy[support.leg].copy()
                        if support.nominal is not None:
                            support.correction = support.pos - support.nominal
                        support.yaw = float(yaws[support.leg])
                    support.frozen = True
                self._swing = SwingState()
                self._ds_rate = 1.0
                self._swing_rate = 1.0
        else:
            self._touchdown_hold = 0.0

        self._refresh_plan(t_step=t_step, speed=speed)

        # -- swing window ---------------------------------------------------
        support = self._steps[self._step]
        in_swing = (not support.initial) and self._tau >= t_ds and t_swing > 1e-6
        s = float(np.clip((self._tau - t_ds) / t_swing, 0.0, 1.0)) if in_swing else 0.0

        # -- DCM reference, then bounded foothold feedback -------------------
        dcm = com_xy + com_vel_xy / omega
        if STEP_TIMING:
            self._step_adjust(
                dcm=dcm, s=s, omega=omega, t_step=t_step,
                t_swing=t_swing, t_ds=t_ds, in_swing=in_swing,
            )
        elif in_swing:
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
            landing = self._steps[self._step + 1]
            if not self._swing.active or self._swing.leg != leg:
                # Lift off from where the foot actually is, not where the plan
                # thought it would be.
                self._swing = SwingState(
                    active=True, leg=leg, start=feet[leg].copy(),
                    start_yaw=float(yaws[leg]),
                )
                self._swing_cleared = False
            if float(feet[leg, 2] - ground_z) > SWING_CLEARED_HEIGHT:
                self._swing_cleared = True
            foothold = np.array([landing.pos[0], landing.pos[1], ground_z])
            clock = 1.0
            if STEP_TIMING:
                clock = self._swing_rate
            elif self._capture_hot and t_swing > 0.20:
                clock = min(1.5, t_swing / 0.20)
            des_pos, des_vel_s, des_acc_s = swing_reference(
                self._swing.start,
                foothold,
                s,
                swing_height=float(
                    self._swing_height if self._latch_valid else self.config.swing_height
                ),
                t_swing=t_swing / clock,
            )
            span_swing = max(t_swing / clock, 1e-3) if STEP_TIMING else max(t_swing, 1e-3)
            shape_d = 6.0 * s * (1.0 - s)
            dyaw = wrap_angle(float(landing.yaw) - float(self._swing.start_yaw))
            self._swing.foothold = foothold
            self._swing.foothold_yaw = float(landing.yaw)
            self._swing.des_pos = des_pos
            self._swing.des_vel = des_vel_s / span_swing
            self._swing.des_acc = des_acc_s / (span_swing * span_swing)
            self._swing.des_yaw = float(self._swing.start_yaw) + dyaw * s * s * (3.0 - 2.0 * s)
            self._swing.des_yaw_rate = dyaw * shape_d / span_swing
            self._swing.s = s
            swing = self._swing
            if leg == 0:
                left_phase = LegPhase.SWING
            else:
                right_phase = LegPhase.SWING
        else:
            self._swing = SwingState(active=False)

        yaw_ref, yaw_rate = self._yaw_now(t_step)
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
            yaw_ref=yaw_ref,
            yaw_rate_ref=yaw_rate,
            stance_yaw=float(support.yaw),
        )

    # -- MPC preview -------------------------------------------------------

    def preview(self, *, horizon: int, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Planned contact flags, foot poses and foot yaws over an MPC horizon.

        Returns ``(contact, foot_pose, foot_yaw)`` with shapes ``(horizon, 2)``
        bool, ``(horizon, 2, 3)`` and ``(horizon, 2)``. ``foot_pose[..., 2]``
        is clearance above the foot line. The centroidal MPC plans 0.3 s
        ahead, which spans most of a step, so replicating the *current*
        contact set across the horizon misstates who is touching the ground.
        """
        contact = np.ones((horizon, 2), dtype=bool)
        foot_pose = np.zeros((horizon, 2, 3))
        foot_yaw = np.zeros((horizon, 2))
        t_step, t_swing, t_ds = self._durations()
        height = float(self._swing_height if self._latch_valid else self.config.swing_height)

        def _xy3(xy: np.ndarray) -> np.ndarray:
            pose = np.zeros(3)
            pose[:2] = xy
            return pose

        here = self._steps.get(self._step)
        if not self._active or t_step <= 0.0 or here is None or here.initial:
            # Standing, or still in the opening transfer: both feet stay where
            # they are for longer than the horizon reaches.
            fallback = self._initial_feet if self._active else np.zeros((2, 2))
            yaws = self._initial_yaws if self._active else np.zeros(2)
            foot_pose[:, :, :2] = fallback
            foot_yaw[:] = yaws
            return contact, foot_pose, foot_yaw

        for k in range(horizon):
            # While waiting on a late touchdown the plant is still in this
            # step's single support; do not tell the MPC the swing foot is
            # already the new stance. k = 0 stays here, later ticks proceed
            # as if we land on the next control step.
            if self._touchdown_hold > 0.0:
                ahead = (t_step - 1e-6) + k * dt
            elif STEP_TIMING:
                ahead = self._tau_ahead((k + 1) * dt, t_ds=t_ds, span=t_step)
            else:
                ahead = self._tau + (k + 1) * dt
            index = self._step + int(ahead // t_step)
            tau = ahead % t_step
            support = self._steps.get(index)
            landing = self._steps.get(index + 1)
            previous = self._steps.get(index - 1)
            if support is None or landing is None:
                if k:
                    foot_pose[k] = foot_pose[k - 1]
                    foot_yaw[k] = foot_yaw[k - 1]
                continue

            if support.initial:
                # Opening double support: both feet are where they started.
                foot_pose[k, :, :2] = self._initial_feet
                foot_yaw[k] = self._initial_yaws
                continue

            swing_leg = landing.leg
            foot_pose[k, support.leg, :2] = support.pos
            foot_yaw[k, support.leg] = support.yaw
            if tau >= t_ds and t_swing > 1e-6:
                contact[k, swing_leg] = False
                if previous is None or previous.initial:
                    start_xy = self._initial_feet[swing_leg]
                    start_yaw = float(self._initial_yaws[swing_leg])
                else:
                    start_xy = previous.pos
                    start_yaw = float(previous.yaw)
                s = float(np.clip((tau - t_ds) / t_swing, 0.0, 1.0))
                pos, _, _ = swing_reference(
                    _xy3(start_xy), _xy3(landing.pos), s, swing_height=height
                )
                foot_pose[k, swing_leg] = pos
                foot_yaw[k, swing_leg] = start_yaw + wrap_angle(
                    float(landing.yaw) - start_yaw
                ) * s * s * (3.0 - 2.0 * s)
            elif previous is None or previous.initial:
                foot_pose[k, swing_leg, :2] = (
                    landing.pos if previous is None else self._initial_feet[swing_leg]
                )
                foot_yaw[k, swing_leg] = float(self._initial_yaws[swing_leg])
            else:
                # Double support: the swing leg is still on its last foothold.
                foot_pose[k, swing_leg, :2] = previous.pos
                foot_yaw[k, swing_leg] = previous.yaw
        return contact, foot_pose, foot_yaw
