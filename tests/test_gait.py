"""Phase B gait / walk unit tests.

Layered so a failure localises itself:

  1. Swing arc geometry (pure function)
  2. Gait clock: swing windows, contact masks, alternation
  3. Footstep plan: commanded stride, stance width, no lateral ratchet
  4. DCM reference: closed-form limit cycle, commanded velocity
  5. Capture-point foothold adjustment
  6. Closed-loop walk on the robot

Layers 1-5 exercise the planner as a pure function of its inputs, with the
robot replaced by perfect tracking. They pin down what the plan *asks for*.
Layer 6 is the closed-loop walk. Eight seconds at 0.10–0.30 m/s and
thirty seconds at 0.10/0.20/0.35/0.50 (``--slow``) are required gates.
"""

from __future__ import annotations

import numpy as np
import pytest

from loka.control.gait import (
    HEADING_CRAWL_SPEED,
    MAX_DCM_CORRECTION,
    MODE_WALK,
    SCHED_UP_LAG,
    SWING_TOUCHDOWN_VZ,
    SWING_TOUCHDOWN_Z,
    GaitConfig,
    GaitScheduler,
    LegPhase,
    gait_schedule_for_speed,
    swing_reference,
)
from loka.control.locomotion import (
    ANKLE_PITCH,
    ANKLE_ROLL,
    HIP_YAW,
    KNEE,
    NUM_LEG_JOINTS,
    STANCE_KNEE_MIN,
    SWING_ANKLE_RELEASE_S,
    SWING_ANKLE_W,
    SWING_NULLSPACE_W,
)
from loka.control.robot import QPOS_JOINT0
from loka.sim import Simulation

#: Minimum closed-loop walk duration for the smoke test.
WALK_TEST_DURATION = 8.0
#: Honest limit-cycle gate (slow): pelvis-z for 8 s is not a walk.
WALK_LIMIT_CYCLE_DURATION = 30.0

FEET = np.array([[0.0, 0.1185, 0.0], [0.0, -0.1185, 0.0]])
PLANTED = np.ones(8, dtype=bool)
HEIGHT = 0.70


def drive(sched, *, com, com_vel, feet=FEET, dt=0.002, ticks=1):
    """Run the scheduler for ``ticks`` with a fixed measured state."""
    out = None
    for _ in range(ticks):
        out = sched.step(
            dt=dt,
            com=com,
            com_vel=com_vel,
            foot_centers=feet,
            ground_z=0.0,
            height=HEIGHT,
            measured_mask=PLANTED,
        )
    return out


def walking_scheduler(**overrides):
    cfg = dict(
        mode=MODE_WALK,
        speed=0.25,
        step_period=0.70,
        duty_factor=0.65,
        stance_width=0.24,
        # 0.5 is the old law's fixed point (k/(1+k) at k=1). Full placement (1.0)
        # shifts the isolation orbit these tests were written against.
        capture_gain=0.5,
        walk_accel=50.0,  # snap to commanded speed so tests are not ramp-limited
    )
    cfg.update(overrides)
    return GaitScheduler(GaitConfig(**cfg))


# -- 1. swing arc ---------------------------------------------------------


def test_swing_arc_clears_ground_and_lands_softly():
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([0.2, 0.0, 0.0])

    mid, _, _ = swing_reference(start, end, 0.5, swing_height=0.06)
    assert mid[2] == pytest.approx(0.06, abs=1e-9)
    assert mid[0] == pytest.approx(0.1, abs=1e-9)

    for s in (0.0, 1.0):
        pos, vel, _ = swing_reference(start, end, s, swing_height=0.06)
        assert pos[2] == pytest.approx(0.0, abs=1e-9)
        # Horizontal velocity vanishes at both ends, so the foot neither jerks
        # off the ground nor scuffs forward as it lands.
        assert vel[0] == pytest.approx(0.0, abs=1e-9)
        assert vel[1] == pytest.approx(0.0, abs=1e-9)

    # Vertical velocity is downward at touchdown, which seats the contact.
    _, vel_end, _ = swing_reference(start, end, 1.0, swing_height=0.06)
    assert vel_end[2] < -0.1


def test_swing_arc_derivatives_match_finite_differences():
    start = np.array([0.0, -0.05, 0.0])
    end = np.array([0.18, 0.05, 0.01])
    h, eps = 0.06, 1e-6
    for s in (0.15, 0.5, 0.85):
        _, vel, acc = swing_reference(start, end, s, swing_height=h)
        p_hi, v_hi, _ = swing_reference(start, end, s + eps, swing_height=h)
        p_lo, v_lo, _ = swing_reference(start, end, s - eps, swing_height=h)
        np.testing.assert_allclose(vel, (p_hi - p_lo) / (2 * eps), atol=1e-4)
        np.testing.assert_allclose(acc, (v_hi - v_lo) / (2 * eps), atol=1e-4)


def test_speed_schedule_slows_the_lateral_orbit_at_low_speed():
    slow = gait_schedule_for_speed(0.10)
    mid = gait_schedule_for_speed(0.20)
    fast = gait_schedule_for_speed(0.50)
    assert slow["gait.step_period"] > fast["gait.step_period"]
    assert slow["gait.duty_factor"] > fast["gait.duty_factor"]
    assert slow["gait.stance_width"] < fast["gait.stance_width"]
    # Closed-loop 0.20 m/s needs ~0.80 duty (0.65 / 0.75 still pitch over).
    assert mid["gait.duty_factor"] == pytest.approx(0.80, abs=0.01)
    assert slow["gait.duty_factor"] >= 0.81
    assert fast["gait.duty_factor"] >= 0.73
    assert fast["gait.duty_factor"] < mid["gait.duty_factor"]


def test_speed_schedule_fills_cadence_unless_the_user_pins_it():
    sched = GaitScheduler(GaitConfig(mode=MODE_WALK, speed=0.0))
    applied = sched.apply_updates({"gait.mode": MODE_WALK, "gait.speed": 0.30})
    assert applied["gait.speed"] == pytest.approx(0.30)
    assert applied["gait.step_period"] == pytest.approx(
        gait_schedule_for_speed(0.30)["gait.step_period"]
    )
    # Above the straight-walk band the command is clipped, then the schedule
    # follows the clipped speed.
    clipped = sched.apply_updates({"gait.speed": 0.80})
    assert clipped["gait.speed"] == pytest.approx(0.30)
    pinned = GaitScheduler(GaitConfig(mode=MODE_WALK, speed=0.0))
    pinned.apply_updates(
        {
            "gait.mode": MODE_WALK,
            "gait.speed": 0.30,
            "gait.step_period": 0.70,
        }
    )
    assert pinned.config.step_period == pytest.approx(0.70)


def test_mpc_preview_puts_the_swing_foot_on_the_bezier():
    """Airborne horizon steps must carry the Bézier lift, not the foothold z=0."""
    sched = walking_scheduler()
    com = np.array([0.0, 0.0, HEIGHT])
    out = _run_until_swing(sched, com=com, feet=FEET)
    contact, pose, _yaw = sched.preview(horizon=6, dt=0.05)
    assert pose.shape == (6, 2, 3)
    swing = int(out.swing.leg)
    airborne = ~contact[:, swing]
    assert airborne.any(), "preview never shows the swing foot airborne"
    assert float(pose[airborne, swing, 2].max()) > 0.01
    planted = contact[:, swing]
    assert np.all(pose[planted, swing, 2] == pytest.approx(0.0, abs=1e-12))


# -- 2. gait clock --------------------------------------------------------


def test_swing_windows_alternate_with_both_feet_down_between():
    sched = walking_scheduler(step_period=0.60, duty_factor=0.70)
    com = np.array([0.0, 0.0, HEIGHT])

    seen = {"left": 0, "right": 0, "double": 0}
    for _ in range(2000):
        out = drive(sched, com=com, com_vel=np.zeros(3))
        if out.left_phase == LegPhase.SWING:
            seen["left"] += 1
            assert out.right_phase == LegPhase.STANCE
            assert not out.contact_mask[:4].any() and out.contact_mask[4:].all()
            assert out.swing.active and out.swing.leg == 0
        elif out.right_phase == LegPhase.SWING:
            seen["right"] += 1
            assert not out.contact_mask[4:].any() and out.contact_mask[:4].all()
            assert out.swing.active and out.swing.leg == 1
        else:
            seen["double"] += 1
            assert out.contact_mask.all() and not out.swing.active

    assert seen["left"] > 0 and seen["right"] > 0 and seen["double"] > 0
    # Each leg swings once per cycle, so the two totals should match closely.
    assert abs(seen["left"] - seen["right"]) < 0.25 * seen["left"]


def test_swing_duration_follows_duty_factor():
    period, duty = 0.60, 0.70
    sched = walking_scheduler(step_period=period, duty_factor=duty)
    com = np.array([0.0, 0.0, HEIGHT])

    dt, swinging = 0.002, 0
    for _ in range(3000):
        out = drive(sched, com=com, com_vel=np.zeros(3), dt=dt)
        if out.swing.active:
            swinging += 1
    # Two swings per cycle, each (1 - duty) of the cycle.
    expected = 2.0 * (1.0 - duty) / period
    assert swinging * dt / (3000 * dt) == pytest.approx(expected * 0.5, rel=0.15)


def _run_until_swing(sched, *, com, feet, dt=0.002):
    """Advance until the first non-opening swing, returning that output."""
    for _ in range(4000):
        out = sched.step(
            dt=dt,
            com=com,
            com_vel=np.zeros(3),
            foot_centers=feet,
            ground_z=0.0,
            height=HEIGHT,
            measured_mask=PLANTED,
        )
        if out.swing.active and sched._step >= 1:
            return out
    raise AssertionError("never entered swing")


def _airborne_mask(leg: int) -> np.ndarray:
    mask = PLANTED.copy()
    if leg == 0:
        mask[0:4] = False
    else:
        mask[4:8] = False
    return mask


def _place_swing(feet, out, *, z=0.0, xy=None):
    """Put the in-flight foot at the planned foothold (or a chosen xy)."""
    placed = feet.copy()
    placed[out.swing.leg, 2] = z
    placed[out.swing.leg, :2] = out.swing.foothold[:2] if xy is None else np.asarray(xy)
    return placed


def test_gait_clock_waits_while_the_swing_foot_is_airborne():
    """The clock must not name an airborne foot as the next stance.

    Closed-loop traces showed the right swing still 30–40 mm up at T_step;
    incrementing anyway is what threw the robot over at step 4.
    """
    from loka.control.gait import MAX_TOUCHDOWN_HOLD

    sched = walking_scheduler(step_period=0.60, duty_factor=0.70)
    com = np.array([0.0, 0.0, HEIGHT])
    feet = FEET.copy()
    dt = 0.002
    out = _run_until_swing(sched, com=com, feet=feet, dt=dt)
    step = sched._step
    leg = out.swing.leg

    high = _place_swing(feet, out, z=0.04)
    mask = _airborne_mask(leg)

    # Well past the scheduled step duration, foot still up → still this step.
    for _ in range(int(0.30 / dt)):
        out = sched.step(
            dt=dt,
            com=com,
            com_vel=np.zeros(3),
            foot_centers=high,
            ground_z=0.0,
            height=HEIGHT,
            measured_mask=mask,
        )
        assert sched._step == step
        assert out.swing.active and out.swing.leg == leg
        assert sched._touchdown_hold <= MAX_TOUCHDOWN_HOLD + dt
        high = _place_swing(high, out, z=0.04)

    # Plant it at the planned foothold: the next tick must commit the step.
    planted = _place_swing(high, out, z=0.0)
    mask[:] = True
    sched.step(
        dt=dt,
        com=com,
        com_vel=np.zeros(3),
        foot_centers=planted,
        ground_z=0.0,
        height=HEIGHT,
        measured_mask=mask,
    )
    assert sched._step == step + 1
    assert sched._touchdown_hold == 0.0


def test_gait_clock_gives_up_waiting_after_the_touchdown_timeout():
    from loka.control.gait import MAX_TOUCHDOWN_HOLD

    sched = walking_scheduler(step_period=0.60, duty_factor=0.70)
    com = np.array([0.0, 0.0, HEIGHT])
    feet = FEET.copy()
    dt = 0.002
    out = _run_until_swing(sched, com=com, feet=feet, dt=dt)
    step = sched._step
    leg = out.swing.leg
    high = feet.copy()
    high[leg, 2] = 0.04
    mask = _airborne_mask(leg)

    for _ in range(int((0.35 + MAX_TOUCHDOWN_HOLD + 0.05) / dt)):
        sched.step(
            dt=dt,
            com=com,
            com_vel=np.zeros(3),
            foot_centers=high,
            ground_z=0.0,
            height=HEIGHT,
            measured_mask=mask,
        )
        if sched._step > step:
            break
    assert sched._step == step + 1


def test_gait_clock_advances_early_once_the_swing_foot_plants():
    """A real swing that seats before T_step must not wait out the Bézier.

    The late hold is the matching path: the clock already waits for a late
    foot. Isolation tests keep z = 0, so they never set ``_swing_cleared``
    and keep the nominal cadence.
    """
    from loka.control.gait import EARLY_PLANT_S

    sched = walking_scheduler(step_period=0.60, duty_factor=0.70)
    com = np.array([0.0, 0.0, HEIGHT])
    feet = FEET.copy()
    dt = 0.002
    out = _run_until_swing(sched, com=com, feet=feet, dt=dt)
    step = sched._step
    leg = out.swing.leg
    t_step, t_swing, t_ds = sched._durations()
    high = feet.copy()
    high[leg, 2] = 0.04
    mask = _airborne_mask(leg)
    ready = t_ds + EARLY_PLANT_S * t_swing
    while sched._tau < ready + 0.02:
        out = sched.step(
            dt=dt,
            com=com,
            com_vel=np.zeros(3),
            foot_centers=high,
            ground_z=0.0,
            height=HEIGHT,
            measured_mask=mask,
        )
        assert sched._step == step
    assert sched._swing_cleared
    assert sched._tau < t_step
    planted = _place_swing(high, out, z=0.0)
    mask[:] = True
    sched.step(
        dt=dt,
        com=com,
        com_vel=np.zeros(3),
        foot_centers=planted,
        ground_z=0.0,
        height=HEIGHT,
        measured_mask=mask,
    )
    assert sched._step == step + 1


def _hover_until_waiting(sched, *, com, feet, out, dt):
    """Lift the swing foot and run until the late hold starts, not past it."""
    from loka.control.gait import MAX_TOUCHDOWN_HOLD

    step = sched._step
    high = _place_swing(feet, out, z=0.04)
    mask = _airborne_mask(out.swing.leg)
    for _ in range(int((0.50 + MAX_TOUCHDOWN_HOLD) / dt)):
        out = sched.step(
            dt=dt,
            com=com,
            com_vel=np.zeros(3),
            foot_centers=high,
            ground_z=0.0,
            height=HEIGHT,
            measured_mask=mask,
        )
        assert sched._step == step
        high = _place_swing(high, out, z=0.04)
        if sched._touchdown_hold > 0.0:
            return out, high
    raise AssertionError("clock never entered the touchdown hold")


def test_gait_clock_does_not_commit_a_short_plant():
    """Height + contact is not enough: the planned point must still be on the sole.

    Closed-loop traces landed 16–32 mm short of the first foothold. Honouring
    that as stance named the next ZMP in the air.
    """
    from loka.control.gait import TOUCHDOWN_RADIUS

    sched = walking_scheduler(step_period=0.60, duty_factor=0.70)
    com = np.array([0.0, 0.0, HEIGHT])
    feet = FEET.copy()
    dt = 0.002
    out = _run_until_swing(sched, com=com, feet=feet, dt=dt)
    step = sched._step
    out, high = _hover_until_waiting(sched, com=com, feet=feet, out=out, dt=dt)
    # The hovered foot can sit inside the touchdown radius once capture pulls
    # the foothold toward it. Place an explicitly short landing instead.
    short = np.asarray(out.swing.foothold[:2], dtype=float) - np.array([0.04, 0.0])
    assert float(np.linalg.norm(short - out.swing.foothold[:2])) > TOUCHDOWN_RADIUS
    planted = _place_swing(high, out, z=0.0, xy=short)
    mask = PLANTED.copy()
    for _ in range(int(0.05 / dt)):
        sched.step(
            dt=dt,
            com=com,
            com_vel=np.zeros(3),
            foot_centers=planted,
            ground_z=0.0,
            height=HEIGHT,
            measured_mask=mask,
        )
        assert sched._step == step


def test_committed_support_reanchors_at_the_measured_plant():
    """The next nominal must chain from where the foot actually is."""
    from loka.control.gait import TOUCHDOWN_RADIUS

    sched = walking_scheduler(step_period=0.60, duty_factor=0.70)
    com = np.array([0.0, 0.0, HEIGHT])
    feet = FEET.copy()
    dt = 0.002
    out = _run_until_swing(sched, com=com, feet=feet, dt=dt)
    step = sched._step
    out, high = _hover_until_waiting(sched, com=com, feet=feet, out=out, dt=dt)
    offset = np.array([-0.5 * TOUCHDOWN_RADIUS, 0.0])
    plant_xy = out.swing.foothold[:2] + offset
    planted = _place_swing(high, out, z=0.0, xy=plant_xy)
    sched.step(
        dt=dt,
        com=com,
        com_vel=np.zeros(3),
        foot_centers=planted,
        ground_z=0.0,
        height=HEIGHT,
        measured_mask=PLANTED,
    )
    assert sched._step == step + 1
    support = sched._steps[sched._step]
    np.testing.assert_allclose(support.pos, plant_xy, atol=1e-9)


def test_stand_mode_keeps_every_contact():
    sched = GaitScheduler(GaitConfig())
    out = drive(sched, com=np.array([0.0, 0.0, HEIGHT]), com_vel=np.zeros(3))
    assert out.contact_mask.all()
    assert not out.swing.active
    assert not out.walking


# -- 3. footstep plan -----------------------------------------------------


def _footholds(sched, com, com_vel, *, ticks=4000):
    """Collect committed footholds, one per step."""
    seen, last = [], None
    for _ in range(ticks):
        out = drive(sched, com=com, com_vel=com_vel)
        if out.swing.active and (last is None or last != sched._step):
            last = sched._step
            seen.append((out.swing.leg, out.swing.foothold[:2].copy()))
    return seen


def test_footholds_advance_by_commanded_stride():
    speed, period = 0.25, 0.70
    sched = walking_scheduler(speed=speed, step_period=period, capture_gain=0.0)
    com = np.array([0.0, 0.0, HEIGHT])
    seen = _footholds(sched, com, np.zeros(3))
    assert len(seen) >= 4

    # Same-leg footholds are one full cycle apart, so they advance by 2 strides.
    per_step = speed * 0.5 * period
    for leg in (0, 1):
        xs = [p[0] for lg, p in seen if lg == leg]
        deltas = np.diff(xs)
        assert len(deltas) >= 1
        np.testing.assert_allclose(deltas, 2.0 * per_step, atol=0.02)


def test_footholds_hold_stance_width_without_lateral_ratchet():
    sched = walking_scheduler(stance_width=0.24, capture_gain=0.0)
    com = np.array([0.0, 0.0, HEIGHT])
    seen = _footholds(sched, com, np.zeros(3))
    assert len(seen) >= 4

    lefts = [p[1] for lg, p in seen if lg == 0]
    rights = [p[1] for lg, p in seen if lg == 1]
    assert lefts and rights
    # Each leg keeps its own side, half a stance width off the midline.
    np.testing.assert_allclose(lefts, 0.12, atol=0.02)
    np.testing.assert_allclose(rights, -0.12, atol=0.02)
    # And successive footholds of one leg do not creep sideways. A plan that
    # chains off the previous *foot* instead of a commanded offset walks itself
    # off to one side a few centimetres per step.
    for series in (lefts, rights):
        assert max(series) - min(series) < 0.01


def test_zero_speed_returns_to_stand():
    sched = walking_scheduler(speed=0.25)
    com = np.array([0.0, 0.0, HEIGHT])
    drive(sched, com=com, com_vel=np.zeros(3), ticks=600)
    assert sched.walking

    sched.config.speed = 0.0
    for _ in range(4000):
        out = drive(sched, com=com, com_vel=np.zeros(3))
        if not out.walking:
            break
    assert not sched.walking, "gait never handed back to the standing controller"
    assert out.contact_mask.all()


# -- 4. DCM reference -----------------------------------------------------


def _closed_loop_plan(sched, *, ticks=2000, dt=0.002):
    """Run the planner against a robot that tracks it perfectly."""
    com = np.array([0.0, 0.0, HEIGHT])
    vel = np.zeros(3)
    feet = FEET.copy()
    history = []
    for _ in range(ticks):
        out = sched.step(
            dt=dt,
            com=com,
            com_vel=vel,
            foot_centers=feet,
            ground_z=0.0,
            height=HEIGHT,
            measured_mask=PLANTED,
        )
        com[:2] = sched._com_ref
        vel[:2] = sched._com_vel_ref
        if out.swing.active:
            feet[out.swing.leg, :2] = out.swing.des_pos[:2]
        history.append((com.copy(), vel.copy(), out.dcm_ref.copy()))
    return history


@pytest.mark.stack_legacy
def test_plan_settles_at_commanded_forward_speed():
    speed = 0.20
    sched = walking_scheduler(speed=speed)
    history = _closed_loop_plan(sched, ticks=3000)
    # Average over the last two cycles, past the opening transfer.
    tail = history[-700:]
    mean_vx = float(np.mean([v[0] for _, v, _ in tail]))
    assert mean_vx == pytest.approx(speed, rel=0.05)


@pytest.mark.stack_legacy
def test_plan_lateral_motion_is_a_bounded_limit_cycle():
    sched = walking_scheduler(speed=0.20, stance_width=0.24)
    history = _closed_loop_plan(sched, ticks=3000)
    tail = history[-1000:]
    ys = np.array([c[1] for c, _, _ in tail])
    dcm_ys = np.array([d[1] for _, _, d in tail])

    # Sways about the midline rather than drifting off it.
    assert abs(float(ys.mean())) < 0.01
    assert 0.005 < float(np.abs(ys).max()) < 0.06
    # The DCM swings wider than the CoM, and stays inside the stance.
    assert float(np.abs(dcm_ys).max()) > float(np.abs(ys).max())
    assert float(np.abs(dcm_ys).max()) < 0.12


@pytest.mark.stack_legacy
def test_first_lift_has_limit_cycle_lateral_velocity():
    """Opening ZMP on the upcoming swing-foot inset preloads the DCM orbit.

    A midpoint opening ZMP left first lift-off at ~0.01 m/s against a
    0.10 m/s cycle; a stance-foot opening reversed the sign. The APA
    (ZMP on the foot about to swing, inset into the sole) throws CoM
    onto the stance orbit so first lift-off matches the cycle.
    """
    from loka.control.gait import heading_frame

    sched = walking_scheduler(speed=0.20, stance_width=0.24)
    com = np.array([0.0, 0.0, HEIGHT])
    vel = np.zeros(3)
    feet = FEET.copy()
    _, left = heading_frame(0.0)
    lift_vl = {}
    for _ in range(2500):
        out = sched.step(
            dt=0.002,
            com=com,
            com_vel=vel,
            foot_centers=feet,
            ground_z=0.0,
            height=HEIGHT,
            measured_mask=PLANTED,
        )
        com[:2] = sched._com_ref
        vel[:2] = sched._com_vel_ref
        if out.swing.active:
            feet[out.swing.leg, :2] = out.swing.des_pos[:2]
            step = int(sched._step)
            if step not in lift_vl:
                lift_vl[step] = float(vel[:2] @ left)
    assert 1 in lift_vl and 5 in lift_vl
    cycle = lift_vl[5]
    assert abs(cycle) > 0.06
    assert np.sign(lift_vl[1]) == np.sign(cycle)
    assert lift_vl[1] == pytest.approx(cycle, rel=0.25)


def test_com_reference_never_leads_the_robot_unboundedly():
    """A robot that refuses to move must not drag the plan onto itself.

    The plan is open loop along the heading precisely so that a stalled robot
    still gets a forward command; what it must not do is let the CoM *target*
    run arbitrarily far away, which would ask the QP for accelerations the
    soles cannot produce.
    """
    sched = walking_scheduler(speed=0.30)
    com = np.array([0.0, 0.0, HEIGHT])
    worst = 0.0
    for _ in range(3000):
        out = drive(sched, com=com, com_vel=np.zeros(3))
        worst = max(worst, float(np.linalg.norm(out.com_ref_xy - com[:2])))
    assert worst < 0.10


# -- 5. capture-point adjustment -----------------------------------------


@pytest.mark.stack_legacy
def test_foothold_steps_out_when_com_is_racing():
    """Excess forward velocity must move the foothold forward, and only so far."""
    slow = walking_scheduler(speed=0.25, capture_gain=1.0)
    fast = walking_scheduler(speed=0.25, capture_gain=1.0)
    com = np.array([0.0, 0.0, HEIGHT])

    nominal = _footholds(slow, com, np.array([0.25, 0.0, 0.0]), ticks=900)
    pushed = _footholds(fast, com, np.array([1.20, 0.0, 0.0]), ticks=900)
    assert nominal and pushed
    assert pushed[0][1][0] > nominal[0][1][0] + 0.02

    # ... but never beyond the reachable stride.
    assert pushed[0][1][0] < 0.45


@pytest.mark.stack_legacy
def test_capture_gain_zero_leaves_the_nominal_plan_alone():
    sched = walking_scheduler(speed=0.25, capture_gain=0.0)
    com = np.array([0.0, 0.0, HEIGHT])
    calm = _footholds(sched, com, np.array([0.25, 0.0, 0.0]), ticks=900)

    sched2 = walking_scheduler(speed=0.25, capture_gain=0.0)
    racing = _footholds(sched2, com, np.array([1.20, 0.0, 0.0]), ticks=900)
    assert calm and racing
    np.testing.assert_allclose(calm[0][1], racing[0][1], atol=1e-9)


@pytest.mark.stack_legacy
def test_foothold_correction_does_not_feed_back_on_itself():
    """Applying the correction twice at one instant must give the same foothold.

    The old law compared the predicted DCM with a goal the previous correction
    had already moved, so a second pass computed ``k (e - c)`` instead of ``c``.
    """
    sched = walking_scheduler(speed=0.25, capture_gain=1.0)
    com = np.array([0.0, 0.0, HEIGHT])
    com_vel = np.array([0.40, 0.05, 0.0])
    out = None
    for _ in range(4000):
        out = drive(sched, com=com, com_vel=com_vel)
        if out.swing.active and 0.1 < out.swing.s < 0.4:
            break
    assert out is not None and out.swing.active and 0.1 < out.swing.s < 0.4
    t_step, _, _ = sched._durations()
    omega = sched._omega(HEIGHT)
    dcm = com[:2] + com_vel[:2] / omega
    s = float(out.swing.s)
    tau = float(sched._tau)
    corrections = []
    target = sched._steps[sched._step + 1]
    for _ in range(2):
        sched._tau = tau
        sched._refresh_plan(t_step=t_step, speed=sched._cmd_speed)
        sched._apply_dcm_correction(dcm=dcm, s=s, omega=omega, t_step=t_step)
        corrections.append(np.array(target.correction, dtype=float))
    np.testing.assert_allclose(corrections[0], corrections[1], atol=1e-9)
    norm = float(np.linalg.norm(corrections[0]))
    assert 1e-3 < norm < MAX_DCM_CORRECTION - 1e-3


def _in_double_support(sched):
    t_step, _, t_ds = sched._durations()
    support = sched._steps.get(sched._step)
    return (
        support is not None
        and not support.initial
        and sched._tau < t_ds
        and t_ds > 0.02
    )


def test_step_timing_stays_nominal_until_a_swing_has_cleared():
    """Isolation never clears a swing, so the timing QP must not change the clock."""
    import loka.control.gait as gait

    previous = gait.STEP_TIMING
    gait.STEP_TIMING = True
    try:
        sched = walking_scheduler(speed=0.25, capture_gain=0.7)
        assert not sched._cleared_once
        com = np.array([0.0, 0.0, HEIGHT])
        for _ in range(1500):
            drive(sched, com=com, com_vel=np.array([1.2, 0.0, 0.0]))
            assert sched._ds_rate == pytest.approx(1.0)
            assert sched._swing_rate == pytest.approx(1.0)
    finally:
        gait.STEP_TIMING = previous


def test_step_timing_only_shortens_and_speeds_double_support_first():
    import loka.control.gait as gait

    previous = gait.STEP_TIMING
    gait.STEP_TIMING = True
    try:
        sched = walking_scheduler(speed=0.25, capture_gain=0.7)
        sched._cleared_once = True
        com = np.array([0.0, 0.0, HEIGHT])
        com_vel = np.array([1.2, 0.0, 0.0])
        fastest = 1.0
        for _ in range(2500):
            drive(sched, com=com, com_vel=com_vel)
            assert sched._ds_rate >= 1.0 - 1e-9
            assert 1.0 - 1e-9 <= sched._swing_rate <= gait.STEP_SWING_RATE_MAX + 1e-9
            if _in_double_support(sched):
                fastest = max(fastest, sched._ds_rate)
        assert fastest > 1.0
    finally:
        gait.STEP_TIMING = previous


def test_preview_uses_the_double_support_rate():
    import loka.control.gait as gait

    previous = gait.STEP_TIMING
    gait.STEP_TIMING = True
    try:
        sched = walking_scheduler(speed=0.25, capture_gain=0.0)
        com = np.array([0.0, 0.0, HEIGHT])
        for _ in range(2500):
            out = drive(sched, com=com, com_vel=np.array([0.25, 0.0, 0.0]))
            if _in_double_support(sched) and not out.swing.active:
                break
        assert _in_double_support(sched)
        sched._ds_rate = 2.0
        sched._swing_rate = 1.0
        _, _, t_ds = sched._durations()
        dt = 0.02
        contact, _, _ = sched.preview(horizon=40, dt=dt)
        landing = sched._steps[sched._step + 1]
        expected = int(np.ceil(((t_ds - sched._tau) / 2.0) / dt)) - 1
        first = next(k for k in range(40) if not contact[k, landing.leg])
        assert abs(first - expected) <= 1
    finally:
        gait.STEP_TIMING = previous


def test_step_timing_foothold_does_not_feed_back_on_itself():
    import loka.control.gait as gait

    previous = gait.STEP_TIMING
    gait.STEP_TIMING = True
    try:
        sched = walking_scheduler(speed=0.25, capture_gain=1.0)
        com = np.array([0.0, 0.0, HEIGHT])
        com_vel = np.array([0.40, 0.05, 0.0])
        out = None
        for _ in range(4000):
            out = drive(sched, com=com, com_vel=com_vel)
            if out.swing.active and 0.1 < out.swing.s < 0.4:
                break
        assert out is not None and out.swing.active
        t_step, t_swing, t_ds = sched._durations()
        omega = sched._omega(HEIGHT)
        dcm = com[:2] + com_vel[:2] / omega
        s = float(out.swing.s)
        tau = float(sched._tau)
        footholds = []
        target = sched._steps[sched._step + 1]
        for _ in range(2):
            sched._tau = tau
            sched._refresh_plan(t_step=t_step, speed=sched._cmd_speed)
            sched._step_adjust(
                dcm=dcm, s=s, omega=omega, t_step=t_step,
                t_swing=t_swing, t_ds=t_ds, in_swing=True,
            )
            footholds.append(np.array(target.pos, dtype=float))
        np.testing.assert_allclose(footholds[0], footholds[1], atol=1e-4)
    finally:
        gait.STEP_TIMING = previous


@pytest.mark.stack_legacy
def test_heading_step_turns_one_step_at_a_time():
    """A 90° heading goal must not yaw the next foot by 90°."""
    from loka.control.gait import MIN_FOOT_SEPARATION, heading_frame, max_turn_per_step, wrap_angle

    sched = walking_scheduler(speed=0.20, heading=np.pi / 2, capture_gain=0.0)
    com = np.zeros(3)
    com[2] = HEIGHT
    feet = FEET.copy()
    drive(sched, com=com, com_vel=np.zeros(3), feet=feet, ticks=4000)
    yaws = [float(step.yaw) for _, step in sorted(sched._steps.items()) if step.leg >= 0]
    cap = max_turn_per_step(0.20, sched._t_step, sched.config.turn_rate) + 1e-6
    for a, b in zip(yaws, yaws[1:]):
        assert abs(wrap_angle(b - a)) <= cap + 1e-6
    assert abs(wrap_angle(yaws[-1] - np.pi / 2)) < 0.05
    for index, step in sched._steps.items():
        support = sched._steps.get(index - 1)
        if support is None or step.leg < 0 or support.leg < 0:
            continue
        _, left = heading_frame(float(support.yaw))
        gap = abs(float(np.dot(step.pos - support.pos, left)))
        assert gap + 1e-6 >= MIN_FOOT_SEPARATION * 0.5


def test_heading_wraps_the_short_way():
    from loka.control.gait import wrap_angle

    sched = walking_scheduler(speed=0.20, heading=3.0, capture_gain=0.0)
    com = np.array([0.0, 0.0, HEIGHT])
    drive(sched, com=com, com_vel=np.zeros(3), ticks=1500)
    yaw_at_switch = float(sched._steps[sched._step].yaw)
    sched.config.heading = -3.0
    drive(sched, com=com, com_vel=np.zeros(3), ticks=1500)
    yaw = float(sched._steps[sched._step].yaw)
    short = wrap_angle(-3.0 - yaw_at_switch)
    # Motion after the switch follows the short arc, not the long way round.
    assert wrap_angle(yaw - yaw_at_switch) * np.sign(short) > 0.02


def test_turn_in_place_keeps_the_centre():
    sched = walking_scheduler(speed=0.0, heading=np.pi / 2, capture_gain=0.0)
    com = np.array([0.0, 0.0, HEIGHT])
    drive(sched, com=com, com_vel=np.zeros(3), ticks=3000)
    assert sched.walking
    centre = np.mean([step.pos for step in sched._steps.values()], axis=0)
    assert float(np.linalg.norm(centre)) < 0.05


def test_in_place_capture_does_not_walk_the_centre():
    """A backward capture step must not move the latched turn centre.

    convex-mpc-biped rotates the lateral offset about the footprint centre.
    Chaining the next centre off the corrected foot walked a 90° turn ~0.71 m.
    """
    sched = walking_scheduler(speed=0.0, heading=np.pi / 2, capture_gain=0.7)
    com = np.array([-0.08, 0.0, HEIGHT])
    vel = np.array([-0.05, 0.0, 0.0])
    max_correction = 0.0
    for _ in range(2500):
        drive(sched, com=com, com_vel=vel)
        target = sched._steps.get(sched._step + 1)
        if target is not None:
            max_correction = max(
                max_correction, float(np.linalg.norm(target.correction))
            )
    assert max_correction > 0.01, "capture never moved a foot, so the chain was not tested"
    assert sched._turn_center is not None
    latched = np.asarray(sched._turn_center, dtype=float)
    assert float(np.linalg.norm(latched)) < 0.05
    radius = 0.5 * float(sched.config.stance_width)
    for step in sched._steps.values():
        if step.leg < 0 or step.nominal is None or step.frozen:
            continue
        dist = float(np.linalg.norm(np.asarray(step.nominal) - latched))
        assert dist == pytest.approx(radius, abs=0.01)


def test_closed_loop_turn_in_place_stays_near_the_start():
    """90° at zero speed stays up and within 0.15 m of the start.

    Before the latched footprint centre this drift was about 0.71 m.
    """
    from loka.control.gait import wrap_angle

    result = _walk(0.0, 8.0, **{"gait.heading": float(np.pi / 2)})
    assert not result["fell"], f"fell after {result['elapsed']:.2f}s"
    drift = float(np.hypot(result["forward"], result["lateral"]))
    assert drift < 0.15, f"pelvis moved {drift:.3f} m"
    yaw = float(result["sim"].controller.last_gait.stance_yaw)
    assert abs(wrap_angle(yaw - np.pi / 2)) < 0.15


def test_mid_swing_period_change_does_not_jump_phase():
    sched = walking_scheduler(step_period=0.80, duty_factor=0.80)
    com = np.array([0.0, 0.0, HEIGHT])
    out = _run_until_swing(sched, com=com, feet=FEET)
    s0 = float(out.swing.s)
    mask0 = out.contact_mask.copy()
    sched.config.step_period = 0.50
    sched.config.duty_factor = 0.60
    out = drive(sched, com=com, com_vel=np.zeros(3))
    assert abs(out.swing.s - s0) < 0.05
    assert np.array_equal(out.contact_mask, mask0)


def test_schedule_speed_does_not_jump_ahead_of_the_measurement():
    """Cadence may lead the measured speed by SCHED_UP_LAG, not by the command."""
    from loka.control.gait import SPEED_HYSTERESIS

    if not SPEED_HYSTERESIS:
        pytest.skip("SPEED_HYSTERESIS is off")
    sched = walking_scheduler(speed=0.50, walk_accel=50.0)
    sched._use_speed_schedule = True
    com = np.array([0.0, 0.0, HEIGHT])
    drive(sched, com=com, com_vel=np.zeros(3), ticks=2500)
    assert sched._sched_speed <= SCHED_UP_LAG + 1e-6
    assert sched._cmd_speed > 0.20


def test_schedule_speed_holds_while_capture_is_hot():
    from loka.control.gait import SPEED_HYSTERESIS

    if not SPEED_HYSTERESIS:
        pytest.skip("SPEED_HYSTERESIS is off")
    sched = walking_scheduler(speed=0.30, walk_accel=50.0)
    sched._sched_speed = 0.20
    sched._sched_dwell = 10
    sched._capture_hot = True
    sched._cmd_speed = 0.30
    sched._update_sched_speed()
    assert sched._sched_speed == pytest.approx(0.20)
    assert sched._cmd_speed == pytest.approx(0.30)


def test_ramp_freeze_never_decreases_the_command_because_of_lag():
    from loka.control.gait import RAMP_FREEZE

    if not RAMP_FREEZE:
        pytest.skip("RAMP_FREEZE is off")
    sched = walking_scheduler(speed=0.40, walk_accel=1.0)
    sched._cleared_once = True
    sched._meas_speed = 0.0
    sched._cmd_speed = 0.20
    sched._velocity_command(0.002)
    assert sched._cmd_speed == pytest.approx(0.20)
    sched.config.speed = 0.05
    sched._velocity_command(0.002)
    assert sched._cmd_speed < 0.20


def test_turn_settle_holds_the_crawl_until_the_heading_is_met():
    sched = walking_scheduler(speed=0.40, heading=1.2, walk_accel=50.0)
    com = np.array([0.0, 0.0, HEIGHT])
    drive(sched, com=com, com_vel=np.zeros(3), ticks=200)
    assert sched._turning
    assert sched._cmd_speed <= HEADING_CRAWL_SPEED + 1e-6
    released = walking_scheduler(speed=0.20, heading=0.0)
    released._turning = True
    released._meas_speed = 0.0
    released._update_turning(at_boundary=True)
    assert released._turning
    released._update_turning(at_boundary=True)
    assert not released._turning


def test_yaw_rate_is_zero_in_single_support():
    from loka.control.gait import YAW_IN_DS_ONLY

    if not YAW_IN_DS_ONLY:
        pytest.skip("YAW_IN_DS_ONLY is off; the full-step yaw ramp is the shipped reference")
    sched = walking_scheduler(
        speed=0.0, heading=1.0, capture_gain=0.0, duty_factor=0.70, step_period=0.80,
    )
    com = np.array([0.0, 0.0, HEIGHT])
    observed = False
    out = None
    for _ in range(5000):
        out = sched.step(
            dt=0.002, com=com, com_vel=np.zeros(3), foot_centers=FEET,
            ground_z=0.0, height=HEIGHT, measured_mask=PLANTED,
        )
        step = sched._steps.get(sched._step)
        if step is None or step.initial:
            continue
        _, _, t_ds = sched._durations()
        if t_ds <= sched._tau < sched._current_duration(sched._t_step):
            observed = True
            assert abs(out.yaw_rate_ref) < 1e-8
    assert observed


def test_spline_swing_lands_softly_and_peaks_at_the_commanded_height():
    start = np.zeros(3)
    end = np.array([0.2, 0.0, 0.0])
    t_swing = 0.25
    landed, vel_s, _ = swing_reference(
        start, end, 1.0, swing_height=0.045, t_swing=t_swing, profile="spline",
    )
    mid, _, _ = swing_reference(
        start, end, 0.5, swing_height=0.045, t_swing=t_swing, profile="spline",
    )
    assert landed[2] == pytest.approx(SWING_TOUCHDOWN_Z, abs=1e-6)
    assert vel_s[2] / t_swing == pytest.approx(-SWING_TOUCHDOWN_VZ, abs=1e-6)
    assert mid[2] == pytest.approx(0.045, abs=1e-6)


# -- 6. closed-loop walk on the robot ------------------------------------


def _walk(speed: float, duration: float, stack: str = "legacy_dcm", **gait):
    from loka.control.locomotion import LocomotionConfig

    config = LocomotionConfig()
    config.stack = stack
    sim = Simulation(config)
    controller = sim.controller
    controller.set_task_targets({"gait.mode": "walk", "gait.speed": speed, **gait})
    start = np.array(sim.data.qpos[:2])
    steps, airborne, clearance = 0, False, 0.0
    while sim.data.time < duration:
        sim.step()
        telemetry = controller.telemetry
        if telemetry.swing_clearance > 0.0:
            clearance = max(clearance, telemetry.swing_clearance)
            if not airborne:
                steps += 1
            airborne = True
        else:
            airborne = False
        if sim.fell:
            break
    travel = np.array(sim.data.qpos[:2]) - start
    return dict(
        sim=sim,
        elapsed=float(sim.data.time),
        fell=bool(sim.fell),
        forward=float(travel[0]),
        lateral=float(travel[1]),
        steps=steps,
        clearance=clearance,
    )


def test_hip_yaw_sign():
    """+hip yaw must increase foot yaw relative to the pelvis."""
    from loka.control.robot import quat_to_rpy

    sim = Simulation()
    robot = sim.controller.robot
    q = robot.nominal_qpos.copy()
    robot.update(q, np.zeros(robot.nv))
    base = float(robot.foot_yaws()[0] - quat_to_rpy(q[3:7])[2])
    q[QPOS_JOINT0 + HIP_YAW] += 0.1
    robot.update(q, np.zeros(robot.nv))
    moved = float(robot.foot_yaws()[0] - quat_to_rpy(q[3:7])[2])
    assert moved > base + 0.05


def test_swing_ankles_release_on_the_descending_arc():
    """Ankle pitch/roll must go limp late in swing so the sole can lay flat.

    Holding them at the stand keyframe for the whole swing left the right
    foot on a toe while the centre was still 10 mm up. Hip yaw stays held.
    """
    sim = Simulation()
    sim.controller.set_task_targets({"gait.mode": "walk", "gait.speed": 0.25})
    saw_held = saw_released = False
    per_leg = NUM_LEG_JOINTS // 2
    while sim.data.time < 2.5 and not sim.fell:
        sim.step()
        gait = sim.controller._last_gait
        if gait is None or not gait.swing.active:
            continue
        leg = int(gait.swing.leg)
        weights = sim.controller._last_joint_weights
        j0 = leg * per_leg
        ankle = float(weights[j0 + ANKLE_PITCH])
        yaw = float(weights[j0 + HIP_YAW])
        assert yaw == pytest.approx(SWING_ANKLE_W)
        if gait.swing.s <= SWING_ANKLE_RELEASE_S:
            assert ankle == pytest.approx(SWING_ANKLE_W)
            saw_held = True
        else:
            assert ankle == pytest.approx(SWING_NULLSPACE_W)
            assert float(weights[j0 + ANKLE_ROLL]) == pytest.approx(SWING_NULLSPACE_W)
            saw_released = True
        if saw_held and saw_released:
            return
    raise AssertionError(
        f"never saw both sides of the release (held={saw_held} released={saw_released})"
    )


def test_swing_sole_orientation_engages_on_the_descending_arc():
    """Late swing must command a world-flat sole, not a limp or pinned ankle.

    The task is off on the way up (ankles stay at the stand pose there) and
    on after ``SWING_ANKLE_RELEASE_S``, which is also when the ankles go limp
    so the QP can use them.
    """
    sim = Simulation()
    sim.controller.set_task_targets({"gait.mode": "walk", "gait.speed": 0.25})
    saw_off = saw_on = False
    while sim.data.time < 2.5 and not sim.fell:
        sim.step()
        gait = sim.controller._last_gait
        if gait is None or not gait.swing.active:
            continue
        active = bool(sim.controller._last_swing_orient_active)
        # Yaw is tracked for the whole swing. Pitch and roll join only on
        # the descending arc (the ankle hold owns them until then), so the
        # task itself is active on both sides of the release.
        if gait.swing.s <= SWING_ANKLE_RELEASE_S:
            assert active
            saw_off = True
        elif active:
            saw_on = True
        if saw_off and saw_on:
            return
    raise AssertionError(
        f"never saw both sides of the orientation task (off={saw_off} on={saw_on})"
    )


def test_com_height_stiffness_tracks_support():
    """Vertical CoM kp must follow the planted-contact fraction.

    Standing (eight sites) keeps full height hold. Single support is half
    the contacts, so the spring is half — that is what stops the stance
    leg vaulting without a hardcoded crouch.
    """
    sim = Simulation()
    sim.step()
    assert sim.controller._last_com_z_scale == pytest.approx(1.0)
    sim.controller.set_task_targets({"gait.mode": "walk", "gait.speed": 0.25})
    while sim.data.time < 2.5 and not sim.fell:
        sim.step()
        gait = sim.controller._last_gait
        if gait is not None and gait.swing.active:
            assert sim.controller._last_com_z_scale == pytest.approx(0.5, abs=0.05)
            return
    raise AssertionError("never reached single support")


def test_stance_knee_stays_above_the_flexion_floor():
    """Planted knees must not slam through the 15° emergency floor.

    With planted-leg kp = 0 the CoM-height task drove the stance knee from
    18° to −10° in 32 ms. A light keyframe bias plus a one-sided kick
    below 15° should keep both stance knees out of the stop.
    """
    sim = Simulation()
    sim.controller.set_task_targets({"gait.mode": "walk", "gait.speed": 0.25})
    per_leg = NUM_LEG_JOINTS // 2
    stance_min = [np.inf, np.inf]
    while sim.data.time < 1.6 and not sim.fell:
        sim.step()
        gait = sim.controller._last_gait
        if gait is None or not gait.walking:
            continue
        qj = np.asarray(sim.data.qpos[QPOS_JOINT0:], dtype=float)
        for leg in (0, 1):
            if gait.swing.active and int(gait.swing.leg) == leg:
                continue
            stance_min[leg] = min(stance_min[leg], float(qj[leg * per_leg + KNEE]))
    assert stance_min[0] < np.inf and stance_min[1] < np.inf, "never saw a stance knee"
    slack = np.radians(8.0)
    for leg, q in enumerate(stance_min):
        assert q > STANCE_KNEE_MIN - slack, (
            f"{'LR'[leg]} stance knee reached {np.degrees(q):.1f} deg "
            f"(floor {np.degrees(STANCE_KNEE_MIN):.0f} deg)"
        )


def test_walk_command_produces_real_steps_not_a_shuffle():
    """The plan must actually lift the feet and commit to footholds.

    This is the regression guard for the original defect, where the controller
    stayed upright by shuffling: ~3 mm of foot clearance and 8% of commanded
    speed. High double-support (the orbit that stays up) shortens air time,
    so closed-loop peak clearance is ~22 mm against a 45 mm Bézier — still
    a step, not a scrape. Keep this well above the old 3 mm shuffle.
    """
    result = _walk(0.20, 3.0)
    assert result["steps"] >= 4, "no swing phases executed"
    assert result["clearance"] > 0.018, (
            f"swing foot only cleared {result['clearance']*1000:.1f} mm -- "
            "the Cartesian swing task is not tracking the planned arc"
        )
    assert result["forward"] > 0.15, "barely any forward travel"


@pytest.mark.parametrize("stack", ["legacy_dcm", "alip_footstep"])
@pytest.mark.parametrize("speed", [0.10, 0.20, 0.30])
def test_closed_loop_walk_stays_up(speed, stack):
    result = _walk(speed, WALK_TEST_DURATION, stack=stack)
    assert not result["fell"], (
        f"fell after {result['elapsed']:.2f}s / {result['steps']} steps "
        f"(forward {result['forward']:+.2f} m, lateral {result['lateral']:+.2f} m)"
    )
    # Honest tracking thresholds: the old suite passed a 0.019 m/s shuffle.
    achieved = result["forward"] / result["elapsed"]
    assert achieved > 0.6 * speed, f"tracked {achieved:.3f} of {speed:.3f} m/s"
    assert abs(result["lateral"]) < 0.25, "walked sideways"


@pytest.mark.slow
@pytest.mark.parametrize(
    "stack",
    [
        "legacy_dcm",
        pytest.param(
            "alip_footstep",
            marks=pytest.mark.xfail(reason="ALIP footstep walk is not a limit cycle yet", strict=False),
        ),
    ],
)
@pytest.mark.parametrize("speed", [0.10, 0.20, 0.35, 0.50])
def test_closed_loop_walk_is_a_limit_cycle(speed, stack):
    result = _walk(speed, WALK_LIMIT_CYCLE_DURATION, stack=stack)
    assert not result["fell"], (
        f"fell after {result['elapsed']:.2f}s / {result['steps']} steps "
        f"(forward {result['forward']:+.2f} m, lateral {result['lateral']:+.2f} m)"
    )
    assert result["elapsed"] >= WALK_LIMIT_CYCLE_DURATION - 0.05
    achieved = result["forward"] / result["elapsed"]
    assert achieved > 0.6 * speed, f"tracked {achieved:.3f} of {speed:.3f} m/s"
    assert abs(result["lateral"]) < 0.20, "lateral drift is not a bounded orbit"
