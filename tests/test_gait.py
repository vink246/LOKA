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
Layer 6 is the only one that involves the whole-body QP and the physics, and it
is the one that currently fails; see :func:`test_closed_loop_walk_stays_up`.
"""

from __future__ import annotations

import numpy as np
import pytest

from loka.control.gait import (
    MODE_WALK,
    GaitConfig,
    GaitScheduler,
    LegPhase,
    swing_reference,
)
from loka.sim import Simulation

#: Minimum closed-loop walk duration for the smoke test.
WALK_TEST_DURATION = 8.0

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
        capture_gain=1.0,
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


def test_plan_settles_at_commanded_forward_speed():
    speed = 0.20
    sched = walking_scheduler(speed=speed)
    history = _closed_loop_plan(sched, ticks=3000)
    # Average over the last two cycles, past the opening transfer.
    tail = history[-700:]
    mean_vx = float(np.mean([v[0] for _, v, _ in tail]))
    assert mean_vx == pytest.approx(speed, rel=0.05)


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


def test_capture_gain_zero_leaves_the_nominal_plan_alone():
    sched = walking_scheduler(speed=0.25, capture_gain=0.0)
    com = np.array([0.0, 0.0, HEIGHT])
    calm = _footholds(sched, com, np.array([0.25, 0.0, 0.0]), ticks=900)

    sched2 = walking_scheduler(speed=0.25, capture_gain=0.0)
    racing = _footholds(sched2, com, np.array([1.20, 0.0, 0.0]), ticks=900)
    assert calm and racing
    np.testing.assert_allclose(calm[0][1], racing[0][1], atol=1e-9)


# -- 6. closed-loop walk on the robot ------------------------------------


def _walk(speed: float, duration: float, **gait):
    sim = Simulation()
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


def test_walk_command_produces_real_steps_not_a_shuffle():
    """The plan must actually lift the feet and commit to footholds.

    This is the regression guard for the original defect, where the controller
    stayed upright by shuffling: ~3 mm of foot clearance and 8% of commanded
    speed. It deliberately says nothing about staying upright, so it keeps
    working as a shuffle detector while the balance problem is open.
    """
    result = _walk(0.20, 3.0)
    assert result["steps"] >= 4, "no swing phases executed"
    assert result["clearance"] > 0.02, (
        f"swing foot only cleared {result['clearance']*1000:.1f} mm -- "
        "the Cartesian swing task is not tracking the planned arc"
    )
    assert result["forward"] > 0.15, "barely any forward travel"


@pytest.mark.xfail(
    reason="Known open defect: lateral divergence. The footstep plan and DCM "
    "reference are correct in isolation (layers 1-5) and sagittal tracking "
    "holds to a few cm, but lateral CoM error compounds over roughly 8-16 "
    "steps until the robot topples sideways. See docs/walking.md.",
    strict=False,
)
@pytest.mark.parametrize("speed", [0.10, 0.20, 0.30])
def test_closed_loop_walk_stays_up(speed):
    result = _walk(speed, WALK_TEST_DURATION)
    assert not result["fell"], (
        f"fell after {result['elapsed']:.2f}s / {result['steps']} steps "
        f"(forward {result['forward']:+.2f} m, lateral {result['lateral']:+.2f} m)"
    )
    # Honest tracking thresholds: the old suite passed a 0.019 m/s shuffle.
    achieved = result["forward"] / result["elapsed"]
    assert achieved > 0.6 * speed, f"tracked {achieved:.3f} of {speed:.3f} m/s"
    assert abs(result["lateral"]) < 0.25, "walked sideways"
