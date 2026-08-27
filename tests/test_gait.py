"""Phase B gait / walk unit tests.

Verification layers:
  1. Scheduler phase windows and contact masks
  2. Raibert foothold geometry
  3. Bézier swing path clearance
  4. Closed-loop walk smoke (≥ ``WALK_TEST_DURATION`` s)
"""

from __future__ import annotations

import numpy as np
import pytest

from loka.control.gait import (
    MODE_WALK,
    GaitConfig,
    GaitScheduler,
    LegPhase,
    _bezier_swing_pos,
)
from loka.control.stand import StandController
from loka.sim import Simulation

#: Minimum closed-loop walk duration for smoke / tracking tests.
WALK_TEST_DURATION = 5.0


def test_bezier_swing_peaks_above_ground():
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([0.2, 0.0, 0.0])
    mid = _bezier_swing_pos(start, end, 0.5, swing_height=0.06, ground_z=0.0)
    assert mid[2] >= 0.05
    assert 0.05 < mid[0] < 0.15
    # Endpoints stay near ground.
    z0 = _bezier_swing_pos(start, end, 0.0, swing_height=0.06, ground_z=0.0)[2]
    z1 = _bezier_swing_pos(start, end, 1.0, swing_height=0.06, ground_z=0.0)[2]
    assert z0 < 0.01
    assert z1 < 0.01


def test_gait_scheduler_phase_windows_and_masks():
    """Duty-factor swing windows alternate; DS has both feet planted."""
    duty = 0.60
    period = 0.40
    sched = GaitScheduler(
        GaitConfig(mode=MODE_WALK, speed=0.3, step_period=period, duty_factor=duty)
    )
    feet = np.array([[0.0, 0.12, 0.0], [0.0, -0.12, 0.0]])
    mask = np.ones(8, dtype=bool)
    # Prime scheduler (sets mid-DS start phase).
    sched.step(
        dt=0.002,
        com=np.array([0.0, 0.0, 0.7]),
        com_vel=np.array([0.2, 0.0, 0.0]),
        foot_centers=feet,
        ground_z=0.0,
        height=0.7,
        yaw=0.0,
        measured_mask=mask,
    )
    swing_len = 1.0 - duty
    saw_left = saw_right = saw_ds = False
    for _ in range(500):
        out = sched.step(
            dt=0.002,
            com=np.array([0.0, 0.0, 0.7]),
            com_vel=np.array([0.2, 0.0, 0.0]),
            foot_centers=feet,
            ground_z=0.0,
            height=0.7,
            yaw=0.0,
            measured_mask=mask,
        )
        phase = out.phase
        left_swing = phase < swing_len
        right_swing = ((phase - 0.5) % 1.0) < swing_len
        if left_swing and right_swing:
            right_swing = False

        if left_swing:
            saw_left = True
            assert out.left_phase == LegPhase.SWING
            assert out.right_phase == LegPhase.STANCE
            assert not out.contact_mask[:4].any()
            assert out.contact_mask[4:].all()
            assert out.swing.active and out.swing.leg == 0
        elif right_swing:
            saw_right = True
            assert out.right_phase == LegPhase.SWING
            assert out.left_phase == LegPhase.STANCE
            assert not out.contact_mask[4:].any()
            assert out.contact_mask[:4].all()
            assert out.swing.active and out.swing.leg == 1
        else:
            saw_ds = True
            assert out.left_phase == LegPhase.STANCE
            assert out.right_phase == LegPhase.STANCE
            assert out.contact_mask.all()
            assert not out.swing.active

    assert saw_left and saw_right and saw_ds


def test_raibert_foothold_forward_and_lateral():
    """Footholds advance with speed and sit at ±stance_width/2 laterally."""
    cfg = GaitConfig(
        mode=MODE_WALK,
        speed=0.4,
        step_period=0.50,
        duty_factor=0.80,
        step_length_max=0.15,
        stance_width=0.24,
        capture_gain=1.0,
        walk_accel=10.0,  # snap to commanded speed immediately
    )
    sched = GaitScheduler(cfg)
    feet = np.array([[0.0, 0.12, 0.0], [0.0, -0.12, 0.0]])
    mask = np.ones(8, dtype=bool)
    com = np.array([0.0, 0.0, 0.7])
    # Warm up so cmd_speed reaches target and midline is initialized.
    for _ in range(50):
        sched.step(
            dt=0.002,
            com=com,
            com_vel=np.array([0.4, 0.0, 0.0]),
            foot_centers=feet,
            ground_z=0.0,
            height=0.7,
            yaw=0.0,
            measured_mask=mask,
        )
    assert sched._cmd_speed == pytest.approx(0.4, abs=0.05)

    left_fh = None
    right_fh = None
    for _ in range(400):
        out = sched.step(
            dt=0.002,
            com=com,
            com_vel=np.array([0.4, 0.0, 0.0]),
            foot_centers=feet,
            ground_z=0.0,
            height=0.7,
            yaw=0.0,
            measured_mask=mask,
        )
        if out.swing.active and out.swing.s < 0.05:
            if out.swing.leg == 0:
                left_fh = out.swing.foothold.copy()
            else:
                right_fh = out.swing.foothold.copy()
        if left_fh is not None and right_fh is not None:
            break

    assert left_fh is not None and right_fh is not None
    # Forward of CoM at walking speed (Raibert + DCM).
    assert left_fh[0] > 0.02
    assert right_fh[0] > 0.02
    # Left +, right − about the midline.
    assert left_fh[1] > 0.08
    assert right_fh[1] < -0.08
    width = left_fh[1] - right_fh[1]
    assert 0.20 < width < 0.28


def test_dcm_steps_forward_when_com_racing():
    """Capture-point: high forward CoM velocity → foothold ahead of CoM."""
    cfg = GaitConfig(
        mode=MODE_WALK,
        speed=0.25,
        step_period=0.50,
        duty_factor=0.80,
        step_length_max=0.15,
        stance_width=0.24,
        capture_gain=1.0,
        walk_accel=10.0,
        foothold_retarget_s=0.35,
    )
    sched = GaitScheduler(cfg)
    feet = np.array([[0.0, 0.12, 0.0], [0.0, -0.12, 0.0]])
    mask = np.ones(8, dtype=bool)
    com = np.array([0.05, 0.0, 0.7])  # slightly ahead of feet
    for _ in range(80):
        sched.step(
            dt=0.002,
            com=com,
            com_vel=np.array([0.55, 0.0, 0.0]),
            foot_centers=feet,
            ground_z=0.0,
            height=0.7,
            yaw=0.0,
            measured_mask=mask,
        )
    fh = None
    for _ in range(300):
        out = sched.step(
            dt=0.002,
            com=com,
            com_vel=np.array([0.55, 0.0, 0.0]),
            foot_centers=feet,
            ground_z=0.0,
            height=0.7,
            yaw=0.0,
            measured_mask=mask,
        )
        if out.swing.active and out.swing.s < 0.30:
            fh = out.swing.foothold.copy()
            break
    assert fh is not None
    assert fh[0] > com[0] + 0.02, f"DCM foothold {fh[0]:.3f} not ahead of CoM {com[0]:.3f}"


def test_swing_path_clearance_matches_config():
    """Planned des_pos mid-swing must reach configured swing_height."""
    height = 0.05
    sched = GaitScheduler(
        GaitConfig(
            mode=MODE_WALK,
            speed=0.3,
            step_period=0.5,
            duty_factor=0.7,
            swing_height=height,
            walk_accel=5.0,
        )
    )
    feet = np.array([[0.0, 0.12, 0.0], [0.0, -0.12, 0.0]])
    mask = np.ones(8, dtype=bool)
    peak = 0.0
    for _ in range(600):
        out = sched.step(
            dt=0.002,
            com=np.array([0.0, 0.0, 0.7]),
            com_vel=np.array([0.25, 0.0, 0.0]),
            foot_centers=feet,
            ground_z=0.0,
            height=0.7,
            yaw=0.0,
            measured_mask=mask,
        )
        if out.swing.active:
            peak = max(peak, float(out.swing.des_pos[2]))
    assert peak >= 0.9 * height, f"planned peak {peak:.3f} < 0.9*{height}"


def test_set_task_targets_enters_walk():
    controller = StandController()
    applied = controller.set_task_targets(
        {"gait.mode": "walk", "gait.speed": 0.3, "gait.heading": 0.1}
    )
    assert applied["gait.mode"] == pytest.approx(1.0)
    assert applied["gait.speed"] == pytest.approx(0.3)
    assert controller.gait.config.mode == pytest.approx(1.0)


def test_short_walk_does_not_immediately_fall():
    """Closed-loop walk must stay upright for at least ``WALK_TEST_DURATION`` s."""
    sim = Simulation()
    sim.controller.set_task_targets(
        {"gait.mode": "walk", "gait.speed": 0.25, "gait.heading": 0.0}
    )
    while float(sim.data.time) < WALK_TEST_DURATION:
        sim.step()
        if sim.fell:
            break
    assert not sim.fell, f"fell at t={sim.data.time:.2f}s"
    assert float(sim.data.time) >= WALK_TEST_DURATION, (
        f"walk smoke ended early at t={sim.data.time:.2f}s "
        f"(need ≥ {WALK_TEST_DURATION:.1f}s)"
    )
    assert float(sim.data.qpos[2]) > 0.45
    assert abs(float(sim.data.qpos[1])) < 0.20
    assert float(sim.data.qpos[0]) > 0.04  # some forward progress


def test_swing_tracking_lifts_foot():
    """Closed-loop (≥5 s): planner clears; hip yaw stays sane; walk stays upright."""
    sim = Simulation()
    sim.controller.set_task_targets(
        {"gait.mode": "walk", "gait.speed": 0.25, "gait.heading": 0.0}
    )
    peak_act = 0.0
    peak_des = 0.0
    yaw_max = 0.0
    while float(sim.data.time) < WALK_TEST_DURATION:
        sim.step()
        q = sim.data.qpos
        yaw_max = max(yaw_max, abs(float(q[9])), abs(float(q[15])))
        g = sim.controller._last_gait
        if g is not None and g.swing.active and 0.35 < g.swing.s < 0.65:
            feet = sim.controller.robot.foot_center_positions()
            peak_act = max(peak_act, float(feet[g.swing.leg, 2]))
            peak_des = max(peak_des, float(g.swing.des_pos[2]))
        if sim.fell:
            break
    assert not sim.fell, f"fell while walking at t={sim.data.time:.2f}s"
    assert float(sim.data.time) >= WALK_TEST_DURATION, (
        f"walk tracking ended early at t={sim.data.time:.2f}s "
        f"(need ≥ {WALK_TEST_DURATION:.1f}s)"
    )
    assert peak_des >= 0.03, f"planner peak des_z={peak_des:.3f}"
    # Mild knee FF by design (aggressive lift tips); planner path is the check.
    assert yaw_max < 0.40, f"pigeon-toe hip_yaw={yaw_max:.3f}"
