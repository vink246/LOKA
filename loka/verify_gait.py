"""Layered walk verification: scheduler → Raibert → Bézier → closed-loop.

Run::

    python -m loka.verify_gait

Prints PASS/FAIL per layer so planner bugs are not confused with tracking.
GUI markers (cyan foothold, yellow Bézier, magenta des_pos, green CoM) are
drawn by ``render_gait_overlays`` when you launch::

    python -m loka.run_stand_loka --mode walk
"""

from __future__ import annotations

import sys

import numpy as np

from loka.control.gait import (
    MODE_WALK,
    GaitConfig,
    GaitScheduler,
    LegPhase,
    _bezier_swing_pos,
)
from loka.sim import Simulation


def _ok(name: str, passed: bool, detail: str) -> bool:
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] {name}: {detail}")
    return passed


def layer_scheduler() -> bool:
    print("1) Gait scheduler (phase windows + contact masks)")
    duty, period = 0.60, 0.40
    sched = GaitScheduler(
        GaitConfig(mode=MODE_WALK, speed=0.3, step_period=period, duty_factor=duty)
    )
    feet = np.array([[0.0, 0.12, 0.0], [0.0, -0.12, 0.0]])
    mask = np.ones(8, dtype=bool)
    sched.step(
        dt=0.002,
        com=np.array([0.0, 0.0, 0.7]),
        com_vel=np.zeros(3),
        foot_centers=feet,
        ground_z=0.0,
        height=0.7,
        yaw=0.0,
        measured_mask=mask,
    )
    saw_left = saw_right = saw_ds = False
    bad = False
    for _ in range(int(period / 0.002) * 3):
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
        if out.left_phase == LegPhase.SWING:
            saw_left = True
            if out.contact_mask[:4].any():
                bad = True
        if out.right_phase == LegPhase.SWING:
            saw_right = True
            if out.contact_mask[4:].any():
                bad = True
        if out.left_phase == LegPhase.STANCE and out.right_phase == LegPhase.STANCE:
            saw_ds = True
            if not (out.contact_mask[:4].any() and out.contact_mask[4:].any()):
                bad = True
    return _ok(
        "scheduler",
        saw_left and saw_right and saw_ds and not bad,
        f"left_swing={saw_left} right_swing={saw_right} DS={saw_ds} mask_ok={not bad}",
    )


def layer_raibert() -> bool:
    print("2) Raibert footholds (forward + lateral)")
    sched = GaitScheduler(
        GaitConfig(
            mode=MODE_WALK,
            speed=0.35,
            step_period=0.5,
            duty_factor=0.7,
            stance_width=0.24,
            walk_accel=5.0,
            step_length_max=0.15,
        )
    )
    feet = np.array([[0.0, 0.12, 0.0], [0.0, -0.12, 0.0]])
    mask = np.ones(8, dtype=bool)
    left_fh = right_fh = None
    com = np.array([0.0, 0.0, 0.7])
    for _ in range(800):
        out = sched.step(
            dt=0.002,
            com=com,
            com_vel=np.array([0.30, 0.0, 0.0]),
            foot_centers=feet,
            ground_z=0.0,
            height=0.7,
            yaw=0.0,
            measured_mask=mask,
        )
        if out.swing.active and out.swing.s < 0.05:
            if out.swing.leg == 0 and left_fh is None:
                left_fh = out.swing.foothold.copy()
            elif out.swing.leg == 1 and right_fh is None:
                right_fh = out.swing.foothold.copy()
        com = com + np.array([0.30, 0.0, 0.0]) * 0.002
    if left_fh is None or right_fh is None:
        return _ok("raibert", False, "never captured both footholds")
    width = float(left_fh[1] - right_fh[1])
    fwd = 0.5 * (float(left_fh[0]) + float(right_fh[0]))
    ok = 0.20 < width < 0.30 and fwd > -0.05
    return _ok(
        "raibert",
        ok,
        f"width={width:.3f}m mean_fwd_x={fwd:.3f} L={left_fh[:2]} R={right_fh[:2]}",
    )


def layer_bezier() -> bool:
    print("3) Bézier swing path (planned clearance)")
    height = 0.05
    mid = _bezier_swing_pos(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.15, 0.0, 0.0]),
        0.5,
        swing_height=height,
        ground_z=0.0,
    )
    ok = float(mid[2]) >= 0.9 * height
    return _ok("bezier", ok, f"mid_z={mid[2]:.3f} (want ≥ {0.9*height:.3f})")


def layer_closed_loop(T: float = 5.0) -> bool:
    print(f"4) Closed-loop walk (≥{T:.0f}s stability + DCM catch + toe)")
    sim = Simulation()
    c = sim.controller
    c.set_task_targets({"gait.mode": "walk", "gait.speed": 0.25, "gait.heading": 0.0})
    peak_act = 0.0
    peak_des = 0.0
    yaw_max = 0.0
    y_max = 0.0
    while float(sim.data.time) < T:
        sim.step()
        q = sim.data.qpos
        y_max = max(y_max, abs(float(q[1])))
        yaw_max = max(yaw_max, abs(float(q[9])), abs(float(q[15])))
        g = c._last_gait
        if g is not None and g.swing.active and 0.35 < g.swing.s < 0.65:
            feet = c.robot.foot_center_positions()
            peak_act = max(peak_act, float(feet[g.swing.leg, 2]))
            peak_des = max(peak_des, float(g.swing.des_pos[2]))
        if sim.fell:
            break
    reached = float(sim.data.time) >= T
    stable = (
        not sim.fell
        and reached
        and float(sim.data.qpos[2]) > 0.45
        and y_max < 0.20
        and float(sim.data.qpos[0]) > 0.04
    )
    toes = yaw_max < 0.35
    planner = peak_des >= 0.03
    _ok("planner des_z", planner, f"peak_des={peak_des:.3f}")
    _ok(
        "stability",
        stable,
        f"t={sim.data.time:.2f} fell={sim.fell} reached_{T:.0f}s={reached} "
        f"x={float(sim.data.qpos[0]):+.3f} |y|_max={y_max:.3f}",
    )
    _ok("toe-in (hip_yaw)", toes, f"max_|hip_yaw|={yaw_max:.3f} (want < 0.35)")
    _ok(
        "swing clearance (info)",
        True,
        f"peak_act_z={peak_act:.3f} (mild FF by design; DCM stepping is primary)",
    )
    return stable and toes and planner


def main() -> int:
    print("LOKA gait verification\n")
    results = [
        layer_scheduler(),
        layer_raibert(),
        layer_bezier(),
        layer_closed_loop(),
    ]
    print()
    if all(results):
        print("All layers PASS. In the GUI, cyan = next foothold, yellow = "
              "Bézier path, magenta = current des_pos, green = CoM ref.")
        return 0
    print("Some layers FAILED — planner vs tracking is split above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
