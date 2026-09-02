"""Walking diagnostics: report numbers, not verdicts.

``python -m loka.verify_gait`` prints, in order:

  1. the planner's own limit cycle, with the robot replaced by perfect
     tracking, so plan defects separate cleanly from tracking defects;
  2. closed-loop behaviour across a matrix of gait policies.

The layering matters because the two halves currently disagree: the plan is a
textbook DCM limit cycle that settles on the commanded velocity, while the
closed loop topples sideways after roughly 8-16 steps. Keeping them apart is
what stops a plan bug and a balance bug from being mistaken for one another.

This deliberately reports measurements rather than PASS/FAIL. The pass/fail
gates live in ``tests/test_gait.py``, where the open defect is a documented
xfail instead of a silently loosened threshold -- the previous version of both
files passed a 0.019 m/s shuffle with 3 mm of foot clearance.
"""

from __future__ import annotations

import numpy as np

from loka.control.gait import MODE_WALK, GaitConfig, GaitScheduler
from loka.sim import Simulation

HEIGHT = 0.70
FEET = np.array([[0.0, 0.1185, 0.0], [0.0, -0.1185, 0.0]])
PLANTED = np.ones(8, dtype=bool)


def plan_only(speed: float, *, ticks: int = 3000, dt: float = 0.002, **overrides):
    """Run the planner against a robot that tracks it exactly."""
    cfg = dict(
        mode=MODE_WALK,
        speed=speed,
        step_period=0.70,
        duty_factor=0.65,
        stance_width=0.24,
        capture_gain=1.0,
        walk_accel=50.0,
    )
    cfg.update(overrides)
    sched = GaitScheduler(GaitConfig(**cfg))

    com = np.array([0.0, 0.0, HEIGHT])
    vel = np.zeros(3)
    feet = FEET.copy()
    rows = []
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
        rows.append((com.copy(), vel.copy(), out.dcm_ref.copy()))
    return rows


def report_plan() -> None:
    print("1) Planner in isolation (robot replaced by perfect tracking)")
    print(f"   {'speed':>6}  {'settled vx':>10}  {'CoM sway':>9}  {'DCM sway':>9}  "
          f"{'y drift':>8}")
    for speed in (0.05, 0.10, 0.20, 0.30, 0.40):
        rows = plan_only(speed)
        tail = rows[-1000:]
        vx = float(np.mean([v[0] for _, v, _ in tail]))
        ys = np.array([c[1] for c, _, _ in tail])
        dcm = np.array([d[1] for _, _, d in tail])
        print(f"   {speed:6.2f}  {vx:10.3f}  {np.abs(ys).max():9.4f}  "
              f"{np.abs(dcm).max():9.4f}  {ys.mean():+8.4f}")
    print("   (settled vx should equal speed; sway bounded and centred on 0)")


def walk(speed: float, duration: float, **gait) -> dict:
    sim = Simulation()
    controller = sim.controller
    controller.set_task_targets({"gait.mode": "walk", "gait.speed": speed, **gait})

    start = np.array(sim.data.qpos[:2])
    steps, airborne, clearance, swing_err = 0, False, 0.0, 0.0
    while sim.data.time < duration:
        sim.step()
        telemetry = controller.telemetry
        if telemetry.swing_clearance > 0.0:
            clearance = max(clearance, telemetry.swing_clearance)
            swing_err = max(swing_err, telemetry.swing_error)
            if not airborne:
                steps += 1
            airborne = True
        else:
            airborne = False
        if sim.fell:
            break

    travel = np.array(sim.data.qpos[:2]) - start
    elapsed = float(sim.data.time)
    return dict(
        elapsed=elapsed,
        fell=bool(sim.fell),
        forward=float(travel[0]),
        lateral=float(travel[1]),
        speed=float(travel[0]) / max(elapsed, 1e-9),
        steps=steps,
        clearance=clearance,
        swing_err=swing_err,
    )


def report_closed_loop(duration: float = 10.0) -> None:
    print(f"\n2) Closed loop on the robot (up to {duration:.0f} s)")
    print(f"   {'policy':<34} {'t':>5} {'fell':>5} {'fwd':>7} {'lat':>7} "
          f"{'v':>6} {'n':>3} {'lift':>6} {'swerr':>6}")
    matrix = [
        ("speed=0.10", dict(speed=0.10)),
        ("speed=0.20", dict(speed=0.20)),
        ("speed=0.30", dict(speed=0.30)),
        ("speed=0.20 period=0.55", dict(speed=0.20, **{"gait.step_period": 0.55})),
        ("speed=0.20 period=0.90", dict(speed=0.20, **{"gait.step_period": 0.90})),
        ("speed=0.20 duty=0.55", dict(speed=0.20, **{"gait.duty_factor": 0.55})),
        ("speed=0.20 duty=0.75", dict(speed=0.20, **{"gait.duty_factor": 0.75})),
        ("speed=0.20 width=0.28", dict(speed=0.20, **{"gait.stance_width": 0.28})),
        ("speed=0.20 capture=1.0", dict(speed=0.20, **{"gait.capture_gain": 1.0})),
        ("speed=0.20 heading=0.3", dict(speed=0.20, **{"gait.heading": 0.3})),
    ]
    for label, spec in matrix:
        speed = spec.pop("speed")
        result = walk(speed, duration, **spec)
        print(f"   {label:<34} {result['elapsed']:5.2f} "
              f"{str(result['fell']):>5} {result['forward']:+7.3f} "
              f"{result['lateral']:+7.3f} {result['speed']:+6.3f} "
              f"{result['steps']:3d} {result['clearance']*1000:5.1f}mm "
              f"{result['swing_err']*1000:5.1f}mm")


def main() -> None:
    report_plan()
    report_closed_loop()
    print(
        "\nOpen defect: lateral CoM error compounds until the robot topples "
        "sideways.\nThe plan and the sagittal loop are sound; see docs/walking.md."
    )


if __name__ == "__main__":
    main()
