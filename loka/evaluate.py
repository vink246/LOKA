#!/usr/bin/env python3
"""Robustness suite for the standing controller.

    python -m loka.evaluate

Reports three things:

* **hold** -- how tightly the CoM and torso stay on target when undisturbed.
* **push** -- the largest impulse the robot survives from each direction,
  found by bisection. This is the headline robustness number.
* **crouch** -- height-tracking error across the usable stance range.

Useful as a regression check after touching the controller, and as the metric
LOKA optimises against when it starts mutating weights.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from loka.control.locomotion import LocomotionConfig
from loka.control.stacks import add_stack_argument, apply_stack_arg
from loka.sim import Push, Simulation

DIRECTIONS = {
    "forward": np.array([1.0, 0.0, 0.0]),
    "backward": np.array([-1.0, 0.0, 0.0]),
    "left": np.array([0.0, 1.0, 0.0]),
    "right": np.array([0.0, -1.0, 0.0]),
    "diagonal": np.array([0.707, 0.707, 0.0]),
}


def survives_push(
    config: LocomotionConfig, direction: np.ndarray, impulse: float, settle: float = 3.0
) -> bool:
    """True if the robot is still standing ``settle`` seconds after the push."""
    push = Push(impulse=impulse, direction=direction, time=1.0)
    sim = Simulation(config, pushes=[push])
    sim.run(1.0 + settle)
    if sim.fell:
        return False
    # Also require it to have actually recovered, not merely be mid-topple.
    return bool(np.linalg.norm(sim.controller.telemetry.com_error) < 0.05)


def max_recoverable_impulse(
    config: LocomotionConfig,
    direction: np.ndarray,
    low: float = 0.0,
    high: float = 40.0,
    tolerance: float = 0.5,
) -> float:
    """Bisect for the largest survivable impulse in one direction [N.s]."""
    if not survives_push(config, direction, low + tolerance):
        return 0.0
    while high - low > tolerance:
        mid = 0.5 * (low + high)
        if survives_push(config, direction, mid):
            low = mid
        else:
            high = mid
    return low


def evaluate_hold(config: LocomotionConfig, duration: float = 10.0) -> None:
    sim = Simulation(config)
    started = time.perf_counter()
    stats = sim.run(duration)
    stats.realtime_factor = stats.duration / (time.perf_counter() - started)
    print("hold (no disturbance)")
    print("  " + stats.report().replace("\n", "\n  "))
    print(f"  realtime factor : {stats.realtime_factor:.2f}x")


def evaluate_pushes(config: LocomotionConfig) -> None:
    print("\npush recovery (max survivable impulse)")
    mass = Simulation(config).controller.robot.total_mass
    for name, direction in DIRECTIONS.items():
        impulse = max_recoverable_impulse(config, direction)
        print(
            f"  {name:<9}: {impulse:5.1f} N.s   "
            f"(equivalent to {impulse / mass:.2f} m/s of CoM velocity)"
        )


def evaluate_crouch(config: LocomotionConfig) -> None:
    print("\nheight tracking")
    nominal = Simulation(config).controller.nominal_height
    for height in (nominal, nominal - 0.05, nominal - 0.10, nominal - 0.15):
        sim = Simulation(config)
        sim.controller.command.height = height
        stats = sim.run(4.0)
        achieved = sim.controller.telemetry.com[2]
        target = sim.controller.telemetry.com_reference[2]
        status = (
            "FELL"
            if stats.fell
            else f"reached {achieved:.3f} m (error {(achieved - target) * 1e3:+6.1f} mm)"
        )
        print(f"  commanded {height:.3f} m : {status}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-pushes", action="store_true")
    add_stack_argument(parser)
    args = parser.parse_args(argv)

    config = LocomotionConfig()
    apply_stack_arg(config, args)
    evaluate_hold(config)
    if not args.skip_pushes:
        evaluate_pushes(config)
    evaluate_crouch(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
