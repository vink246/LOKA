#!/usr/bin/env python3
"""G1 torque smoke test against unitree_mujoco (unitree_hg LowCmd/LowState).

Equivalent purpose to unitree_mujoco/simulate_python/test/test_unitree_sdk2.py,
but for G1: subscribe lowstate, apply 1 Nm on all 29 motors.

Usage (sim running via python sim/run_g1_sim.py):
  python sim/demos/g1_torque_smoke.py
  python sim/demos/g1_torque_smoke.py --zero          # hold zero torque
  python sim/demos/g1_torque_smoke.py enp3s0          # real robot NIC
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow `python sim/demos/g1_torque_smoke.py` from repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.g1_interface import (  # noqa: E402
    DEFAULT_REAL_DOMAIN_ID,
    DEFAULT_SIM_DOMAIN_ID,
    DEFAULT_SIM_INTERFACE,
    LOKA_G1_Interface,
    NUM_MOTORS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "interface",
        nargs="?",
        default=None,
        help="Network interface for real robot (omit for sim: domain 1 / lo)",
    )
    parser.add_argument(
        "--zero",
        action="store_true",
        help="Send zero torque instead of 1 Nm",
    )
    parser.add_argument(
        "--print-every",
        type=float,
        default=1.0,
        help="Seconds between IMU / motor prints (0 to disable)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.interface:
        g1 = LOKA_G1_Interface(DEFAULT_REAL_DOMAIN_ID, args.interface)
        print(f"Real robot mode: domain={DEFAULT_REAL_DOMAIN_ID} iface={args.interface}")
    else:
        g1 = LOKA_G1_Interface(DEFAULT_SIM_DOMAIN_ID, DEFAULT_SIM_INTERFACE)
        print(f"Sim mode: domain={DEFAULT_SIM_DOMAIN_ID} iface={DEFAULT_SIM_INTERFACE}")

    tau = 0.0 if args.zero else 1.0
    torques = [tau] * NUM_MOTORS
    print(f"Publishing tau={tau} Nm on {NUM_MOTORS} motors at ~500 Hz (Ctrl+C to stop)")

    last_print = 0.0
    while True:
        g1.send_torques(torques)

        now = time.time()
        if args.print_every > 0 and (now - last_print) >= args.print_every:
            last_print = now
            imu = g1.state.imu_state
            m0 = g1.state.motor_state[0]
            # Bridge fills quat/gyro/acc (not rpy) from MuJoCo sensors.
            q = imu.quaternion
            g = imu.gyroscope
            a = imu.accelerometer
            print(
                f"IMU quat=({q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f})  "
                f"gyro=({g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f})  "
                f"acc=({a[0]:.3f}, {a[1]:.3f}, {a[2]:.3f})  "
                f"motor[0] q={m0.q:.3f} dq={m0.dq:.3f} tau_est={m0.tau_est:.3f}"
            )

        time.sleep(0.002)


if __name__ == "__main__":
    main()
