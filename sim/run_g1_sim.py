#!/usr/bin/env python3
"""Launch the unitree_mujoco Python simulator with G1 (29 DoF) defaults.

Configures the upstream simulate_python/config.py in-memory (no submodule edits),
then starts the MuJoCo viewer and DDS bridge on domain 1 / lo.

Elastic band (humanoid): press 9 to attach/release, 8 to lift, 7 to lower.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from threading import Thread

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_PY = REPO_ROOT / "unitree_mujoco" / "simulate_python"


def main() -> None:
    if not SIM_PY.is_dir():
        raise SystemExit(
            f"Missing submodule at {SIM_PY}. Run:\n"
            "  git submodule update --init --recursive"
        )

    os.chdir(SIM_PY)
    if str(SIM_PY) not in sys.path:
        sys.path.insert(0, str(SIM_PY))

    import config

    config.ROBOT = "g1"
    config.ROBOT_SCENE = "../unitree_robots/g1/scene_29dof.xml"
    config.DOMAIN_ID = 1
    config.INTERFACE = "lo"
    config.ENABLE_ELASTIC_BAND = True
    config.USE_JOYSTICK = 0
    config.PRINT_SCENE_INFORMATION = True

    # Import after mutating config so the bridge selects unitree_hg IDL.
    import unitree_mujoco as um

    print(
        "G1 sim running (29 DoF, domain_id=1, interface=lo, elastic band on).\n"
        "  Keys: 9 attach/release band, 8 lift, 7 lower.\n"
        "In another terminal: python sim/demos/g1_torque_smoke.py"
    )

    viewer_thread = Thread(target=um.PhysicsViewerThread, name="g1_viewer")
    sim_thread = Thread(target=um.SimulationThread, name="g1_sim")
    viewer_thread.start()
    sim_thread.start()
    viewer_thread.join()
    sim_thread.join()


if __name__ == "__main__":
    main()
