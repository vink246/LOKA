#!/usr/bin/env python3
"""Run the G1 standing controller in MuJoCo.

    python -m loka.main                       # interactive viewer
    python -m loka.main --headless -T 10      # scripted run, prints a summary
    python -m loka.main --push 4              # shove the pelvis every 3 s
    python -m loka.main --height 0.60         # crouch

The controller is a centroidal MPC feeding a whole-body QP; see
``loka/control/stand.py``. Use ``python -m loka.evaluate`` for the full
robustness suite.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from loka.control.stand import DEFAULT_MODEL, StandConfig
from loka.recording import RecordingToggle
from loka.sim import Simulation, push_sequence

#: Redraw rate for the interactive viewer, independent of the control rate.
RENDER_HZ = 60.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML overriding the controller defaults")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--headless", action="store_true",
                        help="No viewer; run as fast as possible and report at the end")
    parser.add_argument("-T", "--duration", type=float, default=None,
                        help="Stop after this many seconds of simulated time")
    parser.add_argument("--height", type=float, default=None,
                        help="CoM height above the feet [m]; default is the model's stance")
    parser.add_argument("--push", type=float, default=0.0,
                        help="Impulse applied to the pelvis [N.s]")
    parser.add_argument("--push-period", type=float, default=3.0,
                        help="Seconds between pushes")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def build_simulation(args: argparse.Namespace) -> Simulation:
    config = StandConfig.from_yaml(args.config) if args.config else StandConfig()
    duration = args.duration or (30.0 if not args.headless else 10.0)
    pushes = (
        push_sequence(
            args.push,
            args.push_period,
            count=max(1, int(duration / args.push_period)),
            seed=args.seed,
        )
        if args.push > 0.0
        else []
    )
    sim = Simulation(config, model_path=args.model, pushes=pushes)
    if args.height is not None:
        sim.controller.command.height = args.height
    return sim


def run_headless(sim: Simulation, duration: float) -> int:
    started = time.perf_counter()
    stats = sim.run(duration)
    stats.realtime_factor = stats.duration / (time.perf_counter() - started)
    if stats.fell:
        print(f"FELL at t={sim.data.time:.2f}s", file=sys.stderr)
    print(stats.report())
    print(f"realtime factor : {stats.realtime_factor:.2f}x")
    return 1 if stats.fell else 0


def run_viewer(sim: Simulation, duration: float | None) -> int:
    """Drive the controller at full rate but redraw at :data:`RENDER_HZ`.

    ``viewer.sync()`` costs a few milliseconds, so calling it once per control
    tick would spend more time drawing than simulating and drop the loop to a
    fraction of realtime. Batching a frame's worth of control steps between
    redraws keeps the physics honest and the window smooth.
    """
    import mujoco.viewer

    steps_per_frame = max(1, round(1.0 / (RENDER_HZ * sim.config.control_dt)))
    frame_dt = steps_per_frame * sim.config.control_dt
    recording = RecordingToggle(sim.model, sim.config.model_path)
    print("[LOKA] Press R in the MuJoCo viewer to start/stop recording.")

    def key_callback(keycode: int) -> None:
        recording.key_callback(keycode, sim.data.time)

    try:
        with mujoco.viewer.launch_passive(
            sim.model,
            sim.data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=key_callback,
        ) as viewer:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            next_frame = time.perf_counter()
            while viewer.is_running():
                if duration is not None and sim.data.time >= duration:
                    break
                for _ in range(steps_per_frame):
                    sim.step()
                recording.maybe_capture(sim.data)
                viewer.sync()
                next_frame += frame_dt
                sleep = next_frame - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_frame = time.perf_counter()  # fell behind; resynchronise
    finally:
        recording.close()
    print(sim.stats().report())
    return 1 if sim.fell else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sim = build_simulation(args)
    print(
        f"G1 stand: MPC {1.0 / (sim.config.control_dt * sim.config.mpc_decimation):.0f} Hz"
        f" / WBC {1.0 / sim.config.control_dt:.0f} Hz"
        f" / physics {1.0 / sim.model.opt.timestep:.0f} Hz"
    )
    if args.headless:
        return run_headless(sim, args.duration or 10.0)
    return run_viewer(sim, args.duration)


if __name__ == "__main__":
    raise SystemExit(main())
