#!/usr/bin/env python3
"""The G1 locomotion plant with the LOKA orchestrator attached.

    python -m loka.run_loka
    python -m loka.run_loka --mode walk --speed 0.3
    python -m loka.run_loka --operator "lean left and crouch"
    python -m loka.run_loka --no-llm --task height=0.60,lean_x=0.03
    python -m loka.run_loka --fault mass --headless -T 12

``--operator`` is free-text language and requires the LLM (``OPENAI_API_KEY``).
``--no-llm`` ablations may only use structured ``--task`` setpoints — never
keyword-matched phrases. The control loop never blocks on the model.

To drive the same plant by hand, without any language in the loop, use
``python -m loka.dashboard`` instead.

Walk geometry (the footstep plan, DCM reference and swing arcs) is classical;
the LLM only sets high-level ``gait.*`` Task_Targets (mode, speed, heading,
cadence, …).

Session logs rewrite ``logs/loka/session.log`` (+ ``.jsonl``) each run.

Interactive viewer commands (stdin) and keys:
  mass / friction|ice / push / dead / clear / help
  Keys: 1 mass, 2 ice, 3 push, 4 dead knee, 5 clear, H help
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import mujoco

from loka.control.locomotion import DEFAULT_MODEL, LocomotionConfig, TASK_PARAMETER_NAMES
from loka.sim import Simulation
from loka.agent.faults import FaultSpec
from loka.agent.interactive import FAULT_HELP, parse_fault_command
from loka.agent.runtime import LokaConfig, LokaRuntime
from loka.agent.session_log import default_log_path
from loka.viz import begin_frame, render_fault_overlays, render_gait_overlays


def parse_task_spec(spec: str) -> dict[str, float]:
    """Parse ``height=0.60,lean_x=0.03`` into Task_Targets keys."""

    def _is_float_token(token: str) -> bool:
        try:
            float(token)
            return True
        except ValueError:
            return False

    updates: dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise argparse.ArgumentTypeError(
                f"Bad --task fragment {part!r}; expected name=value"
            )
        name, raw = part.split("=", 1)
        name = name.strip()
        if name not in TASK_PARAMETER_NAMES and name not in (
            "com_offset_x",
            "com_offset_y",
        ):
            raise argparse.ArgumentTypeError(
                f"Unknown task setpoint {name!r}; "
                f"allowed: {sorted(TASK_PARAMETER_NAMES)}"
            )
        if name == "gait.mode" and not _is_float_token(raw):
            updates[name] = raw.strip()
            continue
        try:
            updates[name] = float(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Non-numeric value for {name}: {raw!r}"
            ) from exc
    if not updates:
        raise argparse.ArgumentTypeError("--task was empty")
    return updates


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("-T", "--duration", type=float, default=None)
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Ablation: plant only (no language; use --task for structured setpoints)",
    )
    parser.add_argument(
        "--operator",
        type=str,
        default=None,
        help="Free-text operator request (requires LLM; queued at --operator-at)",
    )
    parser.add_argument(
        "--task",
        type=parse_task_spec,
        default=None,
        help="Structured Task_Targets for ablations, e.g. height=0.60,lean_x=0.03",
    )
    parser.add_argument("--operator-at", type=float, default=3.0)
    parser.add_argument(
        "--fault",
        choices=("none", "mass", "friction", "actuator_dead", "push"),
        default="none",
    )
    parser.add_argument("--fault-at", type=float, default=4.0)
    parser.add_argument("--fault-mu", type=float, default=0.25)
    parser.add_argument("--fault-mass", type=float, default=5.0)
    parser.add_argument("--fault-actuator", type=str, default="right_knee")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Session log directory (default: logs/loka; rewritten each run)",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable writing compressor/LLM session logs",
    )
    parser.add_argument(
        "--quiet-compressor",
        action="store_true",
        help="Do not print the full compressor user-turn to stdout (still logged)",
    )
    parser.add_argument(
        "--max-interventions",
        type=int,
        default=5,
        help="Max LLM interventions per failure episode before accepting residual "
        "(default: 5; plateau may stop earlier)",
    )
    parser.add_argument(
        "--mode",
        choices=("stand", "walk"),
        default="stand",
        help="Initial locomotion mode (walk engages gait clock + CP footsteps)",
    )
    parser.add_argument("--speed", type=float, default=0.25, help="Initial gait.speed if --mode walk")
    parser.add_argument(
        "--heading",
        type=float,
        default=0.0,
        help="Initial gait.heading [rad] world-frame travel direction",
    )
    args = parser.parse_args(argv)
    if args.operator and args.no_llm:
        parser.error(
            "--operator is language and requires the LLM; "
            "drop --no-llm, or use --task height=...,lean_x=... for ablations"
        )
    if args.operator and args.task:
        parser.error("pass only one of --operator or --task")
    return args


def build_faults(args: argparse.Namespace) -> list[FaultSpec]:
    if args.fault == "none":
        return []
    if args.fault == "mass":
        return [
            FaultSpec(
                "mass",
                args.fault_at,
                {"delta_kg": args.fault_mass, "body": "torso_link"},
            )
        ]
    if args.fault == "friction":
        return [FaultSpec("friction", args.fault_at, {"mu": args.fault_mu})]
    if args.fault == "actuator_dead":
        return [
            FaultSpec(
                "actuator_dead",
                args.fault_at,
                {"actuator": args.fault_actuator},
            )
        ]
    if args.fault == "push":
        return [
            FaultSpec(
                "push",
                args.fault_at,
                {"impulse": 6.0, "direction": (1.0, 0.0, 0.0)},
            )
        ]
    return []


def _handle_interactive_line(
    runtime: LokaRuntime, line: str, *, llm_enabled: bool
) -> None:
    parsed = parse_fault_command(line)
    if parsed == "help":
        print(FAULT_HELP)
        return
    if parsed == "clear":
        runtime.clear_interactive_faults()
        return
    if isinstance(parsed, FaultSpec):
        runtime.inject_fault_now(parsed)
        return
    text = line.strip()
    if not text:
        return
    if llm_enabled:
        runtime.submit_operator(text)
        print(f"[operator] queued: {text}")
    else:
        print(
            "[!] Not a fault command and LLM is off. "
            "Type 'help' for faults, or run without --no-llm for language."
        )


def _stdin_command_thread(
    runtime: LokaRuntime, stop: threading.Event, *, llm_enabled: bool
) -> None:
    print(FAULT_HELP)
    print("[interactive] type a fault command or operator request, then Enter")
    while not stop.is_set():
        try:
            line = sys.stdin.readline()
        except Exception:
            break
        if not line:
            break
        _handle_interactive_line(runtime, line, llm_enabled=llm_enabled)


def _make_key_callback(runtime: LokaRuntime):
    def key_callback(keycode: int) -> None:
        # GLFW keycodes: digits and letters are their Unicode code points.
        if keycode == ord("1"):
            runtime.inject_fault_now(
                FaultSpec("mass", 0.0, {"delta_kg": 5.0, "body": "torso_link"})
            )
        elif keycode == ord("2"):
            runtime.inject_fault_now(FaultSpec("friction", 0.0, {"mu": 0.2}))
        elif keycode == ord("3"):
            runtime.inject_fault_now(
                FaultSpec(
                    "push",
                    0.0,
                    {"impulse": 6.0, "direction": (1.0, 0.0, 0.0)},
                )
            )
        elif keycode == ord("4"):
            runtime.inject_fault_now(
                FaultSpec("actuator_dead", 0.0, {"actuator": "right_knee"})
            )
        elif keycode == ord("5"):
            runtime.clear_interactive_faults()
        elif keycode == ord("6"):
            runtime.inject_fault_now(
                FaultSpec(
                    "mass",
                    0.0,
                    {"delta_kg": 5.0, "body": "left_shoulder_roll_link"},
                )
            )
        elif keycode == ord("7"):
            runtime.inject_fault_now(
                FaultSpec(
                    "mass",
                    0.0,
                    {"delta_kg": 5.0, "body": "right_shoulder_roll_link"},
                )
            )
        elif keycode in (ord("h"), ord("H")):
            print(FAULT_HELP)

    return key_callback


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = LocomotionConfig.from_yaml(args.config) if args.config else LocomotionConfig()
    sim = Simulation(config, model_path=args.model)
    log_path = None
    if not args.no_log:
        log_path = default_log_path(args.log_dir)
    runtime = LokaRuntime(
        sim,
        LokaConfig(
            enable_llm=not args.no_llm,
            enable_session_log=not args.no_log,
            log_path=log_path,
            print_compressor=not args.quiet_compressor,
            max_interventions_per_episode=args.max_interventions,
        ),
        faults=build_faults(args),
    )
    if args.mode == "walk":
        applied = runtime.controller.set_task_targets(
            {
                "gait.mode": "walk",
                "gait.speed": float(args.speed),
                "gait.heading": float(args.heading),
            }
        )
        print(f"[gait] initial walk Task_Targets: {applied}")

    duration = args.duration or (20.0 if args.headless else None)
    stop = threading.Event()

    if args.operator or args.task:
        def _queue_later():
            while not stop.is_set() and sim.data.time < args.operator_at:
                time.sleep(0.05)
            if stop.is_set():
                return
            if args.task is not None:
                applied = runtime.controller.set_task_targets(args.task)
                print(f"[ablation] Task_Targets applied: {applied}")
            else:
                runtime.submit_operator(args.operator)

        threading.Thread(target=_queue_later, daemon=True).start()

    print(
        f"G1 stand + LOKA: LLM={'on' if not args.no_llm else 'off'}  "
        f"fault={args.fault}"
    )

    if args.headless:
        history = runtime.run(duration or 15.0)
        fell = sim.fell
        print(sim.stats().report())
        print(f"tracking samples: {len(history)}  fell={fell}")
        return 1 if fell else 0

    import mujoco.viewer
    from loka.main import RENDER_HZ

    steps_per_frame = max(1, round(1.0 / (RENDER_HZ * sim.config.control_dt)))
    frame_dt = steps_per_frame * sim.config.control_dt
    stdin_stop = threading.Event()
    if sys.stdin.isatty():
        threading.Thread(
            target=_stdin_command_thread,
            args=(runtime, stdin_stop),
            kwargs={"llm_enabled": not args.no_llm},
            daemon=True,
        ).start()

    try:
        with mujoco.viewer.launch_passive(
            sim.model,
            sim.data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=_make_key_callback(runtime),
        ) as viewer:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            next_frame = time.perf_counter()
            while viewer.is_running():
                if duration is not None and sim.data.time >= duration:
                    break
                for _ in range(steps_per_frame):
                    runtime.step()
                    if sim.fell:
                        break
                begin_frame(viewer)
                render_fault_overlays(viewer, sim.model, sim.data, runtime.viz)
                render_gait_overlays(viewer, runtime.controller)
                viewer.sync()
                next_frame += frame_dt
                sleep = next_frame - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_frame = time.perf_counter()
                if sim.fell:
                    print(f"FELL at t={sim.data.time:.2f}s", file=sys.stderr)
                    break
    finally:
        stop.set()
        stdin_stop.set()

    print(sim.stats().report())
    return 1 if sim.fell else 0


if __name__ == "__main__":
    raise SystemExit(main())
