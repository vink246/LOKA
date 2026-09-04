import argparse
import sys
import time

import mujoco
import mujoco.viewer

from loka.compressor import DEFAULT_TELEMETRY_SECTIONS, TelemetrySections
from loka.error_spec import format_error_spec
from loka.recording import (
    DEFAULT_RECORD_CAMERA,
    DEFAULT_RECORD_FPS,
    DEFAULT_RECORD_HEIGHT,
    DEFAULT_RECORD_PATH,
    DEFAULT_RECORD_WIDTH,
    VideoRecorder,
)
from loka.task_catalog import known_task_ids
from loka.walker_runtime import (
    DEFAULT_TASK_ID,
    WalkerRuntime,
    start_operator_input_thread,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the LOKA MuJoCo MPC simulation.")
    parser.add_argument(
        "--xml-path",
        default=None,
        help=(
            "Path to the MJPC task XML. If omitted, LOKA resolves stock XML from "
            "--task-id (Walker uses models/walker/task.xml; others use mujoco_mpc)."
        ),
    )
    parser.add_argument(
        "--task-id",
        default=DEFAULT_TASK_ID,
        help=(
            "MJPC task id passed to mujoco_mpc.Agent (default: Walker). "
            f"Known: {', '.join(known_task_ids())}."
        ),
    )
    parser.add_argument(
        "--objective",
        default=None,
        help="Primary mission text injected into the orchestrator system prompt.",
    )
    parser.add_argument(
        "--telemetry-section-0",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TELEMETRY_SECTIONS.section_0_state,
        help="Include section 0 (tracked state) in LLM telemetry.",
    )
    parser.add_argument(
        "--telemetry-section-1",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TELEMETRY_SECTIONS.section_1_cost,
        help="Include section 1 (cost / deviation landscape) in LLM telemetry.",
    )
    parser.add_argument(
        "--telemetry-section-2",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TELEMETRY_SECTIONS.section_2_motor_load,
        help="Include section 2 (motor load) in LLM telemetry.",
    )
    parser.add_argument(
        "--telemetry-section-3",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TELEMETRY_SECTIONS.section_3_directive,
        help="Include section 3 (orchestrator directive) in LLM telemetry.",
    )
    parser.add_argument(
        "--record",
        nargs="?",
        const=DEFAULT_RECORD_PATH,
        default=None,
        metavar="PATH",
        help=(
            "Record the sim from a tracking camera to a video file. "
            f"Optional PATH (default: {DEFAULT_RECORD_PATH})."
        ),
    )
    parser.add_argument(
        "--record-camera",
        default=DEFAULT_RECORD_CAMERA,
        help=(
            "MuJoCo camera used when --record is set "
            f"(default: {DEFAULT_RECORD_CAMERA}, side view that moves with the robot)."
        ),
    )
    parser.add_argument(
        "--record-fps",
        type=float,
        default=DEFAULT_RECORD_FPS,
        help=f"Recording frame rate (default: {DEFAULT_RECORD_FPS:g}).",
    )
    parser.add_argument(
        "--record-width",
        type=int,
        default=DEFAULT_RECORD_WIDTH,
        help=f"Recording width in pixels (default: {DEFAULT_RECORD_WIDTH}).",
    )
    parser.add_argument(
        "--record-height",
        type=int,
        default=DEFAULT_RECORD_HEIGHT,
        help=f"Recording height in pixels (default: {DEFAULT_RECORD_HEIGHT}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    telemetry_sections = TelemetrySections(
        section_0_state=args.telemetry_section_0,
        section_1_cost=args.telemetry_section_1,
        section_2_motor_load=args.telemetry_section_2,
        section_3_directive=args.telemetry_section_3,
    )
    runtime = WalkerRuntime(
        task_id=args.task_id,
        xml_path=args.xml_path,
        objective=args.objective,
        enable_llm=True,
        speed_goal=1.0,
        telemetry_sections=telemetry_sections,
    )

    enabled = [
        name
        for name, on in (
            ("0:state", telemetry_sections.section_0_state),
            ("1:cost", telemetry_sections.section_1_cost),
            ("2:motor_load", telemetry_sections.section_2_motor_load),
            ("3:directive", telemetry_sections.section_3_directive),
        )
        if on
    ]
    print(f"[LOKA] task_id={args.task_id}")
    print(f"[LOKA] xml_path={runtime.xml_path}")
    print(f"[LOKA] objective: {runtime.loka_state['primary_objective']}")
    print(
        f"[LOKA] model nq={runtime.model.nq} nv={runtime.model.nv} nu={runtime.model.nu}"
    )
    print(f"[LOKA] Telemetry sections enabled: {', '.join(enabled) or 'none'}")
    print(
        "[LOKA] Default Error_Tracking:\n"
        + format_error_spec(runtime.loka_state["error_spec"])
    )
    print(
        "[LOKA] Operator requests: type a message in this terminal and press Enter.\n"
        "       Optional prefix: loka: <request>\n"
        "       Example: walk crouched at 1 m/s"
    )

    recorder = None
    if args.record is not None:
        recorder = VideoRecorder(
            runtime.model,
            path=args.record,
            camera=args.record_camera,
            fps=args.record_fps,
            width=args.record_width,
            height=args.record_height,
        )
        print(
            f"[LOKA] Recording camera={args.record_camera} "
            f"({args.record_width}x{args.record_height} @ {args.record_fps:g} fps) "
            f"-> {args.record}"
        )

    start_operator_input_thread(runtime.operator_request_queue)

    def key_callback(keycode):
        try:
            if chr(keycode).lower() == "f":
                injected = runtime.toggle_right_hip_fault()
                status = "INJECTED" if injected else "CLEARED"
                print(f"\n\n[KEYBOARD] Right hip fault {status}!")
        except ValueError:
            pass

    try:
        with mujoco.viewer.launch_passive(
            runtime.model, runtime.data, key_callback=key_callback
        ) as viewer:
            runtime.reset()
            while viewer.is_running():
                step_start = time.time()
                step = runtime.step()
                if recorder is not None:
                    recorder.maybe_capture(runtime.data)
                viewer.sync()

                episode_tag = f" epR{step.failure_round}" if step.failure_round else ""
                sys.stdout.write(
                    f"\rSim Time: {step.time:.2f}s | Err: {step.error:.3f} | "
                    f"{step.status}{episode_tag} "
                )
                sys.stdout.flush()

                time_until_next = runtime.model.opt.timestep - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)
    finally:
        if recorder is not None:
            recorder.close()
        runtime.close()


if __name__ == "__main__":
    main()
