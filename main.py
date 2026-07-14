import argparse
import os
import sys
import time
import threading
import queue
from collections import deque

import mujoco
import mujoco.viewer
import mujoco_mpc
import numpy as np
from mujoco_mpc import agent as mpc_agent

from loka.compressor import (
    ANOMALY_COLLECTION_S,
    DEFAULT_TELEMETRY_SECTIONS,
    ERROR_TRIGGER_THRESHOLD,
    get_tracking_error,
    synthesize_generalized_telemetry,
    TelemetrySections,
    TELEMETRY_WINDOW_S,
)
from loka.model_state import apply_loka_mutations, zero_dead_actuator_commands
from loka.orchestrator import apply_scratchpad, llm_worker, load_system_prompt
from loka.robot_context import build_robot_model_context, capture_nominal_params, format_current_mpc_configuration
from loka.session import FailureEpisode, OperatorSession

LOKA_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_XML_PATH = os.path.join(LOKA_ROOT, "models", "walker", "task.xml")
OPERATOR_REQUEST_PREFIX = "loka:"


def start_operator_input_thread(request_queue: queue.Queue) -> None:
    """Read operator requests from stdin without blocking the sim loop."""

    def _reader():
        while True:
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if not line:
                break

            text = line.strip()
            if not text:
                continue
            if text.lower().startswith(OPERATOR_REQUEST_PREFIX):
                text = text[len(OPERATOR_REQUEST_PREFIX) :].strip()
            if text:
                request_queue.put(text)
                print(f"\n[LOKA] Queued operator request: {text}")

    threading.Thread(target=_reader, daemon=True).start()


def build_snapshot_telemetry(nominal_buffer, anomaly_buffer, model, sections):
    """Build telemetry from the most recent frames for operator requests."""
    recent_frames = int(TELEMETRY_WINDOW_S / model.opt.timestep)
    source = list(nominal_buffer) if nominal_buffer else list(anomaly_buffer)
    if not source:
        return "No telemetry frames available yet."

    recent = source[-recent_frames:] if len(source) > recent_frames else source
    baseline = list(nominal_buffer) if nominal_buffer else recent
    return synthesize_generalized_telemetry(
        baseline,
        deque(recent),
        model,
        sections=sections,
    )


def dispatch_to_orchestrator(
    user_turn,
    session,
    system_prompt,
    llm_queue,
    label,
):
    api_messages = session.compose_api_messages(system_prompt, user_turn)
    print(user_turn)
    print(f"\n[INFO] Dispatching to LOKA Orchestrator ({label})...")
    threading.Thread(
        target=llm_worker,
        args=(api_messages, llm_queue),
        daemon=True,
    ).start()
    return user_turn, session


def parse_args():
    parser = argparse.ArgumentParser(description="Run the LOKA MuJoCo MPC simulation.")
    parser.add_argument("--xml-path", default=DEFAULT_XML_PATH)
    parser.add_argument(
        "--telemetry-section-0",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TELEMETRY_SECTIONS.section_0_torso,
        help="Include section 0 (torso kinematic state) in LLM telemetry.",
    )
    parser.add_argument(
        "--telemetry-section-1",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TELEMETRY_SECTIONS.section_1_cost,
        help="Include section 1 (cost landscape differential) in LLM telemetry.",
    )
    parser.add_argument(
        "--telemetry-section-2",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TELEMETRY_SECTIONS.section_2_command_deficit,
        help="Include section 2 (actuator command deficit) in LLM telemetry.",
    )
    parser.add_argument(
        "--telemetry-section-3",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TELEMETRY_SECTIONS.section_3_directive,
        help="Include section 3 (orchestrator directive) in LLM telemetry.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    telemetry_sections = TelemetrySections(
        section_0_torso=args.telemetry_section_0,
        section_1_cost=args.telemetry_section_1,
        section_2_command_deficit=args.telemetry_section_2,
        section_3_directive=args.telemetry_section_3,
    )
    model = mujoco.MjModel.from_xml_path(args.xml_path)
    data = mujoco.MjData(model)
    agent = mpc_agent.Agent(task_id="Walker", model=model)
    robot_context = build_robot_model_context(model, args.xml_path)
    system_prompt = load_system_prompt(robot_context)
    enabled = [
        name
        for name, on in (
            ("0:torso", telemetry_sections.section_0_torso),
            ("1:cost", telemetry_sections.section_1_cost),
            ("2:cmd_deficit", telemetry_sections.section_2_command_deficit),
            ("3:directive", telemetry_sections.section_3_directive),
        )
        if on
    ]
    print(f"[LOKA] Telemetry sections enabled: {', '.join(enabled) or 'none'}")
    print(
        "[LOKA] Operator requests: type a message in this terminal and press Enter.\n"
        "       Optional prefix: loka: <request>\n"
        "       Example: try a slower hopping gait with more planner exploration"
    )

    operator_session = OperatorSession()
    operator_request_queue = queue.Queue()
    start_operator_input_thread(operator_request_queue)
    try:
        agent.set_task_parameter("Speed Goal", 1.0)
    except Exception as e:
        print(f"Warning: Could not set Speed Goal ({e}).")

    fault_state = {"active": False}
    right_hip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_hip")
    nominal_gears = model.actuator_gear[:, 0].copy()

    def apply_planning_model():
        """MPC internal model: nominal hardware plus LOKA mutations (no hidden faults)."""
        apply_loka_mutations(model, loka_state, nominal_gears)

    def apply_physical_faults():
        """Ground-truth hardware faults for physics — override planner model at sim time."""
        if fault_state["active"]:
            model.actuator_gear[right_hip_id, 0] = 0.0

    def enforce_faulted_controls(actions):
        if fault_state["active"]:
            actions[right_hip_id] = 0.0
        return actions

    def key_callback(keycode):
        try:
            if chr(keycode).lower() == "f":
                fault_state["active"] = not fault_state["active"]
                status = "INJECTED" if fault_state["active"] else "CLEARED"
                print(f"\n\n[KEYBOARD] Right hip fault {status}!")
        except ValueError:
            pass

    buffer_maxlen = int(TELEMETRY_WINDOW_S / model.opt.timestep)
    nominal_buffer = deque(maxlen=buffer_maxlen)
    anomaly_buffer = deque(maxlen=buffer_maxlen)

    COOLDOWN_PERIOD = 5.0
    last_dispatch_time = -COOLDOWN_PERIOD
    anomaly_collect_since = None

    planner_timestep = 0.01
    last_planner_time = 0.0

    llm_queue = queue.Queue()
    llm_is_busy = False
    failure_episode = None
    pending_user_turn = None
    pending_session = None
    queued_operator_requests = 0

    loka_state = {
        "mutations": [],
        "nominal_params": capture_nominal_params(model),
    }

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        mujoco.mj_resetData(model, data)

        while viewer.is_running():
            step_start = time.time()

            if not llm_queue.empty():
                worker_result = llm_queue.get()
                completed_session = pending_session
                if worker_result and pending_user_turn is not None and completed_session is not None:
                    completed_session.append_exchange(
                        pending_user_turn, worker_result["raw_yaml"]
                    )
                    agent = apply_scratchpad(
                        agent,
                        worker_result["scratchpad"],
                        loka_state,
                        model,
                        episode=completed_session,
                        data=data,
                    )
                pending_user_turn = None
                pending_session = None
                llm_is_busy = False
                if isinstance(completed_session, FailureEpisode):
                    anomaly_buffer.clear()
                if get_tracking_error(data) > ERROR_TRIGGER_THRESHOLD:
                    anomaly_collect_since = data.time

            if not llm_is_busy and not operator_request_queue.empty():
                operator_request = operator_request_queue.get()
                telemetry = build_snapshot_telemetry(
                    nominal_buffer, anomaly_buffer, model, telemetry_sections
                )
                mpc_config = format_current_mpc_configuration(agent, model)
                user_turn = operator_session.build_request_turn(
                    operator_request, telemetry, loka_state, data.time, mpc_config
                )
                pending_user_turn, pending_session = dispatch_to_orchestrator(
                    user_turn,
                    operator_session,
                    system_prompt,
                    llm_queue,
                    f"operator request, {operator_session.prior_turn_count} prior turn(s)",
                )
                llm_is_busy = True
                last_dispatch_time = data.time

            queued_operator_requests = operator_request_queue.qsize()

            apply_planning_model()

            qpos_before = data.qpos.copy()
            if data.time >= last_planner_time:
                agent.set_state(time=data.time, qpos=data.qpos, qvel=data.qvel, act=data.act)
                agent.planner_step()
                last_planner_time += planner_timestep

            planner_cmd = agent.get_action().copy()
            raw_actions = planner_cmd.copy()
            raw_actions = zero_dead_actuator_commands(raw_actions, loka_state)
            raw_actions = enforce_faulted_controls(raw_actions)

            apply_physical_faults()
            data.ctrl[:] = raw_actions
            mujoco.mj_step(model, data)
            joint_delta = np.abs(data.qpos - qpos_before)

            current_error = get_tracking_error(data)

            frame_data = {
                "time": data.time,
                "qpos": data.qpos.copy(),
                "qvel": data.qvel.copy(),
                "ctrl": data.ctrl.copy(),
                "planner_cmd": planner_cmd,
                "joint_delta": joint_delta,
            }

            anomaly_buffer.append(frame_data)
            if current_error == 0.0 and not llm_is_busy:
                nominal_buffer.append(frame_data)

            if (
                failure_episode is not None
                and current_error == 0.0
                and not llm_is_busy
            ):
                print(
                    f"\n[INFO] Failure episode recovered at t={data.time:.2f}s. "
                    "Closing LLM conversation thread."
                )
                failure_episode = None
                anomaly_collect_since = None

            in_failure = current_error > ERROR_TRIGGER_THRESHOLD
            if in_failure:
                if anomaly_collect_since is None:
                    anomaly_collect_since = data.time

                collection_ready = (
                    data.time - anomaly_collect_since
                ) >= ANOMALY_COLLECTION_S
                cooldown_ready = (
                    data.time - last_dispatch_time
                ) >= COOLDOWN_PERIOD

                if collection_ready and cooldown_ready and not llm_is_busy:
                    if failure_episode is None:
                        failure_episode = FailureEpisode(
                            data.time, nominal_baseline=list(nominal_buffer)
                        )

                    telemetry = synthesize_generalized_telemetry(
                        failure_episode.nominal_baseline,
                        anomaly_buffer,
                        model,
                        sections=telemetry_sections,
                    )
                    mpc_config = format_current_mpc_configuration(agent, model)

                    if failure_episode.round_number == 1:
                        user_turn = failure_episode.build_initial_user_turn(
                            telemetry, loka_state, data.time, mpc_config
                        )
                    else:
                        user_turn = failure_episode.build_user_turn(
                            telemetry, loka_state, data.time, mpc_config
                        )

                    pending_user_turn, pending_session = dispatch_to_orchestrator(
                        user_turn,
                        failure_episode,
                        system_prompt,
                        llm_queue,
                        (
                            f"failure episode round {failure_episode.round_number}, "
                            f"{failure_episode.prior_turn_count} prior turn(s)"
                        ),
                    )
                    llm_is_busy = True
                    last_dispatch_time = data.time
                    anomaly_collect_since = None
            else:
                anomaly_collect_since = None

            viewer.sync()

            if llm_is_busy:
                status_char = "[THINKING]"
            elif queued_operator_requests:
                status_char = f"[REQ Q={queued_operator_requests}]"
            elif in_failure:
                if anomaly_collect_since is not None:
                    collected = data.time - anomaly_collect_since
                    if collected < ANOMALY_COLLECTION_S:
                        status_char = (
                            f"[COLLECT {collected:.1f}/{ANOMALY_COLLECTION_S:.0f}s]"
                        )
                    else:
                        cooldown_left = COOLDOWN_PERIOD - (
                            data.time - last_dispatch_time
                        )
                        if cooldown_left > 0:
                            status_char = f"[COOLDOWN {cooldown_left:.1f}s]"
                        else:
                            status_char = "[FAIL]"
                else:
                    status_char = "[FAIL]"
            else:
                status_char = "[NOMINAL]"
            episode_tag = (
                f" epR{failure_episode.round_number}" if failure_episode else ""
            )
            sys.stdout.write(
                f"\rSim Time: {data.time:.2f}s | Err: {current_error:.3f} | "
                f"{status_char}{episode_tag} "
            )
            sys.stdout.flush()

            time_until_next = model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)


if __name__ == "__main__":
    main()
