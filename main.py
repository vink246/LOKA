import argparse
import sys
import time
import threading
import queue
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np
from mujoco_mpc import agent as mpc_agent

from loka.compressor import (
    ANOMALY_COLLECTION_S,
    DEFAULT_TELEMETRY_SECTIONS,
    get_tracking_error,
    synthesize_generalized_telemetry,
    TelemetrySections,
    TELEMETRY_WINDOW_S,
)
from loka.error_spec import format_error_spec
from loka.model_state import apply_loka_mutations, zero_dead_actuator_commands
from loka.orchestrator import apply_scratchpad, llm_worker, load_system_prompt
from loka.robot_context import (
    build_robot_model_context,
    capture_nominal_params,
    discover_capabilities,
    format_current_mpc_configuration,
)
from loka.session import FailureEpisode, OperatorSession
from loka.task_catalog import initial_error_spec, known_task_ids, resolve_xml_path

DEFAULT_TASK_ID = "Walker"
DEFAULT_OBJECTIVE = (
    "Sustained locomotion: maintain Height Goal (~1.2 m) and Speed Goal (1.0 m/s)."
)
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


def build_snapshot_telemetry(nominal_buffer, anomaly_buffer, model, sections, error_spec):
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
        error_spec=error_spec,
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


def rebuild_system_prompt(loka_state, robot_context, agent, model):
    capabilities = discover_capabilities(agent, model)
    return load_system_prompt(
        robot_context=robot_context,
        objective=loka_state.get("primary_objective"),
        capabilities=capabilities,
        error_spec=loka_state.get("error_spec"),
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
    return parser.parse_args()


def main():
    args = parse_args()
    xml_path = resolve_xml_path(args.task_id, args.xml_path)
    objective = args.objective
    if objective is None:
        objective = (
            DEFAULT_OBJECTIVE
            if args.task_id == "Walker"
            else f"Complete the '{args.task_id}' task successfully."
        )

    telemetry_sections = TelemetrySections(
        section_0_state=args.telemetry_section_0,
        section_1_cost=args.telemetry_section_1,
        section_2_motor_load=args.telemetry_section_2,
        section_3_directive=args.telemetry_section_3,
    )
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    agent = mpc_agent.Agent(task_id=args.task_id, model=model)
    robot_context = build_robot_model_context(model, xml_path)

    loka_state = {
        "mutations": [],
        "nominal_params": capture_nominal_params(model),
        "primary_objective": objective,
        "error_spec": initial_error_spec(args.task_id, model),
    }
    system_prompt = rebuild_system_prompt(loka_state, robot_context, agent, model)

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
    print(f"[LOKA] xml_path={xml_path}")
    print(f"[LOKA] objective: {objective}")
    print(f"[LOKA] model nq={model.nq} nv={model.nv} nu={model.nu}")
    print(f"[LOKA] Telemetry sections enabled: {', '.join(enabled) or 'none'}")
    print("[LOKA] Default Error_Tracking:\n" + format_error_spec(loka_state["error_spec"]))
    print(
        "[LOKA] Operator requests: type a message in this terminal and press Enter.\n"
        "       Optional prefix: loka: <request>\n"
        "       Example: walk crouched at 1 m/s"
    )

    operator_session = OperatorSession()
    operator_request_queue = queue.Queue()
    start_operator_input_thread(operator_request_queue)
    try:
        task_params = agent.get_task_parameters()
        if "Speed Goal" in task_params:
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
        if fault_state["active"] and right_hip_id != -1:
            model.actuator_gear[right_hip_id, 0] = 0.0

    def enforce_faulted_controls(actions):
        if fault_state["active"] and right_hip_id != -1:
            actions[right_hip_id] = 0.0
        return actions

    def key_callback(keycode):
        try:
            if chr(keycode).lower() == "f":
                if right_hip_id == -1:
                    print("\n\n[KEYBOARD] No right_hip actuator; fault inject skipped.")
                    return
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

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        mujoco.mj_resetData(model, data)

        while viewer.is_running():
            step_start = time.time()
            error_spec = loka_state["error_spec"]
            trigger_threshold = error_spec.trigger_threshold

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
                        task_id=args.task_id,
                    )
                    # Refresh injected capabilities / Error_Tracking after LLM updates.
                    system_prompt = rebuild_system_prompt(
                        loka_state, robot_context, agent, model
                    )
                pending_user_turn = None
                pending_session = None
                llm_is_busy = False
                if isinstance(completed_session, FailureEpisode):
                    anomaly_buffer.clear()
                error_spec = loka_state["error_spec"]
                if get_tracking_error(data, error_spec) > error_spec.trigger_threshold:
                    anomaly_collect_since = data.time

            if not llm_is_busy and not operator_request_queue.empty():
                operator_request = operator_request_queue.get()
                loka_state["primary_objective"] = operator_request
                system_prompt = rebuild_system_prompt(
                    loka_state, robot_context, agent, model
                )
                telemetry = build_snapshot_telemetry(
                    nominal_buffer,
                    anomaly_buffer,
                    model,
                    telemetry_sections,
                    loka_state["error_spec"],
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

            current_error = get_tracking_error(data, error_spec)

            frame_data = {
                "time": data.time,
                "qpos": data.qpos.copy(),
                "qvel": data.qvel.copy(),
                "ctrl": data.ctrl.copy(),
                "planner_cmd": planner_cmd,
                "joint_delta": joint_delta,
                "actuator_force": data.actuator_force.copy(),
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

            in_failure = current_error > trigger_threshold
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
                        error_spec=error_spec,
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
