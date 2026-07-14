import numpy as np
import mujoco
from dataclasses import dataclass

# Stability deadband — error stays 0 inside these bands; LOKA triggers above ERROR_TRIGGER_THRESHOLD.
NOMINAL_HEIGHT_M = 1.15
HEIGHT_TOLERANCE_M = 0.10
PITCH_TOLERANCE_RAD = 0.35
SPEED_TOLERANCE_MPS = 0.25
ERROR_TRIGGER_THRESHOLD = 0.25

HEIGHT_GOAL_M = 1.2
SPEED_GOAL_MPS = 1.0

# Anomaly metrics window and minimum collection time before dispatching the LLM.
TELEMETRY_WINDOW_S = 2.0
ANOMALY_COLLECTION_S = 1.0

# Actuator command deficit: planner command vs actual joint motion (see _mean_actuator_cmd_deficit).
ACTUAL_DELTA_FLOOR = 1e-3
CMD_DEFICIT_DELTA_MIN = 8.0

LEG_CHAINS = (
    ("right_hip", "right_knee", "right_ankle"),
    ("left_hip", "left_knee", "left_ankle"),
)


@dataclass(frozen=True)
class TelemetrySections:
    """Toggle each telemetry block included in the LLM prompt."""

    section_0_torso: bool = True
    section_1_cost: bool = True
    section_2_command_deficit: bool = True
    section_3_directive: bool = True


DEFAULT_TELEMETRY_SECTIONS = TelemetrySections()


def get_tracking_error(data):
    torso_height = 1.3 + data.qpos[0]
    torso_pitch = data.qpos[2]
    forward_vel = data.qvel[1]

    height_error = 0.0
    if torso_height < (NOMINAL_HEIGHT_M - HEIGHT_TOLERANCE_M):
        height_error = (NOMINAL_HEIGHT_M - HEIGHT_TOLERANCE_M) - torso_height

    pitch_error = 0.0
    if abs(torso_pitch) > PITCH_TOLERANCE_RAD:
        pitch_error = abs(torso_pitch) - PITCH_TOLERANCE_RAD

    speed_error = 0.0
    if forward_vel < (SPEED_GOAL_MPS - SPEED_TOLERANCE_MPS):
        speed_error = (SPEED_GOAL_MPS - SPEED_TOLERANCE_MPS) - forward_vel

    return (height_error * 2.0) + pitch_error + speed_error


def extract_torso_state(buffer):
    """Mean planar-walker root (torso) kinematics over a frame buffer."""
    if not buffer:
        return None

    qpos_arr = np.array([f["qpos"] for f in buffer])
    qvel_arr = np.array([f["qvel"] for f in buffer])

    return {
        "height_m": float(np.mean(1.3 + qpos_arr[:, 0])),
        "forward_x_m": float(np.mean(qpos_arr[:, 1])),
        "pitch_rad": float(np.mean(qpos_arr[:, 2])),
        "forward_vel_mps": float(np.mean(qvel_arr[:, 1])),
        "vertical_vel_mps": float(np.mean(qvel_arr[:, 0])),
        "pitch_rate_radps": float(np.mean(qvel_arr[:, 2])),
    }


def extract_torso_snapshot(frame):
    """Instantaneous torso state from a single simulation frame."""
    return {
        "height_m": float(1.3 + frame["qpos"][0]),
        "forward_x_m": float(frame["qpos"][1]),
        "pitch_rad": float(frame["qpos"][2]),
        "forward_vel_mps": float(frame["qvel"][1]),
        "vertical_vel_mps": float(frame["qvel"][0]),
        "pitch_rate_radps": float(frame["qvel"][2]),
    }


def _torso_status_label(metric, current, nominal=None, tol_frac=0.15):
    if metric == "height_m" and current < (NOMINAL_HEIGHT_M - HEIGHT_TOLERANCE_M):
        return "[ERR: BELOW SAFE HEIGHT]"
    if metric == "pitch_rad" and abs(current) > PITCH_TOLERANCE_RAD:
        return "[ERR: EXCESSIVE LEAN]"
    if metric == "vertical_vel_mps" and current < -0.5:
        return "[ERR: FALLING]"
    if metric == "pitch_rate_radps" and abs(current) > 2.0:
        return "[ERR: UNSTABLE ROTATION]"
    if metric == "forward_vel_mps" and current < (SPEED_GOAL_MPS - SPEED_TOLERANCE_MPS):
        return "[ERR: SPEED LOSS]"

    if nominal is not None:
        delta = abs(current - nominal)
        ref = max(abs(nominal), 1e-3)
        if delta > ref * 2.0:
            return "[ERR: CRITICAL DEVIATION]"
        if delta > ref * tol_frac + 0.05:
            return "[ERR: MODERATE DEVIATION]"
        return "[NOMINAL]"

    return "[NOMINAL]"


def format_torso_state_section(anom_slice, nominal_buffer, has_baseline):
    anom_state = extract_torso_state(anom_slice)
    if anom_state is None:
        return ""

    snapshot = extract_torso_snapshot(anom_slice[-1])
    nom_state = extract_torso_state(nominal_buffer) if has_baseline else None

    lines = ["0. TORSO KINEMATIC STATE (Planar Walker Root)"]
    lines.append(
        "Root DOFs: height (Z), forward position (X), pitch angle, and their velocities."
    )
    lines.append(
        f"Window = mean over recent {TELEMETRY_WINDOW_S:.1f}s anomaly frames; "
        "Snapshot = value at trigger instant.\n"
    )

    metrics = [
        ("height_m", "Height", "m"),
        ("forward_x_m", "Forward Position", "m"),
        ("pitch_rad", "Pitch (Attitude)", "rad"),
        ("forward_vel_mps", "Forward Velocity", "m/s"),
        ("vertical_vel_mps", "Vertical Velocity", "m/s"),
        ("pitch_rate_radps", "Pitch Rate", "rad/s"),
    ]

    for key, label, unit in metrics:
        current = anom_state[key]
        snap = snapshot[key]
        if has_baseline and nom_state is not None:
            nominal = nom_state[key]
            status = _torso_status_label(key, current, nominal=nominal)
            lines.append(
                f"- {label}: Nominal {nominal:.2f} {unit} | "
                f"Current {current:.2f} {unit} | Snapshot {snap:.2f} {unit}  {status}"
            )
        else:
            status = _torso_status_label(key, current)
            goal_note = ""
            if key == "height_m":
                goal_note = f" (goal ~{HEIGHT_GOAL_M:.1f} m)"
            elif key == "forward_vel_mps":
                goal_note = f" (goal ~{SPEED_GOAL_MPS:.1f} m/s)"
            lines.append(
                f"- {label}: Current {current:.2f} {unit} | "
                f"Snapshot {snap:.2f} {unit}{goal_note}  {status}"
            )

    return "\n".join(lines) + "\n\n"


def extract_cost_proxies(buffer):
    if not buffer:
        return {"Height": 0.0, "Rotation": 0.0, "Speed": 0.0, "Control": 0.0}

    qpos_arr = np.array([f["qpos"] for f in buffer])
    qvel_arr = np.array([f["qvel"] for f in buffer])
    ctrl_arr = np.array([f["ctrl"] for f in buffer])

    heights = 1.3 + qpos_arr[:, 0]
    height_dev = np.mean(np.abs(1.2 - heights))
    pitch_dev = np.mean(np.abs(qpos_arr[:, 2]))
    speeds = qvel_arr[:, 1]
    speed_dev = np.mean(np.abs(1.0 - speeds))
    effort_dev = np.mean(np.abs(ctrl_arr))

    return {"Height": height_dev, "Rotation": pitch_dev, "Speed": speed_dev, "Control": effort_dev}


def _mean_actuator_cmd_deficit(frames, model):
    """Map actuator name -> mean |planner_cmd| / |actual joint delta|."""
    totals = {}
    counts = {}

    for frame in frames:
        cmd = frame.get("planner_cmd")
        joint_delta = frame.get("joint_delta")
        if cmd is None or joint_delta is None:
            continue

        for actuator_id in range(model.nu):
            if model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
                continue

            joint_id = model.actuator_trnid[actuator_id, 0]
            qpos_idx = model.jnt_qposadr[joint_id]
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)

            cmd_mag = abs(float(cmd[actuator_id]))
            act_mag = max(abs(float(joint_delta[qpos_idx])), ACTUAL_DELTA_FLOOR)
            ratio = cmd_mag / act_mag

            totals[name] = totals.get(name, 0.0) + ratio
            counts[name] = counts.get(name, 0) + 1

    return {name: totals[name] / counts[name] for name in totals}


def _chain_for_actuator(name):
    for chain in LEG_CHAINS:
        if name in chain:
            return chain
    return None


def _select_cmd_deficit_fault(anom_deficits, nom_deficits):
    """Pick upstream-most actuator with the largest command-deficit increase."""
    candidates = []
    for name, anom_ratio in anom_deficits.items():
        deficit_delta = anom_ratio - nom_deficits.get(name, 0.0)
        if deficit_delta > CMD_DEFICIT_DELTA_MIN:
            candidates.append({
                "name": name,
                "nominal": nom_deficits.get(name, 0.0),
                "current": anom_ratio,
                "deficit_delta": deficit_delta,
            })

    if not candidates:
        return None

    candidates.sort(key=lambda item: item["deficit_delta"], reverse=True)
    top = candidates[0]
    chain = _chain_for_actuator(top["name"])
    if chain is None:
        return top

    chain_candidates = [c for c in candidates if c["name"] in chain]
    for joint_name in chain:
        for candidate in chain_candidates:
            if candidate["name"] == joint_name:
                return candidate

    return top


def _format_joint_tracking_section(anom_slice, nominal_baseline, model, has_baseline):
    lines = [
        "2. ACTUATOR COMMAND DEFICIT (Planner Command vs Joint Motion)",
        "Compares MPC planner command magnitude to actual joint motion per actuator.",
        "Commands are captured before hidden fault overrides zero the actuator.",
        "Hidden physical faults are excluded from planning but enforced in simulation.",
        "Reports the upstream-most actuator on the limb with the largest deficit increase.",
        "",
    ]

    if not has_baseline:
        lines.append(
            "- No nominal baseline for command deficit. "
            "Cannot localize hardware faults without healthy reference gait.\n"
        )
        return "\n".join(lines) + "\n"

    anom_deficits = _mean_actuator_cmd_deficit(anom_slice, model)
    nom_deficits = _mean_actuator_cmd_deficit(nominal_baseline, model)
    fault = _select_cmd_deficit_fault(anom_deficits, nom_deficits)

    if fault is None:
        lines.append(
            "- No localized actuator command deficits detected. "
            "Instability may be global momentum or an unmodeled disturbance.\n"
        )
        return "\n".join(lines) + "\n"

    chain = _chain_for_actuator(fault["name"])
    if chain is not None:
        lines.append(f"Suspect limb chain: {', '.join(chain)}.\n")

    lines.append(f">> {fault['name']}")
    lines.append(
        f"- Command/Motion Ratio: Nominal {fault['nominal']:.1f} | "
        f"Current {fault['current']:.1f} (delta +{fault['deficit_delta']:.1f})"
    )
    lines.append(
        "- Trend Tag: [TRACKING_FAILURE] (Planner commands motion, joint does not respond)\n"
    )
    return "\n".join(lines) + "\n"


def _format_cost_section(anom_costs, nom_costs, has_baseline):
    lines = ["1. COST LANDSCAPE DIFFERENTIAL"]
    for cost_name in anom_costs.keys():
        a_val = anom_costs[cost_name]
        if has_baseline:
            n_val = nom_costs[cost_name]
            if a_val > (n_val * 2.0 + 0.05):
                status = "[ERR: CRITICAL DEVIATION]"
            elif a_val > (n_val * 1.3 + 0.02):
                status = "[ERR: MODERATE DEVIATION]"
            else:
                status = "[NOMINAL]"
            lines.append(
                f"- {cost_name}_Cost: Nominal {n_val:.2f} | Current {a_val:.2f}  {status}"
            )
        else:
            lines.append(
                f"- {cost_name}_Cost: Nominal [Unknown] | Current {a_val:.2f}  [NO BASELINE]"
            )
    return "\n".join(lines) + "\n\n"


def _format_directive_section(sections: TelemetrySections):
    sources = []
    if sections.section_0_torso:
        sources.append("torso state")
    if sections.section_1_cost:
        sources.append("cost landscape")
    if sections.section_2_command_deficit:
        sources.append("actuator command deficit tags")

    if not sources:
        source_text = "available telemetry"
    elif len(sources) == 1:
        source_text = sources[0]
    else:
        source_text = ", ".join(sources[:-1]) + f", and {sources[-1]}"

    lines = [
        "3. SYSTEM ORCHESTRATOR DIRECTIVE",
        f"Diagnose the root physical failure using the {source_text}. "
        "Update the YAML scratchpad to rewrite MPC parameters.\n",
    ]
    return "\n".join(lines) + "\n"


def synthesize_generalized_telemetry(
    nominal_baseline,
    anomaly_buffer,
    model,
    sections: TelemetrySections = DEFAULT_TELEMETRY_SECTIONS,
):
    if not anomaly_buffer:
        return "Error: No anomaly buffer available."

    print("\n" + "=" * 55)
    print("[SYSTEM] EXECUTING TELEMETRY COMPRESSION")
    print("=" * 55)

    recent_frames = int(TELEMETRY_WINDOW_S / model.opt.timestep)
    anom_slice = (
        list(anomaly_buffer)[-recent_frames:]
        if len(anomaly_buffer) > recent_frames
        else list(anomaly_buffer)
    )

    nominal_baseline = list(nominal_baseline or [])
    has_baseline = len(nominal_baseline) > 0

    prompt = "--- TELEMETRY ANOMALY REPORT ---\n"
    if has_baseline:
        prompt += (
            "The MPC is not behaving nominally. Compare the Nominal Baseline to the Current Anomaly.\n\n"
        )
        nom_costs = extract_cost_proxies(nominal_baseline)
    else:
        prompt += (
            "CRITICAL: Cold Start Failure. No nominal baseline exists. "
            "Evaluating against absolute safety limits.\n\n"
        )
        nom_costs = {}

    if sections.section_0_torso:
        prompt += format_torso_state_section(anom_slice, nominal_baseline, has_baseline)

    if sections.section_1_cost:
        anom_costs = extract_cost_proxies(anom_slice)
        prompt += _format_cost_section(anom_costs, nom_costs, has_baseline)

    if sections.section_2_command_deficit:
        prompt += _format_joint_tracking_section(
            anom_slice, nominal_baseline, model, has_baseline
        )

    if sections.section_3_directive:
        prompt += _format_directive_section(sections)

    return prompt
