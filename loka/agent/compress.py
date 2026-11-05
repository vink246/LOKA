"""Telemetry compression: recent plant frames into an LLM-facing report."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from loka.agent.loka_config import loka_config

_SESSION = loka_config()["session"]
#: Seconds of frames summarised in a report.
TELEMETRY_WINDOW_S = float(_SESSION["telemetry_window_s"])


def _window_frames(buffer: Sequence[dict], control_dt: float) -> list[dict]:
    if not buffer:
        return []
    n = max(1, int(TELEMETRY_WINDOW_S / max(control_dt, 1e-3)))
    # recent_frames is a deque and does not support slicing
    frames = list(buffer)
    return frames[-n:]


def _gait_band_line(stack: str = "legacy_dcm") -> str:
    """Safe gait band from the coarse sweep. apply_updates clips to this."""
    return (
        "  gait safe band (out of range is clipped; defaults are the 0.25 m/s walk): "
        + gait_band_text(stack)
    )


def gait_band_text(stack: str = "legacy_dcm") -> str:
    """Each gait knob's clip range and default, ``; ``-separated."""
    from loka.control.gait import gait_knobs, gait_schedule_for_speed

    sched = gait_schedule_for_speed(0.25)
    bits = []
    for knob in gait_knobs(stack):
        if knob.name == "gait.mode":
            continue
        default = sched.get(knob.name)
        if default is None:
            default = {
                "gait.speed": 0.25,
                "gait.heading": 0.0,
                "gait.step_length_max": 0.30,
                "gait.swing_height": 0.045,
                "gait.capture_gain": 0.7,
                "gait.walk_accel": 0.5,
                "gait.foothold_retarget_s": 0.70,
                "gait.turn_rate": 0.40,
            }.get(knob.name)
        extra = f" default {default:.3g}" if default is not None else ""
        bits.append(f"{knob.name} [{knob.low:.3g}, {knob.high:.3g}]{extra}")
    return "; ".join(bits)


def _progress_lines(latest: dict, qpos: np.ndarray, qvel: np.ndarray) -> list[str]:
    """Heading-frame speed, and closing speed when a world goal is active.

    ``planar speed`` has no sign. A walk that has already left the origin can
    still be going backwards; closing speed is the component toward the goal.
    """
    from loka.control.gait import heading_frame

    walking = bool(latest.get("walking", False))
    goal_on = bool(latest.get("goal_active", False))
    if not walking and not goal_on:
        return []
    heading = float(latest.get("heading_goal", latest.get("cmd_heading", 0.0)) or 0.0)
    forward, _ = heading_frame(heading)
    along = float(np.asarray(qvel[:2], dtype=float) @ forward)
    lines = [
        f"  along heading [m/s]: {along:+.3f}"
        "  (positive = commanded walk direction; negative = walking backwards)",
    ]
    if not goal_on:
        return lines
    goal = np.array(
        [float(latest.get("goal_x", 0.0)), float(latest.get("goal_y", 0.0))],
        dtype=float,
    )
    delta = goal - np.asarray(qpos[:2], dtype=float)
    dist = float(np.linalg.norm(delta))
    closing = float(np.asarray(qvel[:2], dtype=float) @ delta / dist) if dist > 1e-4 else 0.0
    if closing < -0.02:
        tag = "MOVING AWAY FROM GOAL"
    elif along < -0.02:
        tag = "WALKING BACKWARDS"
    else:
        tag = "toward goal"
    lines.append(
        f"  goal xy [m]: {goal[0]:+.3f} {goal[1]:+.3f}  distance {dist:.3f} m"
    )
    lines.append(
        f"  closing speed [m/s]: {closing:+.3f}"
        f"  (positive = toward the goal)  {tag}"
    )
    return lines


def format_full_state_section(frames: Sequence[dict]) -> str:
    """Attitude, position, and velocity at the trigger, plus the gait command."""
    if not frames:
        return "FULL STATE\n  (no frames)\n"
    latest = frames[-1]
    qpos = np.asarray(latest.get("qpos", np.zeros(7)), dtype=float)
    qvel = np.asarray(latest.get("qvel", np.zeros(6)), dtype=float)
    rpy = np.asarray(latest.get("rpy", np.zeros(3)), dtype=float)
    quat = qpos[3:7] if qpos.size >= 7 else np.zeros(4)
    lines = [
        "FULL STATE (snapshot at this invoke)",
        f"  t={float(latest.get('time', 0.0)):.2f}s",
        f"  pelvis xyz [m] (x is pitch, y is roll, z is height): {qpos[0]:+.3f} {qpos[1]:+.3f} {qpos[2]:+.3f}",
        f"  attitude [deg] roll=y pitch=x yaw=heading: {np.degrees(rpy[0]):+.1f} "
        f"{np.degrees(rpy[1]):+.1f} {np.degrees(rpy[2]):+.1f}",
        f"  quat wxyz: {quat[0]:+.3f} {quat[1]:+.3f} {quat[2]:+.3f} {quat[3]:+.3f}",
        f"  linear vel xyz [m/s]: {qvel[0]:+.3f} {qvel[1]:+.3f} {qvel[2]:+.3f}",
        f"  angular vel xyz [rad/s]: {qvel[3]:+.3f} {qvel[4]:+.3f} {qvel[5]:+.3f}",
        f"  planar speed: {float(latest.get('planar_speed', np.linalg.norm(qvel[:2]))):.3f} m/s"
        "  (unsigned; use along heading and closing speed for direction)",
        *_progress_lines(latest, qpos, qvel),
        f"  gait cmd: mode={latest.get('cmd_mode')} speed={latest.get('cmd_speed')} "
        f"heading={latest.get('cmd_heading')} walking={latest.get('walking')}",
        _gait_band_line(str(latest.get("stack", "legacy_dcm"))),
        f"  yaw_ref={latest.get('yaw_ref')} heading_goal={latest.get('heading_goal')} "
        f"heading_err_raw={latest.get('heading_error_raw')} "
        f"heading_err_gated={latest.get('heading_error')} "
        f"cross_track={latest.get('cross_track')}",
        "",
    ]
    return "\n".join(lines)


_EFFORT_NOTE = (
    "Measured force minus the belief model's inverse dynamics, resolved into actuators through the foot contacts. "
    "Measured force is actuator torque (current) plus contact force (foot sensors). "
    "Zero means the believed dynamics explain the motion. "
    "A persistent positive value means the plant outweighs the belief along that joint (mass or COM too low). "
    "A persistent negative value means the belief overshoots the plant. "
    "A correct mass or COM edit drives these values to zero; if the sign flips, the last edit went too far. "
    "|bias| is the sum of the absolute per-actuator values. "
    "consecutive out of bounds is how many control steps in a row |bias| has stayed above the standing or walking threshold, "
    "written as the live count over the configured streak length. "
    "A difference of more than 2 Nm between symmetric left/right actuators indicates an asymmetric payload or COM offset. "
    "A roll residual is a com y search. A pitch residual is a com x search. Height is a com z search."
)


def format_effort_section(observer) -> str:
    """Per-actuator effort deviation, largest magnitude first."""
    lines = ["EFFORT DEVIATION [Nm]", f"  {_EFFORT_NOTE}"]
    if observer is None or observer.joint_bias is None:
        lines.append("  (no samples)")
        return "\n".join(lines) + "\n"
    bias = np.asarray(observer.joint_bias, dtype=float)
    names = list(observer.names or tuple(str(i) for i in range(bias.size)))
    mode = "walking" if observer.is_walking else "standing"
    threshold = (
        observer.walk_threshold_nm if observer.is_walking else observer.stand_threshold_nm
    )
    lines.append(
        f"  |bias|={float(np.sum(np.abs(bias))):.2f}  mode={mode}"
        f"  threshold={threshold:.1f}"
    )
    lines.append(
        f"  consecutive out of bounds: {observer.consecutive_out_of_bounds}"
        f" / {observer.consecutive_steps}"
    )
    ranked = sorted(
        (
            (abs(float(bias[i])), float(bias[i]), names[i])
            for i in range(min(len(names), bias.size))
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    for _magnitude, value, name in ranked:
        lines.append(f"  {name}: {value:+.2f}")
    if observer.motor_bias is not None:
        lines.append(
            f"  |motor|={observer.motor_nm:.2f}  threshold={observer.motor_threshold_nm:.1f}"
            "  commanded torque minus applied torque (current). Zero means the gear matches."
        )
        motor = np.asarray(observer.motor_bias, dtype=float)
        shown = 0
        order = np.argsort(-np.abs(motor))
        for index in order:
            if abs(float(motor[index])) < 0.2 or shown >= 4:
                break
            name = names[index] if index < len(names) else str(index)
            lines.append(f"  motor {name}: {float(motor[index]):+.2f}")
            shown += 1
    lines.append(
        f"  stance slip={observer.slip_mps:.3f} m/s"
        f"  threshold={observer.slip_threshold_mps:.2f}"
        "  fastest planted sole, ground plane. Persistent slip with a quiet motor is friction."
    )
    return "\n".join(lines) + "\n"


def format_cost_section(observer) -> str:
    """Per-component MPC state cost, averaged over the observer window."""
    from loka.agent.anomaly import _COST_NAMES

    lines = ["STATE COST"]
    if observer is None or observer.avg_cost is None:
        lines.append("  (no samples)")
        return "\n".join(lines) + "\n"
    cost = np.atleast_1d(np.asarray(observer.avg_cost, dtype=float))
    for index, name in enumerate(_COST_NAMES):
        value = float(cost[index]) if index < cost.size else 0.0
        lines.append(f"  {name}: {value:.4g}")
    return "\n".join(lines) + "\n"


def format_prop_contact_section(frames: Sequence[dict]) -> str:
    """Robot-to-prop contact over the window. Empty when nothing was touched.

    Objects are numbered, not named: what the robot bumped into is for the
    orchestrator to find in the image, as a robot only feels the force.
    """
    peak: dict[str, dict] = {}
    touching_steps: dict[str, int] = {}
    for frame in frames:
        for entry in frame.get("prop_contacts") or ():
            name = entry["prop"]
            touching_steps[name] = touching_steps.get(name, 0) + 1
            best = peak.get(name)
            if best is None or entry["force_n"] > best["force_n"]:
                peak[name] = entry
    if not peak:
        return ""
    latest = {e["prop"] for e in (frames[-1].get("prop_contacts") or ())}
    lines = [
        "OBJECT CONTACT",
        "  The robot touched an object in this window. Contact force enters the effort "
        "residual as an external load. It is not a model error; do not queue a belief "
        "search for it.",
    ]
    ranked = sorted(peak.items(), key=lambda kv: -kv[1]["force_n"])
    for index, (name, entry) in enumerate(ranked, start=1):
        share = touching_steps[name] / max(len(frames), 1)
        now = "touching now" if name in latest else "released"
        lines.append(
            f"  object {index}: peak normal {entry['force_n']:.1f} N, "
            f"{share:.0%} of the window, {now}; robot links {', '.join(entry['links'])}"
        )
    return "\n".join(lines) + "\n"


def format_compact_state_section(frames: Sequence[dict]) -> str:
    """Pose, motion, goal progress and gait command in four lines."""
    latest = frames[-1]
    qpos = np.asarray(latest.get("qpos", np.zeros(7)), dtype=float)
    qvel = np.asarray(latest.get("qvel", np.zeros(6)), dtype=float)
    rpy = np.asarray(latest.get("rpy", np.zeros(3)), dtype=float)
    lines = [
        "STATE",
        f"  t={float(latest.get('time', 0.0)):.2f}s pelvis x={qpos[0]:+.2f} y={qpos[1]:+.2f} "
        f"z={qpos[2]:.2f} m, yaw={np.degrees(rpy[2]):+.0f} deg, roll={np.degrees(rpy[0]):+.1f} "
        f"pitch={np.degrees(rpy[1]):+.1f} deg",
        f"  speed {float(latest.get('planar_speed', np.linalg.norm(qvel[:2]))):.2f} m/s, "
        f"walking={bool(latest.get('walking'))}",
    ]
    if bool(latest.get("goal_active", False)):
        goal = np.array([float(latest.get("goal_x", 0.0)), float(latest.get("goal_y", 0.0))])
        delta = goal - qpos[:2]
        dist = float(np.linalg.norm(delta))
        closing = float(qvel[:2] @ delta / dist) if dist > 1e-4 else 0.0
        bearing = float(np.degrees(np.arctan2(delta[1], delta[0])))
        lines.append(
            f"  goal ({goal[0]:+.2f}, {goal[1]:+.2f}) {dist:.2f} m away at bearing {bearing:+.0f} deg, "
            f"closing speed {closing:+.2f} m/s"
        )
    return "\n".join(lines) + "\n"


def format_compact_effort_section(observer) -> str:
    """|bias| against its threshold, the three largest actuators, motor and slip."""
    if observer is None or observer.joint_bias is None:
        return "EFFORT\n  (no samples)\n"
    bias = np.asarray(observer.joint_bias, dtype=float)
    names = list(observer.names or tuple(str(i) for i in range(bias.size)))
    threshold = observer.walk_threshold_nm if observer.is_walking else observer.stand_threshold_nm
    order = np.argsort(-np.abs(bias))[:3]
    top = ", ".join(f"{names[i]} {bias[i]:+.1f}" for i in order if i < len(names))
    motor = f"{observer.motor_nm:.2f}" if observer.motor_bias is not None else "n/a"
    return (
        "EFFORT\n"
        f"  |bias|={float(np.sum(np.abs(bias))):.1f} Nm (threshold {threshold:.1f}, "
        f"out of bounds {observer.consecutive_out_of_bounds}/{observer.consecutive_steps}); "
        f"largest: {top}\n"
        f"  |motor|={motor} (threshold {observer.motor_threshold_nm:.1f}), "
        f"slip={observer.slip_mps:.2f} m/s (threshold {observer.slip_threshold_mps:.2f})\n"
    )


def synthesize_stand_telemetry(
    frames: Sequence[dict],
    *,
    control_dt: float,
    observer=None,
    compact: bool = False,
    hide_residual: bool = False,
) -> str:
    """Current state, per-actuator effort deviation, and per-component state cost.

    ``compact`` keeps the state, the effort summary and any prop contact, and
    drops the per-actuator list and the state cost. With a tracking-cost
    detector (``observer.kind == "cost"``) the report adds the TRACKING COST
    trigger; ``hide_residual`` then leaves out the effort deviation, motor
    deficit and stance slip.
    """
    window = _window_frames(frames, control_dt)
    if not window:
        return "No telemetry available.\n"
    cost_trigger = getattr(observer, "kind", "residual") == "cost"
    effort = None if hide_residual else (
        format_compact_effort_section(observer) if compact else format_effort_section(observer)
    )
    trigger = None
    if cost_trigger:
        trigger = (
            format_compact_tracking_cost_section(observer)
            if compact
            else format_tracking_cost_section(observer)
        )
    if compact:
        sections = [
            format_compact_state_section(window),
            trigger,
            effort,
            format_prop_contact_section(window),
            # Without the residual, the per-channel cost is what picks a search.
            format_cost_section(observer) if hide_residual else None,
        ]
    else:
        sections = [
            format_full_state_section(window),
            trigger,
            effort,
            format_prop_contact_section(window),
            format_cost_section(observer),
        ]
    return "\n".join(section for section in sections if section)


def format_tracking_cost_section(observer) -> str:
    """The cost detector's windowed signal, threshold and streak."""
    lines = ["TRACKING COST"]
    value = getattr(observer, "trigger_value", None)
    if value is None:
        lines.append("  (no samples)")
        return "\n".join(lines) + "\n"
    mode = "walking" if observer.is_walking else "standing"
    lines.append(
        "  The anomaly signal: the MPC's weighted state residual summed over "
        + ", ".join(observer.cost_channels)
        + " and averaged over the detector window. It rises when the controller cannot "
        "track its centroidal state, whatever the cause."
    )
    lines.append(f"  cost={value:.4g}  mode={mode}  threshold={observer.trigger_threshold:.4g}")
    lines.append(
        f"  consecutive out of bounds: {observer.consecutive_out_of_bounds}"
        f" / {observer.consecutive_steps}"
    )
    return "\n".join(lines) + "\n"


def format_compact_tracking_cost_section(observer) -> str:
    value = getattr(observer, "trigger_value", None)
    if value is None:
        return "TRACKING COST\n  (no samples)\n"
    return (
        "TRACKING COST\n"
        f"  cost={value:.3g} (threshold {observer.trigger_threshold:.3g}, "
        f"out of bounds {observer.consecutive_out_of_bounds}/{observer.consecutive_steps})\n"
    )
