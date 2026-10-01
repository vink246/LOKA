"""Telemetry compression: recent plant frames into an LLM-facing report."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from loka.agent.anomaly import assess_stand_anomaly
from loka.agent.error_spec import ErrorSpec, ErrorTerm

#: Seconds of frames summarised in a report.
TELEMETRY_WINDOW_S = 2.0
#: Seconds the runtime keeps collecting after an anomaly trips, before it
#: compresses and hands the window to the model.
ANOMALY_COLLECTION_S = 1.0

CAPTURE_MARGIN_WARN = 0.02  # m of support left before critical
SLIP_SPEED = 0.15  # m/s at a loaded contact
SAT_UTIL = 0.92
SAT_VEL = 0.05  # rad/s


@dataclass(frozen=True)
class StandTelemetrySections:
    section_0_state: bool = True
    section_1_balance: bool = True
    section_2_motor_load: bool = True
    section_3_directive: bool = True


DEFAULT_SECTIONS = StandTelemetrySections()


def _window_frames(buffer: Sequence[dict], control_dt: float) -> list[dict]:
    if not buffer:
        return []
    n = max(1, int(TELEMETRY_WINDOW_S / max(control_dt, 1e-3)))
    # deques (runtime anomaly/baseline buffers) do not support slicing
    frames = list(buffer)
    return frames[-n:]


def _mean_term_value(frames: Sequence[dict], term: ErrorTerm) -> float | None:
    if not frames:
        return None
    return float(np.mean([term.read_frame(frame) for frame in frames]))


def _term_status(term: ErrorTerm, value: float) -> str:
    excess = term.excess(value)
    if excess <= 0.0:
        return "[NOMINAL]"
    if excess * term.weight > 0.5:
        return f"[ERR: {term.name.upper()} CRITICAL]"
    return f"[ERR: {term.name.upper()} OUT OF BAND]"


def format_tracked_state_section(
    anom_slice: Sequence[dict],
    nominal_buffer: Sequence[dict],
    has_baseline: bool,
    error_spec: ErrorSpec,
    *,
    nominal_label: str = "Nominal",
) -> str:
    """The Error_Tracking terms, each against its own band and baseline."""
    if not anom_slice or not error_spec.terms:
        return ""

    lines = [
        "0. TRACKED STATE (Error_Tracking terms)",
        "Values use each term's formula: offset + signal[index].",
        f"Window = mean over recent {TELEMETRY_WINDOW_S:.1f}s anomaly frames; "
        "Snapshot = value at trigger instant.",
    ]
    if has_baseline:
        lines.append(
            f"{nominal_label} = healthy telemetry under the *current* mission "
            "(re-baselined after Task_Targets / Error_Tracking changes)."
        )
    else:
        lines.append(
            f"{nominal_label} unavailable (waiting to re-baseline under the "
            "current mission)."
        )
    lines.append("")

    snapshot_frame = anom_slice[-1]
    for term in error_spec.terms:
        current = _mean_term_value(anom_slice, term)
        snap = term.read_frame(snapshot_frame)
        if current is None:
            continue

        status = _term_status(term, current)
        band = (
            f"mode={term.mode} target={term.target:.3g} "
            f"tol={term.tolerance:.3g} weight={term.weight:.3g}"
        )
        if has_baseline:
            nominal = _mean_term_value(nominal_buffer, term)
            if nominal is not None:
                lines.append(
                    f"- {term.name}: {nominal_label} {nominal:.3g} | "
                    f"Current {current:.3g} | Snapshot {snap:.3g}  {status}"
                )
                lines.append(f"  ({band})")
                continue

        lines.append(
            f"- {term.name}: Current {current:.3g} | Snapshot {snap:.3g}  {status}"
        )
        lines.append(f"  ({band})")

    return "\n".join(lines) + "\n\n"


def _semantic_tags(frame: dict) -> list[str]:
    tags: list[str] = []
    rpy = np.asarray(frame.get("rpy", (0.0, 0.0, 0.0)), dtype=float)
    com_error = np.asarray(frame.get("com_error", (0.0, 0.0, 0.0)), dtype=float)
    margin = np.asarray(frame.get("support_margin", (0.08, 0.08, 0.1, 0.1)), dtype=float)
    contact_vel = float(frame.get("contact_speed_max", 0.0))
    mask = np.asarray(frame.get("contact_mask", ()), dtype=bool)

    if float(margin[:2].min()) < CAPTURE_MARGIN_WARN:
        tags.append("[CAPTURE_MARGIN_LOW]")
    if contact_vel > SLIP_SPEED and mask.any():
        tags.append("[SLIP]")
    if abs(rpy[1]) > np.deg2rad(8.0) or com_error[0] > 0.03:
        tags.append("[LEAN_FORWARD]" if (rpy[1] > 0 or com_error[0] > 0) else "[LEAN_BACK]")
    if abs(rpy[0]) > np.deg2rad(8.0) or abs(com_error[1]) > 0.03:
        tags.append("[LEAN_LEFT]" if (rpy[0] < 0 or com_error[1] > 0) else "[LEAN_RIGHT]")
    if mask.any() and mask.sum() <= 2:
        tags.append("[SINGLE_SUPPORT_RISK]")

    util = frame.get("joint_util")
    qvel = frame.get("qvel")
    if util is not None and qvel is not None:
        util = np.asarray(util, dtype=float)
        rates = np.asarray(qvel[6:6 + len(util)], dtype=float)
        for i, (u, rate) in enumerate(zip(util, rates)):
            if u >= SAT_UTIL and abs(rate) < SAT_VEL:
                tags.append(f"[SATURATED_SEIZED:joint{i}]")
            elif u >= SAT_UTIL and abs(rate) > 1.0:
                tags.append(f"[SATURATED_FLAILING:joint{i}]")
    return tags


def _force_mismatch(des: np.ndarray, got: np.ndarray) -> float | None:
    """Net force error, summed per foot when both arrays are the eight sole sites."""
    if des.shape != got.shape or des.size == 0:
        return None
    if des.ndim == 2 and des.shape[0] == 8:
        err = 0.0
        for foot in (0, 1):
            sl = slice(foot * 4, (foot + 1) * 4)
            err += float(np.linalg.norm(des[sl].sum(axis=0) - got[sl].sum(axis=0)))
        return err
    return float(np.linalg.norm(des - got))


def format_balance_section(frames: Sequence[dict]) -> str:
    if not frames:
        return "1. BALANCE / CONTACTS\n  (no frames)\n"
    latest = frames[-1]
    com_err = np.asarray(latest.get("com_error", np.zeros(3)), dtype=float)
    rpy = np.asarray(latest.get("rpy", np.zeros(3)), dtype=float)
    margin = np.asarray(latest.get("support_margin", np.zeros(4)), dtype=float)
    mask = np.asarray(latest.get("contact_mask", ()), dtype=bool)
    des = np.asarray(latest.get("desired_forces", np.zeros((1, 3))), dtype=float)
    got = np.asarray(latest.get("contact_forces", np.zeros((1, 3))), dtype=float)
    force_err = _force_mismatch(des, got)
    force_text = "n/a" if force_err is None else f"{force_err:.1f} N"

    com_errs = [float(np.linalg.norm(f.get("com_error", np.zeros(3)))) for f in frames]
    tilts = [float(np.linalg.norm(f.get("rpy", np.zeros(3))[:2])) for f in frames]

    lines = [
        "1. BALANCE / CONTACTS",
        f"  CoM error xyz [mm]: {com_err[0]*1e3:+.1f} {com_err[1]*1e3:+.1f} {com_err[2]*1e3:+.1f}"
        f"  | window mean {np.mean(com_errs)*1e3:.1f} max {np.max(com_errs)*1e3:.1f}",
        f"  torso rpy [deg]: roll {np.degrees(rpy[0]):+.1f}  pitch {np.degrees(rpy[1]):+.1f}"
        f"  yaw {np.degrees(rpy[2]):+.1f}  | tip mean {np.degrees(np.mean(tilts)):.1f}",
        f"  support margin [mm] back/fwd/rightneg/leftpos: "
        f"{margin[0]*1e3:.0f}/{margin[1]*1e3:.0f}/{margin[2]*1e3:.0f}/{margin[3]*1e3:.0f}",
        f"  contacts planted: {int(mask.sum())}/8"
        f"  contact speed max: {float(latest.get('contact_speed_max', 0.0)):.3f} m/s",
        f"  force tracking ||f*-f||: {force_text}",
        "  mpc_cost: "
        + (
            "n/a"
            if latest.get("mpc_cost") is None
            else f"{float(latest.get('mpc_cost')):.3g}"
        ),
        f"  qp_fail wbc/mpc: "
        f"{latest.get('wbc_failures_total', latest.get('wbc_failures', 0))}/"
        f"{latest.get('mpc_failures_total', latest.get('mpc_failures', 0))}"
        f" (Δtick {latest.get('wbc_failures', 0)}/{latest.get('mpc_failures', 0)})",
        f"  command: height={latest.get('cmd_height')} lean_x={latest.get('cmd_lean_x')}"
        f" lean_y={latest.get('cmd_lean_y')} yaw={latest.get('cmd_yaw')}",
    ]
    tags = []
    for frame in frames[-5:]:
        tags.extend(_semantic_tags(frame))
        early = assess_stand_anomaly(frame)
        tags.extend(early.clues)
        if frame.get("anomaly_clues"):
            tags.extend(frame["anomaly_clues"])
    # unique, preserve order
    seen: set[str] = set()
    uniq = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            uniq.append(tag)
    if uniq:
        lines.append("  tags: " + " ".join(uniq))
    latest_early = assess_stand_anomaly(latest)
    if latest_early.score > 0:
        lines.append(
            f"  early anomaly score: {latest_early.score:.2f} "
            f"(trigger if sustained ≥ ~0.12)"
        )
    lines.append("")
    return "\n".join(lines)


def format_motor_section(frames: Sequence[dict]) -> str:
    if not frames:
        return "2. MOTOR LOAD\n  (no frames)\n"
    latest = frames[-1]
    torque = np.asarray(latest.get("torque", ()), dtype=float)
    util = np.asarray(latest.get("joint_util", np.zeros_like(torque)), dtype=float)
    if torque.size == 0:
        return "2. MOTOR LOAD\n  (no torque)\n"
    peak_i = int(np.argmax(np.abs(torque)))
    lines = [
        "2. MOTOR LOAD",
        f"  |tau| mean {float(np.mean(np.abs(torque))):.1f} Nm  "
        f"peak {float(np.max(np.abs(torque))):.1f} Nm at joint {peak_i}",
        f"  util mean {float(np.mean(util)):.2f}  peak {float(np.max(util)):.2f}",
    ]
    hot = np.flatnonzero(util >= 0.8)
    if hot.size:
        lines.append(
            "  hot joints (util>=0.8): "
            + ", ".join(f"j{i}:{util[i]:.2f}" for i in hot[:8])
        )
    lines.append("")
    return "\n".join(lines)


def _gait_band_line(stack: str = "legacy_dcm") -> str:
    """Safe gait band from the coarse sweep. apply_updates clips to this."""
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
    return (
        "  gait safe band (out of range is clipped; defaults are the 0.25 m/s walk): "
        + "; ".join(bits)
    )


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
        return "4. FULL STATE\n  (no frames)\n"
    latest = frames[-1]
    qpos = np.asarray(latest.get("qpos", np.zeros(7)), dtype=float)
    qvel = np.asarray(latest.get("qvel", np.zeros(6)), dtype=float)
    rpy = np.asarray(latest.get("rpy", np.zeros(3)), dtype=float)
    quat = qpos[3:7] if qpos.size >= 7 else np.zeros(4)
    lines = [
        "4. FULL STATE (snapshot at this invoke)",
        f"  t={float(latest.get('time', 0.0)):.2f}s",
        f"  pelvis xyz [m]: {qpos[0]:+.3f} {qpos[1]:+.3f} {qpos[2]:+.3f}",
        f"  attitude rpy [deg]: {np.degrees(rpy[0]):+.1f} {np.degrees(rpy[1]):+.1f} "
        f"{np.degrees(rpy[2]):+.1f}",
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


def synthesize_stand_telemetry(
    nominal_baseline: Sequence[dict],
    anomaly_buffer: Sequence[dict],
    *,
    control_dt: float,
    error_spec: ErrorSpec | None = None,
    sections: StandTelemetrySections = DEFAULT_SECTIONS,
    directive: str | None = None,
) -> str:
    """Compress recent stand frames into an LLM-facing report."""
    anom = _window_frames(anomaly_buffer or nominal_baseline, control_dt)
    if not anom:
        return "No telemetry available.\n"

    blocks: list[str] = []
    if sections.section_0_state and error_spec is not None:
        blocks.append(
            format_tracked_state_section(
                anom,
                list(nominal_baseline),
                bool(nominal_baseline),
                error_spec,
                nominal_label="MissionNominal",
            )
        )
    if sections.section_1_balance:
        blocks.append(format_balance_section(anom))
    if sections.section_2_motor_load:
        blocks.append(format_motor_section(anom))
    blocks.append(format_full_state_section(anom))
    if sections.section_3_directive:
        blocks.append(
            "3. DIRECTIVE\n"
            + (
                directive
                or (
                    "Update the YAML scratchpad. Task_Targets include height, lean, yaw, "
                    "and gait.* . Error_Tracking if the success criteria should shift. "
                    "Model_Mutations for belief (gear, friction, mass, com). "
                    "Listeners to choose the next wake, with a message. "
                    "Locked mission directives cannot be edited.\n"
                )
            )
        )
    return "\n".join(blocks)
