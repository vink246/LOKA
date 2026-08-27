"""G1 standing telemetry compression for the LOKA orchestrator."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from loka.compressor import (
    TELEMETRY_WINDOW_S,
    format_tracked_state_section,
)
from loka.error_spec import ErrorSpec
from loka.stand_loka.anomaly import assess_stand_anomaly

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


def _mean_abs(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.abs(values)))


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
    force_err = float(np.linalg.norm(des - got)) if des.shape == got.shape else float("nan")

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
        f"  force tracking ||f*-f||: {force_err:.1f} N"
        f"  mpc_cost: {float(latest.get('mpc_cost', 0.0)):.3g}"
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
        # Reuse Walker formatter: it only needs qpos/qvel in frames.
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
    if sections.section_3_directive:
        blocks.append(
            "3. DIRECTIVE\n"
            + (
                directive
                or (
                    "Update the YAML scratchpad to rewrite Controller_Targets "
                    "(dotted mpc.*/wbc.*/stand.* paths), Task_Targets "
                    "(height, yaw, lean_x, lean_y), Error_Tracking if the mission "
                    "changed, and Model_Mutations for belief (gear/friction/mass).\n"
                )
            )
        )
    return "\n".join(blocks)
