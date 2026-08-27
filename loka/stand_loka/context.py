"""Prompt context for the G1 standing LOKA loop."""

from __future__ import annotations

from pathlib import Path

import mujoco

from loka.control.stand import (
    HEIGHT_BAND,
    MAX_LEAN_XY,
    MAX_YAW,
    TASK_PARAMETER_NAMES,
    StandController,
)
from loka.robot_context import build_robot_model_context
from loka.stand_loka.policy import (
    LLM_STAND_CONTROLLER_ALLOWLIST,
    stand_llm_catalogue,
)


def default_stand_objective() -> str:
    return (
        "Hold a robust standing balance on the Unitree G1, and accept operator "
        "requests to crouch, stand tall, lean, yaw, or walk. On walk requests, "
        "set gait.mode/speed/heading (and optionally cadence / swing height). "
        "On faults, diagnose and mutate typed controller costs / task setpoints "
        "/ model belief so the MPC+WBC plant recovers without generating code."
    )


def build_stand_capabilities(controller: StandController) -> dict:
    limits = controller.lean_limits()
    return {
        "cost_weights": sorted(LLM_STAND_CONTROLLER_ALLOWLIST),
        "task_parameters": sorted(TASK_PARAMETER_NAMES),
        "task_limits": limits,
        "max_lean_xy": MAX_LEAN_XY,
        "height_band": HEIGHT_BAND,
        "max_yaw": MAX_YAW,
        "planner": [],  # no MJPC planner numerics on this plant
    }


def format_stand_capabilities_block(capabilities: dict) -> str:
    lines = [
        "## AVAILABLE CONTROLLER PARAMETERS",
        "Controller_Targets MUST use dotted paths from the allowlist below.",
        "Any other mpc.*/wbc.*/stand.* path is locked and will be ignored.",
        "Never use bare names like weight_force — they are ambiguous across layers.",
        "",
        "### Adaptive Controller_Targets (allowlist)",
        stand_llm_catalogue(),
        "",
        "### Task_Targets",
        "height [m], yaw [rad], lean_x [m forward], lean_y [m left]",
        f"  height in [{capabilities['task_limits']['height_min']:.3f}, "
        f"{capabilities['task_limits']['height_max']:.3f}]",
        f"  lean clamped to ~{LEAN_FRACTION_TEXT} of support margin "
        f"(hard |lean| <= {MAX_LEAN_XY} m)",
        f"  yaw in [{-MAX_YAW:.2f}, {MAX_YAW:.2f}] rad",
        "",
        "### Gait Task_Targets (stand ↔ walk)",
        "gait.mode: stand|walk|tread|limp  (or 0..3)",
        "gait.speed [m/s], gait.heading [rad world], gait.step_period [s],",
        "gait.duty_factor, gait.step_length_max, gait.swing_height,",
        "gait.stance_width, gait.capture_gain, gait.walk_accel",
        "Setting gait.mode=walk with omitted speed defaults to ~0.25 m/s.",
        "Foothold XY and contact schedules are classical (not LLM-settable).",
        "While walking: contact slip, force mismatch, and CoM residuals during",
        "swing are expected. Do NOT lower friction_mu or kp_base_position for",
        "those alone — prefer gait.speed / duty / step_period, or lean.",
        "",
        "### Model_Mutations (belief on the controller's private MjModel)",
        "object_type: actuator|geom|body; attributes: gear|friction|mass",
        "Zero actuator gear for virtual amputation; do not raise gear to 'force' a seized joint.",
    ]
    return "\n".join(lines)


LEAN_FRACTION_TEXT = "55%"


def format_stand_configuration(controller: StandController) -> str:
    tasks = controller.task_snapshot()
    weights = controller.tunables()
    lines = ["## CURRENT STAND CONFIGURATION", "Task_Targets:"]
    for key, value in tasks.items():
        if key == "gait.mode":
            from loka.control.gait import gait_mode_name

            lines.append(f"  {key}: {value:.4g} ({gait_mode_name(value)})")
        else:
            lines.append(f"  {key}: {value:.4g}")
    lines.append("Controller_Targets (allowlisted):")
    for path in sorted(LLM_STAND_CONTROLLER_ALLOWLIST):
        if path in weights:
            lines.append(f"  {path}: {weights[path]:.4g}")
    return "\n".join(lines)


def build_stand_robot_context(controller: StandController) -> str:
    model = controller.robot.model
    xml_path = Path(controller.config.model_path)
    base = build_robot_model_context(model, str(xml_path))
    limits = controller.lean_limits()
    extra = [
        "",
        "## G1 STAND NOTES",
        f"Nominal CoM height above soles: {controller.nominal_height:.3f} m",
        f"Live lean limits [m]: "
        f"x in [{limits['lean_x_min']:.3f}, {limits['lean_x_max']:.3f}], "
        f"y in [{limits['lean_y_min']:.3f}, {limits['lean_y_max']:.3f}]",
        "qpos[0:3]=pelvis xyz, qpos[3:7]=quat wxyz, qpos[7:]=29 joints.",
        "qvel[0:3]=linear, qvel[3:6]=angular, qvel[6:]=joint rates.",
        "Default Error_Tracking uses pelvis_height (qpos[2]), drift (qpos[0]/[1]), "
        "tip_rate (qvel[4]), roll_rate (qvel[3]).",
        "On gait.mode=walk, forward world-x drift is dropped from Error_Tracking "
        "(walking advances); criteria focus on height, lateral drift, tip/roll rates.",
    ]
    # Actuator name list helps amputation.
    names = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        names.append(f"{i}:{name or '?'}")
    extra.append("Actuators: " + ", ".join(names[:16]) + ", ...")
    return base + "\n" + "\n".join(extra)
