"""Robot MJCF reference text and live MPC capability formatting for LOKA."""

import os
import re

import mujoco

from loka.error_spec import ErrorSpec, format_error_spec
from loka.mjcf_utils import SUPPORTED_PLANNER_NUMERICS, get_model_numeric

_JOINT_TYPES = {
    int(mujoco.mjtJoint.mjJNT_FREE): "free",
    int(mujoco.mjtJoint.mjJNT_BALL): "ball",
    int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
    int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
}


def _joint_type_name(jtype):
    return _JOINT_TYPES.get(int(jtype), f"type_{int(jtype)}")


def _actuator_target_joint(model, actuator_id):
    if model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
        return None
    joint_id = model.actuator_trnid[actuator_id, 0]
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)


def load_kinematics_mjcf(xml_path):
    """Load kinematics-relevant MJCF includes referenced by the task XML."""
    task_dir = os.path.dirname(os.path.abspath(xml_path))

    try:
        with open(xml_path, encoding="utf-8") as handle:
            task_xml = handle.read()
    except OSError:
        return None

    snippets = []
    for match in re.finditer(r'<include\s+file="([^"]+)"\s*/>', task_xml):
        include_name = match.group(1)
        if "common" in include_name.lower():
            continue

        include_path = os.path.normpath(os.path.join(task_dir, include_name))
        try:
            with open(include_path, encoding="utf-8") as handle:
                snippets.append(f"<!-- included from {include_name} -->\n{handle.read()}")
        except OSError:
            continue

    if snippets:
        return "\n\n".join(snippets)

    if len(task_xml) <= 8000:
        return task_xml
    return None


def capture_nominal_params(model):
    """Snapshot nominal MJCF parameters before any runtime or LOKA changes."""
    actuators = {}
    for actuator_id in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        actuators[name] = {"gear": float(model.actuator_gear[actuator_id, 0])}

    geoms = {}
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if not name:
            continue
        geoms[name] = {"friction": model.geom_friction[geom_id].tolist()}

    bodies = {}
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        bodies[name] = {"mass": float(model.body_mass[body_id])}

    return {"actuators": actuators, "geoms": geoms, "bodies": bodies}


def discover_capabilities(agent, model) -> dict:
    """Live cost weights, task parameters, and planner numerics from the loaded model."""
    try:
        cost_weights = {name: float(val) for name, val in agent.get_cost_weights().items()}
    except Exception:
        cost_weights = {}

    try:
        task_parameters = {
            name: float(val) for name, val in agent.get_task_parameters().items()
        }
    except Exception:
        task_parameters = {}

    planner = {}
    for name in sorted(SUPPORTED_PLANNER_NUMERICS):
        value = get_model_numeric(model, name)
        if value is not None:
            planner[name] = value

    return {
        "cost_weights": cost_weights,
        "task_parameters": task_parameters,
        "planner": planner,
    }


def format_capabilities_block(capabilities: dict) -> str:
    lines = ["AVAILABLE CONTROLLER / PLANNER / TASK PARAMETERS"]

    weights = capabilities.get("cost_weights", {})
    lines.append("Available Cost Weights (use exact names in Controller_Targets):")
    if weights:
        for name, value in sorted(weights.items()):
            lines.append(f'  - "{name}" (current: {value:.4g})')
    else:
        lines.append("  - [none discovered]")

    planner = capabilities.get("planner", {})
    lines.append("Available Planner Metaparameters:")
    if planner:
        for name, value in sorted(planner.items()):
            lines.append(f'  - "{name}" (current: {value:.4g})')
    else:
        lines.append("  - [none discovered]")

    tasks = capabilities.get("task_parameters", {})
    lines.append("Available Task Parameters (use exact names in Task_Targets):")
    if tasks:
        for name, value in sorted(tasks.items()):
            lines.append(f'  - "{name}" (current: {value:.4g})')
    else:
        lines.append("  - [none discovered]")

    return "\n".join(lines)


def format_primary_objective_block(objective: str) -> str:
    return (
        "PRIMARY OBJECTIVE\n"
        f"{objective.strip()}\n"
        "All MPC targets and Error_Tracking criteria should serve this objective."
    )


def format_current_error_tracking(error_spec: ErrorSpec) -> str:
    return (
        "CURRENT ERROR TRACKING (live success / failure criteria):\n"
        f"{format_error_spec(error_spec)}"
    )


def _lookup_nominal(nominal_params, mutation):
    obj_type = mutation["type"]
    name = mutation["name"]
    attr = mutation["attr"]
    bucket = nominal_params.get(f"{obj_type}s", {})
    obj = bucket.get(name, {})
    return obj.get(attr, "unknown")


def format_current_model_belief(loka_state):
    """Summarize LOKA's active internal-model mutations (no diagnosis history)."""
    mutations = loka_state.get("mutations", [])
    nominal_params = loka_state.get("nominal_params", {})

    if not mutations:
        return "No active mutations. Internal model matches nominal MJCF."

    latest = {}
    for mutation in mutations:
        key = (mutation["type"], mutation["name"], mutation["attr"])
        latest[key] = mutation

    lines = ["Active model mutations (nominal -> believed):"]
    for mutation in latest.values():
        nominal_val = _lookup_nominal(nominal_params, mutation)
        applied_at = mutation.get("applied_at")
        time_note = f", applied at t={applied_at:.2f}s" if applied_at is not None else ""
        lines.append(
            f"  - {mutation['type']} '{mutation['name']}'.{mutation['attr']}: "
            f"{mutation['val']} (nominal: {nominal_val}{time_note})"
        )

    return "\n".join(lines)


def format_current_mpc_configuration(agent, model):
    """Snapshot live MPC cost weights, planner settings, and task goals."""
    lines = ["Live values currently loaded in the MPC agent:"]

    try:
        weights = agent.get_cost_weights()
        lines.append("Controller cost weights:")
        for name in sorted(weights.keys()):
            lines.append(f"  - {name}: {float(weights[name]):.4g}")
    except Exception as exc:
        lines.append(f"Controller cost weights: [unavailable: {exc}]")

    lines.append("Planner metaparameters:")
    for name in sorted(SUPPORTED_PLANNER_NUMERICS):
        value = get_model_numeric(model, name)
        if value is None:
            lines.append(f"  - {name}: [not found in model]")
        else:
            lines.append(f"  - {name}: {value:.4g}")

    try:
        task_params = agent.get_task_parameters()
        lines.append("Task parameters:")
        for name in sorted(task_params.keys()):
            lines.append(f"  - {name}: {float(task_params[name]):.4g}")
    except Exception as exc:
        lines.append(f"Task parameters: [unavailable: {exc}]")

    return "\n".join(lines)


def build_robot_model_context(model, xml_path):
    """Build a static kinematic reference for the LLM orchestrator."""
    lines = [
        "ROBOT MODEL REFERENCE",
        "Use this section to interpret telemetry indices, joint names, and valid mutation targets.",
        f"Source task XML: {os.path.abspath(xml_path)}",
        "",
        "Body kinematic tree (parent -> child):",
    ]

    for body_id in range(1, model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        parent_id = model.body_parentid[body_id]
        parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id) or "world"
        body_pos = model.body_pos[body_id]
        lines.append(
            f"  - {body_name} (parent: {parent_name}, "
            f"mjcf_pos=[{body_pos[0]:.3g}, {body_pos[1]:.3g}, {body_pos[2]:.3g}])"
        )

    lines.extend([
        "",
        "Joint / DOF layout (maps to qpos/qvel; use these indices in Error_Tracking):",
    ])
    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        qpos_idx = model.jnt_qposadr[joint_id]
        qvel_idx = model.jnt_dofadr[joint_id]
        joint_type = _joint_type_name(model.jnt_type[joint_id])
        body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, model.jnt_bodyid[joint_id]
        )

        range_str = ""
        if model.jnt_limited[joint_id]:
            joint_range = model.jnt_range[joint_id]
            range_str = f", range=[{joint_range[0]:.1f}, {joint_range[1]:.1f}]"

        lines.append(
            f"  - {joint_name}: {joint_type} on {body_name}, "
            f"qpos[{qpos_idx}], qvel[{qvel_idx}]{range_str}"
        )

    lines.append("")
    lines.append(
        "Note: for slide joints parented under a body with non-zero mjcf_pos, "
        "world-frame position is often mjcf_pos[axis] + qpos[index]. "
        "Use Error_Tracking.offset when you need world-frame values."
    )

    lines.extend(["", "Actuators (exact lowercase names for Model_Mutations):"])
    for actuator_id in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        target_joint = _actuator_target_joint(model, actuator_id) or "unknown"
        gear = model.actuator_gear[actuator_id, 0]
        ctrl_range = model.actuator_ctrlrange[actuator_id]
        lines.append(
            f"  - {actuator_name}: motor on {target_joint}, "
            f"gear={gear:.1f}, ctrlrange=[{ctrl_range[0]:.1f}, {ctrl_range[1]:.1f}]"
        )

    lines.extend([
        "",
        "Mutable object attributes:",
        "  - actuators: gear",
        "  - geoms: friction",
        "  - bodies: mass",
    ])

    mjcf = load_kinematics_mjcf(xml_path)
    if mjcf:
        lines.extend([
            "",
            "MJCF kinematics definition (from task XML includes):",
            "```xml",
            mjcf.strip(),
            "```",
        ])

    return "\n".join(lines)
