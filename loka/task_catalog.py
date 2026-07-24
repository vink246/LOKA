"""Map MJPC task ids to stock task XML paths shipped with mujoco_mpc."""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import mujoco_mpc

from loka.error_spec import ErrorSpec, ErrorTerm, default_walker_error_spec

LOKA_ROOT = Path(__file__).resolve().parent.parent
LOCAL_WALKER_XML = LOKA_ROOT / "models" / "walker" / "task.xml"

# Relative to mujoco_mpc/mjpc/tasks/
_STOCK_TASK_XML = {
    "Acrobot": "acrobot/task.xml",
    "Allegro": "allegro/task.xml",
    "Bimanual Handover": "bimanual/handover/task.xml",
    "Bimanual Insert": "bimanual/insert/task.xml",
    "Bimanual Reorient": "bimanual/reorient/task.xml",
    "Cartpole": "cartpole/task.xml",
    "FreeFingers": "fingers/task.xml",
    "Humanoid Interact": "humanoid/interact/task.xml",
    "Humanoid Stand": "humanoid/stand/task.xml",
    "Humanoid Track": "humanoid/tracking/task.xml",
    "Humanoid Walk": "humanoid/walk/task.xml",
    "OP3": "op3/task.xml",
    "Particle": "particle/task.xml",
    "ParticleFixed": "particle/task_timevarying.xml",
    "Pick": "panda/task.xml",
    "PickAndPlace": "manipulation/task_panda_bring.xml",
    "Quadruped Flat": "quadruped/task_flat.xml",
    "Quadruped Hill": "quadruped/task_hill.xml",
    "Quadrotor": "quadrotor/task.xml",
    "Rubik": "rubik/task.xml",
    "Shadow": "shadow_reorient/task.xml",
    "Swimmer": "swimmer/task.xml",
    "Walker": "walker/task.xml",
}


def mjpc_tasks_dir() -> Path:
    return Path(mujoco_mpc.__file__).resolve().parent / "mjpc" / "tasks"


def known_task_ids() -> list[str]:
    return sorted(_STOCK_TASK_XML.keys())


def resolve_xml_path(task_id: str, xml_path: str | None) -> str:
    """
    Resolve the MuJoCo task XML to load.

    If xml_path is set explicitly, use it. Otherwise use LOKA's Walker XML for
    Walker, or the stock MJPC task XML for other known task ids.
    """
    if xml_path:
        return os.path.abspath(xml_path)

    if task_id == "Walker" and LOCAL_WALKER_XML.is_file():
        return str(LOCAL_WALKER_XML)

    rel = _STOCK_TASK_XML.get(task_id)
    if rel is None:
        known = ", ".join(known_task_ids())
        raise ValueError(
            f"Unknown task_id '{task_id}' and no --xml-path given. "
            f"Known ids: {known}"
        )

    resolved = mjpc_tasks_dir() / rel
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Stock MJPC XML for task '{task_id}' not found at {resolved}. "
            "Pass --xml-path explicitly."
        )
    return str(resolved)


def initial_error_spec(task_id: str, model: mujoco.MjModel) -> ErrorSpec:
    """Walker keeps its historical defaults; other tasks get a safe bootstrap."""
    if task_id == "Walker":
        return default_walker_error_spec()
    return bootstrap_error_spec(model)


def bootstrap_error_spec(model: mujoco.MjModel) -> ErrorSpec:
    """
    Loose free-joint height watch so the loop is valid until the LLM/operator
    sets proper Error_Tracking for the objective.
    """
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        z_index = int(model.jnt_qposadr[joint_id]) + 2
        return ErrorSpec(
            trigger_threshold=0.5,
            terms=[
                ErrorTerm(
                    name="height",
                    signal="qpos",
                    index=z_index,
                    mode="abs_deviation",
                    target=1.0,
                    tolerance=2.0,
                    weight=1.0,
                ),
            ],
        )

    # No free joint — do not auto-trigger until Error_Tracking is configured.
    return ErrorSpec(trigger_threshold=1e9, terms=[])
