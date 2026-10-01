"""Hidden plant faults for standing LOKA evaluation (not controller belief)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np


@dataclass
class FaultSpec:
    """A plant-side disturbance injected into the *simulator* model."""

    kind: str  # friction | mass | actuator_dead | actuator_scale | push
    time: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class _FaultBackup:
    kind: str
    payload: dict[str, Any]


def apply_plant_fault(model: mujoco.MjModel, fault: FaultSpec) -> _FaultBackup | None:
    """Mutate the physical plant. Returns a backup for clearing later."""
    kind = fault.kind
    p = fault.params

    if kind == "friction":
        mu = float(p.get("mu", 0.2))
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_id < 0:
            # first plane geom
            for i in range(model.ngeom):
                if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
                    floor_id = i
                    break
        if floor_id < 0:
            return None
        previous = model.geom_friction[floor_id].copy()
        model.geom_friction[floor_id, 0] = mu
        return _FaultBackup("friction", {"id": floor_id, "friction": previous})

    if kind == "mass":
        body = str(p.get("body", "torso_link"))
        delta = float(p.get("delta_kg", 5.0))
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if body_id < 0:
            # fall back to pelvis
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if body_id < 0:
            return None
        previous = float(model.body_mass[body_id])
        model.body_mass[body_id] = previous + delta
        _sync_model_constants(model)
        return _FaultBackup("mass", {"id": body_id, "mass": previous})

    if kind in ("actuator_dead", "actuator_scale"):
        resolved = resolve_actuator(model, str(p.get("actuator", "right_knee")))
        if resolved is None:
            return None
        act_id, name = resolved
        previous_gear = float(model.actuator_gear[act_id, 0])
        previous_range = model.actuator_ctrlrange[act_id].copy()
        if kind == "actuator_dead":
            model.actuator_gear[act_id, 0] = 0.0
            model.actuator_ctrlrange[act_id] = 0.0
            return _FaultBackup(
                "actuator_dead",
                {
                    "id": act_id,
                    "gear": previous_gear,
                    "ctrlrange": previous_range,
                    "name": name,
                },
            )
        scale = float(p.get("scale", 0.5))
        applied = previous_gear * scale
        model.actuator_gear[act_id, 0] = applied
        # Motor torque is gear * ctrl, so this is the fraction of the torque
        # the controller asked for. ctrlrange stays nominal: the command is
        # unchanged and the plant simply delivers ``scale`` of it.
        return _FaultBackup(
            "actuator_scale",
            {
                "id": act_id,
                "gear": previous_gear,
                "applied_gear": applied,
                "scale": scale,
                "name": name,
            },
        )

    if kind == "push":
        # Handled by the runtime via Simulation.pushes; nothing on the model.
        return _FaultBackup("push", dict(p))

    raise ValueError(f"Unknown fault kind: {kind}")


def resolve_actuator(model: mujoco.MjModel, name: str) -> tuple[int, str] | None:
    """Exact actuator name, then a case-insensitive substring match."""
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if act_id >= 0:
        resolved = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id) or name
        return act_id, resolved
    needle = name.lower()
    for i in range(model.nu):
        candidate = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if candidate and needle in candidate.lower():
            return i, candidate
    return None


def release_actuator(
    model: mujoco.MjModel, backups: list[_FaultBackup], act_id: int
) -> str | None:
    """Undo prior dead/scale edits on one actuator and drop those backups.

    The earliest backup holds the gear from before any of those edits, so a
    second press sets the scale against nominal instead of compounding it.
    """
    original_gear: float | None = None
    original_ctrlrange: np.ndarray | None = None
    name: str | None = None
    kept: list[_FaultBackup] = []
    for backup in backups:
        same = (
            backup.kind in ("actuator_scale", "actuator_dead")
            and int(backup.payload["id"]) == act_id
        )
        if not same:
            kept.append(backup)
            continue
        if original_gear is None:
            original_gear = float(backup.payload["gear"])
            name = str(backup.payload.get("name", ""))
        if original_ctrlrange is None and "ctrlrange" in backup.payload:
            original_ctrlrange = np.array(backup.payload["ctrlrange"], dtype=float)
    if original_gear is not None:
        model.actuator_gear[act_id, 0] = original_gear
    if original_ctrlrange is not None:
        model.actuator_ctrlrange[act_id] = original_ctrlrange
    backups[:] = kept
    return name


def _sync_model_constants(model: mujoco.MjModel) -> None:
    """Recompute subtree masses and inertias after a body_mass edit.

    ``mj_step`` reads ``body_mass`` directly for gravity, but the composite
    inertias cached by ``mj_setConst`` otherwise stay at the XML values.
    """
    mujoco.mj_setConst(model, mujoco.MjData(model))


def clear_plant_faults(model: mujoco.MjModel, backups: list[_FaultBackup]) -> None:
    # Reverse order so stacked edits of the same field land on the original
    # value (the earliest backup) rather than an intermediate one.
    for backup in reversed(backups):
        if backup.kind == "friction":
            model.geom_friction[backup.payload["id"]] = backup.payload["friction"]
        elif backup.kind == "mass":
            model.body_mass[backup.payload["id"]] = backup.payload["mass"]
        elif backup.kind == "actuator_dead":
            i = backup.payload["id"]
            model.actuator_gear[i, 0] = backup.payload["gear"]
            model.actuator_ctrlrange[i] = backup.payload["ctrlrange"]
        elif backup.kind == "actuator_scale":
            i = backup.payload["id"]
            model.actuator_gear[i, 0] = backup.payload["gear"]
    if any(backup.kind == "mass" for backup in backups):
        _sync_model_constants(model)


def capture_nominal_params(model: mujoco.MjModel) -> dict[str, np.ndarray]:
    return {
        "actuator_gear": model.actuator_gear[:, 0].copy(),
        "geom_friction": model.geom_friction.copy(),
        "body_mass": model.body_mass.copy(),
    }
