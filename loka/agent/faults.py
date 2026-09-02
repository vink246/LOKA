"""Hidden plant faults for standing LOKA evaluation (not controller belief)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np


@dataclass
class FaultSpec:
    """A plant-side disturbance injected into the *simulator* model."""

    kind: str  # friction | mass | actuator_dead | push
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
        return _FaultBackup("mass", {"id": body_id, "mass": previous})

    if kind == "actuator_dead":
        name = str(p.get("actuator", "right_knee"))
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if act_id < 0:
            # try substring match
            for i in range(model.nu):
                candidate = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                if candidate and name.lower() in candidate.lower():
                    act_id = i
                    name = candidate
                    break
        if act_id < 0:
            return None
        previous_gear = float(model.actuator_gear[act_id, 0])
        previous_range = model.actuator_ctrlrange[act_id].copy()
        model.actuator_gear[act_id, 0] = 0.0
        model.actuator_ctrlrange[act_id] = 0.0
        return _FaultBackup(
            "actuator_dead",
            {"id": act_id, "gear": previous_gear, "ctrlrange": previous_range, "name": name},
        )

    if kind == "push":
        # Handled by the runtime via Simulation.pushes; nothing on the model.
        return _FaultBackup("push", dict(p))

    raise ValueError(f"Unknown fault kind: {kind}")


def clear_plant_faults(model: mujoco.MjModel, backups: list[_FaultBackup]) -> None:
    for backup in backups:
        if backup.kind == "friction":
            model.geom_friction[backup.payload["id"]] = backup.payload["friction"]
        elif backup.kind == "mass":
            model.body_mass[backup.payload["id"]] = backup.payload["mass"]
        elif backup.kind == "actuator_dead":
            i = backup.payload["id"]
            model.actuator_gear[i, 0] = backup.payload["gear"]
            model.actuator_ctrlrange[i] = backup.payload["ctrlrange"]


def capture_nominal_params(model: mujoco.MjModel) -> dict[str, np.ndarray]:
    return {
        "actuator_gear": model.actuator_gear[:, 0].copy(),
        "geom_friction": model.geom_friction.copy(),
        "body_mass": model.body_mass.copy(),
    }
