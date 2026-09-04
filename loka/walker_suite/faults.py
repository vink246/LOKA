"""Hidden plant faults for the Walker MJPC suite (planner does not see them)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np


FORBIDDEN_RAW_KEYS = frozenset({"delta_kg", "magnitude"})
FORCE_DIRECTIONS = frozenset({"forward", "backward"})
PERTURBATION_KINDS = frozenset(
    {"actuator_dead", "friction", "mass", "force", "obstacle"}
)


@dataclass
class PlantSnapshot:
    actuator_gear: np.ndarray
    geom_friction: np.ndarray
    body_mass: np.ndarray
    geom_pos: np.ndarray
    geom_size: np.ndarray
    geom_contype: np.ndarray
    geom_conaffinity: np.ndarray


@dataclass
class ResolvedPerturbation:
    kind: str
    params: dict[str, Any]
    body_mass_kg: float = 0.0
    gravity: float = 9.81
    delta_kg: float | None = None
    force_n: float | None = None
    force_vec: np.ndarray | None = None
    duration_s: float | None = None
    actuator_id: int = -1
    actuator_name: str | None = None
    body_id: int = -1
    body_name: str | None = None
    floor_id: int = -1
    obstacle_id: int = -1
    obstacle_pos: np.ndarray | None = None
    obstacle_size: np.ndarray | None = None


def walker_total_mass(model: mujoco.MjModel) -> float:
    """Total mass excluding the world body."""
    return float(np.sum(model.body_mass[1:]))


def walker_gravity(model: mujoco.MjModel) -> float:
    return abs(float(model.opt.gravity[2]))


def capture_plant_snapshot(model: mujoco.MjModel) -> PlantSnapshot:
    return PlantSnapshot(
        actuator_gear=model.actuator_gear[:, 0].copy(),
        geom_friction=model.geom_friction.copy(),
        body_mass=model.body_mass.copy(),
        geom_pos=model.geom_pos.copy(),
        geom_size=model.geom_size.copy(),
        geom_contype=model.geom_contype.copy(),
        geom_conaffinity=model.geom_conaffinity.copy(),
    )


def restore_plant_snapshot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    snap: PlantSnapshot,
    *,
    set_const: bool = False,
) -> None:
    model.actuator_gear[:, 0] = snap.actuator_gear
    model.geom_friction[:] = snap.geom_friction
    model.body_mass[:] = snap.body_mass
    model.geom_pos[:] = snap.geom_pos
    model.geom_size[:] = snap.geom_size
    model.geom_contype[:] = snap.geom_contype
    model.geom_conaffinity[:] = snap.geom_conaffinity
    data.xfrc_applied[:] = 0.0
    if set_const:
        mujoco.mj_setConst(model, data)


def _reject_raw_si(raw: dict[str, Any]) -> None:
    bad = FORBIDDEN_RAW_KEYS.intersection(raw)
    if bad:
        keys = ", ".join(sorted(bad))
        raise ValueError(
            f"Perturbation keys {keys} are not allowed; use mass_frac / force_frac "
            "as decimals of walker mass or bodyweight."
        )


def resolve_perturbation(
    raw: dict[str, Any], model: mujoco.MjModel
) -> ResolvedPerturbation:
    if not isinstance(raw, dict) or "kind" not in raw:
        raise ValueError("Perturbation must be a mapping with a 'kind' field")
    _reject_raw_si(raw)
    kind = str(raw["kind"]).strip()
    if kind not in PERTURBATION_KINDS:
        raise ValueError(f"Unknown perturbation kind '{kind}'")

    mass = walker_total_mass(model)
    gravity = walker_gravity(model)
    resolved = ResolvedPerturbation(
        kind=kind,
        params=dict(raw),
        body_mass_kg=mass,
        gravity=gravity,
    )

    if kind == "actuator_dead":
        name = str(raw.get("actuator", "right_hip"))
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if act_id < 0:
            raise ValueError(f"Unknown actuator '{name}'")
        resolved.actuator_id = int(act_id)
        resolved.actuator_name = name

    elif kind == "friction":
        if "mu" not in raw:
            raise ValueError("friction perturbation requires 'mu'")
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_id < 0:
            raise ValueError("Model has no geom named 'floor'")
        resolved.floor_id = int(floor_id)
        resolved.params["mu"] = float(raw["mu"])

    elif kind == "mass":
        if "mass_frac" not in raw:
            raise ValueError("mass perturbation requires 'mass_frac' (decimal of total mass)")
        frac = float(raw["mass_frac"])
        body_name = str(raw.get("body", "torso"))
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Unknown body '{body_name}'")
        resolved.body_id = int(body_id)
        resolved.body_name = body_name
        resolved.delta_kg = frac * mass
        resolved.params["mass_frac"] = frac

    elif kind == "force":
        if "force_frac" not in raw:
            raise ValueError(
                "force perturbation requires 'force_frac' (decimal of bodyweight m*g)"
            )
        direction = str(raw.get("direction", "forward")).strip().lower()
        if direction not in FORCE_DIRECTIONS:
            raise ValueError("force direction must be 'forward' or 'backward'")
        frac = float(raw["force_frac"])
        force_n = frac * mass * gravity
        sign = 1.0 if direction == "forward" else -1.0
        torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        if torso_id < 0:
            raise ValueError("Model has no body named 'torso'")
        resolved.body_id = int(torso_id)
        resolved.body_name = "torso"
        resolved.force_n = force_n
        resolved.force_vec = np.array([sign * force_n, 0.0, 0.0], dtype=float)
        resolved.duration_s = float(raw.get("duration_s", 0.2))
        resolved.params["force_frac"] = frac
        resolved.params["direction"] = direction

    elif kind == "obstacle":
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle")
        if geom_id < 0:
            raise ValueError("Model has no geom named 'obstacle'")
        size = raw.get("size", [0.15, 0.40, 0.10])
        size_arr = np.asarray(size, dtype=float).reshape(-1)
        if size_arr.size != 3:
            raise ValueError("obstacle size must be [sx, sy, sz]")
        x = float(raw.get("x", 4.0))
        resolved.obstacle_id = int(geom_id)
        resolved.obstacle_size = size_arr
        resolved.obstacle_pos = np.array([x, 0.0, float(size_arr[2])], dtype=float)

    return resolved


def perturbation_metadata(resolved: ResolvedPerturbation) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "kind": resolved.kind,
        "params": dict(resolved.params),
        "body_mass_kg": resolved.body_mass_kg,
        "gravity": resolved.gravity,
    }
    if resolved.delta_kg is not None:
        meta["delta_kg"] = resolved.delta_kg
        meta["mass_frac"] = float(resolved.params["mass_frac"])
    if resolved.force_n is not None:
        meta["force_n"] = resolved.force_n
        meta["force_frac"] = float(resolved.params["force_frac"])
        meta["direction"] = resolved.params.get("direction")
        meta["duration_s"] = resolved.duration_s
    if resolved.actuator_name:
        meta["actuator"] = resolved.actuator_name
    if resolved.obstacle_pos is not None:
        meta["obstacle_pos"] = resolved.obstacle_pos.tolist()
        meta["obstacle_size"] = resolved.obstacle_size.tolist()
    return meta


def apply_resolved_fault(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    fault: ResolvedPerturbation,
    sim_time: float,
    activated_at: float,
    snapshot: PlantSnapshot,
) -> None:
    """Overlay a hidden plant fault onto the physics model (after planning)."""
    kind = fault.kind
    if kind == "actuator_dead" and fault.actuator_id >= 0:
        model.actuator_gear[fault.actuator_id, 0] = 0.0
        return

    if kind == "friction" and fault.floor_id >= 0:
        model.geom_friction[fault.floor_id, 0] = float(fault.params["mu"])
        return

    if kind == "mass" and fault.body_id >= 0 and fault.delta_kg is not None:
        model.body_mass[fault.body_id] = float(
            snapshot.body_mass[fault.body_id] + fault.delta_kg
        )
        mujoco.mj_setConst(model, data)
        return

    if kind == "force" and fault.force_vec is not None and fault.body_id >= 0:
        elapsed = sim_time - activated_at
        if fault.duration_s is None or elapsed < fault.duration_s:
            data.xfrc_applied[fault.body_id, :3] = fault.force_vec
        return

    if kind == "obstacle" and fault.obstacle_id >= 0:
        gid = fault.obstacle_id
        model.geom_pos[gid] = fault.obstacle_pos
        model.geom_size[gid] = fault.obstacle_size
        model.geom_contype[gid] = 0
        model.geom_conaffinity[gid] = 1


class PlantFaults:
    """Hidden *plant* disturbances. These never touch the planner's belief model.

    MJPC is initialized from a separate belief MjModel (nominal + LOKA
    Model_Mutations only). This object mutates only the physics MjModel.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.snapshot = capture_plant_snapshot(model)
        self.active: ResolvedPerturbation | None = None
        self.activated_at: float | None = None

    def clear(self) -> None:
        restore_plant_snapshot(self.model, self.data, self.snapshot)
        self.active = None
        self.activated_at = None

    @property
    def is_active(self) -> bool:
        return self.active is not None

    def activate(self, fault: ResolvedPerturbation, sim_time: float) -> None:
        self.active = fault
        self.activated_at = float(sim_time)
        apply_resolved_fault(
            self.model,
            self.data,
            fault,
            sim_time,
            self.activated_at,
            self.snapshot,
        )

    def apply_physics(self, sim_time: float) -> None:
        """Re-apply the plant overlay for this physics step.

        Persistent faults (gear, friction, mass, obstacle) stay on the plant
        model. Force is time-limited via xfrc_applied on data.
        """
        if self.active is None or self.activated_at is None:
            self.data.xfrc_applied[:] = 0.0
            return
        if self.active.kind == "force":
            self.data.xfrc_applied[:] = 0.0
        apply_resolved_fault(
            self.model,
            self.data,
            self.active,
            sim_time,
            self.activated_at,
            self.snapshot,
        )

    def obstacle_overlay(self) -> dict[str, Any] | None:
        if self.active is None or self.active.kind != "obstacle":
            return None
        return {
            "visible_from": self.activated_at,
            "pos": self.active.obstacle_pos.tolist(),
            "size": self.active.obstacle_size.tolist(),
        }
