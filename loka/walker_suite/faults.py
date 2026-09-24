"""Hidden plant faults for the Walker MJPC suite (planner does not see them)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from loka.model_state import (
    friction_from_mutation,
    friction_mutation_targets,
    mutation_vec3,
)


FORBIDDEN_RAW_KEYS = frozenset({"delta_kg", "magnitude"})
FORCE_DIRECTIONS = frozenset({"forward", "backward"})
PERTURBATION_KINDS = frozenset(
    {"none", "actuator_dead", "friction", "mass", "force", "obstacle"}
)
BACKPACK_VISIBLE_RGBA = np.array([0.18, 0.32, 0.62, 1.0], dtype=float)
ICE_FLOOR_RGBA = np.array([0.72, 0.88, 0.98, 1.0], dtype=float)
DEFAULT_OBSTACLE_SIZE = (0.25, 0.50, 0.30)
OBSTACLE_PARKED_POS = np.array([0.0, 0.0, -5.0], dtype=float)


@dataclass
class PlantSnapshot:
    actuator_gear: np.ndarray
    geom_friction: np.ndarray
    body_mass: np.ndarray
    body_inertia: np.ndarray
    geom_pos: np.ndarray
    geom_size: np.ndarray
    geom_contype: np.ndarray
    geom_conaffinity: np.ndarray
    geom_rgba: np.ndarray
    body_pos: np.ndarray


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
    obstacle_body_id: int = -1
    obstacle_pos: np.ndarray | None = None
    obstacle_size: np.ndarray | None = None
    friction_geom_ids: tuple[int, ...] = ()


def walker_total_mass(model: mujoco.MjModel) -> float:
    """Robot mass only — exclude world and the parked suite obstacle body."""
    total = 0.0
    for i in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if name == "obstacle_body":
            continue
        total += float(model.body_mass[i])
    return total


def walker_gravity(model: mujoco.MjModel) -> float:
    return abs(float(model.opt.gravity[2]))


def plant_friction_geom_ids(model: mujoco.MjModel, floor_id: int) -> tuple[int, ...]:
    """Floor plus robot geoms that actually generate walking contacts.

    MuJoCo mixes the two contacting geoms' sliding friction with an
    element-wise max, so changing only the floor leaves the feet sticky and
    the ice test is a no-op. These IDs are applied on the *plant* model only.
    """
    ids = [int(floor_id)]
    for geom_id in range(model.ngeom):
        if geom_id == floor_id:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name == "obstacle":
            continue
        if int(model.geom_contype[geom_id]) != 0:
            ids.append(int(geom_id))
    return tuple(ids)


def expected_belief_arrays(
    loka_state: dict[str, Any],
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None
]:
    """Nominal XML dynamics plus LOKA Model_Mutations — never plant overlays."""
    gears = np.array(loka_state["nominal_gears"], copy=True, dtype=float)
    friction = np.array(loka_state["nominal_friction"], copy=True, dtype=float)
    mass = np.array(loka_state["nominal_mass"], copy=True, dtype=float)
    inertia = (
        np.array(loka_state["nominal_inertia"], copy=True, dtype=float)
        if loka_state.get("nominal_inertia") is not None
        else None
    )
    ipos = (
        np.array(loka_state["nominal_ipos"], copy=True, dtype=float)
        if loka_state.get("nominal_ipos") is not None
        else None
    )
    for mut in loka_state.get("mutations") or []:
        try:
            kind = mut.get("type")
            attr = mut.get("attr")
            idx = int(mut["id"])
            val = mut.get("val")
            if kind == "actuator" and attr == "gear":
                gears[idx] = float(val)
            elif kind == "geom" and attr == "friction":
                for gid in friction_mutation_targets(idx, loka_state):
                    friction[gid] = friction_from_mutation(val, friction[gid])
            elif kind == "body" and attr == "mass":
                new_mass = float(val)
                if inertia is not None:
                    old_mass = float(mass[idx])
                    if old_mass > 0.0:
                        inertia[idx] = inertia[idx] * (new_mass / old_mass)
                mass[idx] = new_mass
            elif kind == "body" and attr in ("com", "ipos") and ipos is not None:
                ipos[idx] = mutation_vec3(val)
        except Exception:
            continue
    return gears, friction, mass, inertia, ipos


def sanitize_belief_worldview(model: mujoco.MjModel) -> None:
    """Lock the planner copy to the compiled nominal world.

    Suite overlays (raised box, visible backpack, ice, extra mass) are plant-only.
    Call this on the belief model *before* MJPC serializes it. Never call it on
    the plant — the plant must keep the collidable parked box so tests can raise it.
    """
    oid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle"))
    if oid >= 0:
        model.geom_contype[oid] = 0
        model.geom_conaffinity[oid] = 0
    bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "backpack"))
    if bid >= 0:
        # Visual placeholder only. The planner must not see a payload appear.
        model.geom_rgba[bid, 3] = 0.0


def assert_belief_isolated(
    belief_model: mujoco.MjModel, loka_state: dict[str, Any]
) -> None:
    """Fail if the MPC belief model carries a hidden suite perturbation."""
    gears, friction, mass, inertia, ipos = expected_belief_arrays(loka_state)
    if not np.allclose(belief_model.actuator_gear[:, 0], gears, atol=1e-9, rtol=0.0):
        raise RuntimeError(
            "MPC belief actuator_gear diverged from nominal+LOKA; "
            "a suite perturbation leaked into the planner model."
        )
    if not np.allclose(belief_model.geom_friction, friction, atol=1e-9, rtol=0.0):
        raise RuntimeError(
            "MPC belief geom_friction diverged from nominal+LOKA; "
            "ice/friction leaked into the planner model."
        )
    if not np.allclose(belief_model.body_mass, mass, atol=1e-9, rtol=0.0):
        raise RuntimeError(
            "MPC belief body_mass diverged from nominal+LOKA; "
            "a mass perturbation leaked into the planner model."
        )
    if inertia is not None and not np.allclose(
        belief_model.body_inertia, inertia, atol=1e-9, rtol=0.0
    ):
        raise RuntimeError(
            "MPC belief body_inertia diverged from nominal+LOKA; "
            "a mass/backpack perturbation leaked into the planner model."
        )
    if ipos is not None and not np.allclose(
        belief_model.body_ipos, ipos, atol=1e-9, rtol=0.0
    ):
        raise RuntimeError(
            "MPC belief body_ipos diverged from nominal+LOKA; "
            "a COM shift leaked into the planner model."
        )


def place_obstacle_on_model(
    model: mujoco.MjModel,
    *,
    visible: bool,
    pos,
    size,
    geom_id: int | None = None,
) -> int:
    """Park or raise the suite box by moving its body, not a worldbody geom.

    Changing only ``geom_pos`` on a worldbody geom looks right in the renderer
    and still lets the walker walk through. A dedicated body updates collision.
    """
    if geom_id is None:
        geom_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle"))
    if geom_id < 0:
        return -1
    size_arr = np.asarray(size, dtype=float).reshape(3)
    pos_arr = np.asarray(pos if visible else OBSTACLE_PARKED_POS, dtype=float).reshape(3)
    model.geom_size[geom_id] = size_arr
    model.geom_rbound[geom_id] = float(np.linalg.norm(size_arr))
    model.geom_contype[geom_id] = 1
    model.geom_conaffinity[geom_id] = 1
    body_id = int(model.geom_bodyid[geom_id])
    if body_id > 0:
        model.body_pos[body_id] = pos_arr
        model.geom_pos[geom_id] = 0.0
    else:
        model.geom_pos[geom_id] = pos_arr
    return geom_id


def recompute_mass_constants(model: mujoco.MjModel) -> None:
    """Update subtree mass / dof mass without touching the live plant state.

    ``mj_setConst(model, live_data)`` writes ``qpos0`` into ``live_data.qpos``.
    That teleports the walker back to spawn on every backpack step. Always use
    a scratch ``MjData``.
    """
    mujoco.mj_setConst(model, mujoco.MjData(model))


def capture_plant_snapshot(model: mujoco.MjModel) -> PlantSnapshot:
    return PlantSnapshot(
        actuator_gear=model.actuator_gear[:, 0].copy(),
        geom_friction=model.geom_friction.copy(),
        body_mass=model.body_mass.copy(),
        body_inertia=model.body_inertia.copy(),
        geom_pos=model.geom_pos.copy(),
        geom_size=model.geom_size.copy(),
        geom_contype=model.geom_contype.copy(),
        geom_conaffinity=model.geom_conaffinity.copy(),
        geom_rgba=model.geom_rgba.copy(),
        body_pos=model.body_pos.copy(),
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
    model.body_inertia[:] = snap.body_inertia
    model.geom_pos[:] = snap.geom_pos
    model.geom_size[:] = snap.geom_size
    model.geom_contype[:] = snap.geom_contype
    model.geom_conaffinity[:] = snap.geom_conaffinity
    model.geom_rgba[:] = snap.geom_rgba
    model.body_pos[:] = snap.body_pos
    data.xfrc_applied[:] = 0.0
    if set_const:
        recompute_mass_constants(model)


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

    if kind == "none":
        return resolved

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
        resolved.friction_geom_ids = plant_friction_geom_ids(model, floor_id)
        resolved.params["mu"] = float(raw["mu"])
        resolved.params["geom_ids"] = list(resolved.friction_geom_ids)

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
        size = raw.get("size", list(DEFAULT_OBSTACLE_SIZE))
        size_arr = np.asarray(size, dtype=float).reshape(-1)
        if size_arr.size != 3:
            raise ValueError("obstacle size must be [sx, sy, sz]")
        x = float(raw.get("x", 4.0))
        resolved.obstacle_id = int(geom_id)
        resolved.obstacle_body_id = int(model.geom_bodyid[geom_id])
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
    if resolved.friction_geom_ids:
        meta["friction_geom_ids"] = list(resolved.friction_geom_ids)
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
    *,
    recompute_mass_const: bool = False,
) -> None:
    """Overlay a hidden plant fault onto the physics model (after planning)."""
    kind = fault.kind
    if kind == "none":
        return

    if kind == "actuator_dead" and fault.actuator_id >= 0:
        model.actuator_gear[fault.actuator_id, 0] = 0.0
        return

    if kind == "friction" and fault.floor_id >= 0:
        mu = float(fault.params["mu"])
        geom_ids = fault.friction_geom_ids or (fault.floor_id,)
        for geom_id in geom_ids:
            # Absolute slide μ on floor *and* feet. Contact friction is
            # [slide, slide, spin, roll, roll]; MuJoCo combines the two
            # geoms with a max, so both sides must drop or ice is a no-op.
            gid = int(geom_id)
            model.geom_friction[gid] = friction_from_mutation(
                mu, snapshot.geom_friction[gid]
            )
        return

    if kind == "mass" and fault.body_id >= 0 and fault.delta_kg is not None:
        bid = int(fault.body_id)
        old_mass = float(snapshot.body_mass[bid])
        new_mass = old_mass + float(fault.delta_kg)
        model.body_mass[bid] = new_mass
        if old_mass > 0.0:
            model.body_inertia[bid] = snapshot.body_inertia[bid] * (new_mass / old_mass)
        pack_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "backpack")
        if pack_id >= 0:
            model.geom_rgba[pack_id] = BACKPACK_VISIBLE_RGBA
        # Never pass the live plant MjData to mj_setConst — it clobbers qpos.
        if recompute_mass_const:
            recompute_mass_constants(model)
        return

    if kind == "force" and fault.force_vec is not None and fault.body_id >= 0:
        elapsed = sim_time - activated_at
        if fault.duration_s is None or elapsed < fault.duration_s:
            data.xfrc_applied[fault.body_id, :3] = fault.force_vec
        return

    if kind == "obstacle" and fault.obstacle_id >= 0:
        place_obstacle_on_model(
            model,
            visible=True,
            pos=fault.obstacle_pos,
            size=fault.obstacle_size,
            geom_id=fault.obstacle_id,
        )


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
        restore_plant_snapshot(self.model, self.data, self.snapshot, set_const=True)
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
            recompute_mass_const=True,
        )

    def apply_physics(self, sim_time: float) -> None:
        """Re-apply the plant overlay for this physics step.

        Persistent faults (gear, friction, mass, obstacle) stay on the plant
        model. Force is time-limited via xfrc_applied on data. Mass constants
        are recomputed only on activate, never here.
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
            recompute_mass_const=False,
        )

    def obstacle_overlay(self) -> dict[str, Any] | None:
        overlay = self.visual_overlay()
        if overlay is None or overlay.get("kind") != "obstacle":
            return None
        return {
            "visible_from": overlay["visible_from"],
            "pos": overlay["obstacle"]["pos"],
            "size": overlay["obstacle"]["size"],
        }

    def visual_overlay(self) -> dict[str, Any] | None:
        """Recording-only appearance of the active plant fault."""
        if self.active is None or self.activated_at is None:
            return None
        overlay: dict[str, Any] = {
            "kind": self.active.kind,
            "visible_from": float(self.activated_at),
        }
        if self.active.kind == "obstacle" and self.active.obstacle_pos is not None:
            overlay["obstacle"] = {
                "pos": self.active.obstacle_pos.tolist(),
                "size": self.active.obstacle_size.tolist(),
            }
            return overlay
        if self.active.kind == "mass":
            overlay["backpack"] = {"rgba": BACKPACK_VISIBLE_RGBA.tolist()}
            return overlay
        if self.active.kind == "friction":
            overlay["ice"] = {"floor_rgba": ICE_FLOOR_RGBA.tolist()}
            return overlay
        return None
