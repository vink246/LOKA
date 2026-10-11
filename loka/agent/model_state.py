"""Runtime model state: LOKA mutations vs hidden physical faults."""

import mujoco
import numpy as np


def resolve_mjcf_name(model, obj_enum, name):
    """Resolve an MJCF object name, tolerating LLM casing mistakes."""
    if not name:
        return None, -1

    obj_id = mujoco.mj_name2id(model, obj_enum, name)
    if obj_id != -1:
        return name, obj_id

    lowered = name.lower()
    if lowered != name:
        obj_id = mujoco.mj_name2id(model, obj_enum, lowered)
        if obj_id != -1:
            return lowered, obj_id

    counts = {
        mujoco.mjtObj.mjOBJ_ACTUATOR: model.nu,
        mujoco.mjtObj.mjOBJ_GEOM: model.ngeom,
        mujoco.mjtObj.mjOBJ_BODY: model.nbody,
    }
    if obj_enum not in counts:
        return None, -1

    for i in range(counts[obj_enum]):
        candidate = mujoco.mj_id2name(model, obj_enum, i)
        if candidate and candidate.lower() == lowered:
            return candidate, i

    return None, -1


def mutation_vec3(val):
    """Parse a Model_Mutations value as a 3-vector, or raise."""
    if isinstance(val, (list, tuple, np.ndarray)):
        arr = np.asarray(val, dtype=float).reshape(-1)
    else:
        raise TypeError("expected a 3-vector")
    if arr.size != 3:
        raise ValueError("expected exactly 3 values")
    return arr


def ensure_nominal_snapshots(model, loka_state):
    """Remember the MJCF prior so each apply is an absolute belief, not a drift."""
    if loka_state.get("nominal_gears") is None:
        loka_state["nominal_gears"] = model.actuator_gear[:, 0].copy()
    if loka_state.get("nominal_friction") is None:
        loka_state["nominal_friction"] = model.geom_friction.copy()
    if loka_state.get("nominal_mass") is None:
        loka_state["nominal_mass"] = model.body_mass.copy()
    if loka_state.get("nominal_inertia") is None:
        loka_state["nominal_inertia"] = model.body_inertia.copy()
    if loka_state.get("nominal_ipos") is None:
        loka_state["nominal_ipos"] = model.body_ipos.copy()


def apply_loka_mutations(model, loka_state, nominal_gears=None):
    """Reset *belief* to nominal hardware, then apply LOKA Model_Mutations.

    Hidden plant faults must never be written here. ``mj_setConst`` uses a
    scratch ``MjData`` of the belief model — never plant data.
    """
    ensure_nominal_snapshots(model, loka_state)
    gears = loka_state.get("nominal_gears") if nominal_gears is None else nominal_gears
    if gears is not None:
        model.actuator_gear[:, 0] = gears

    friction = loka_state.get("nominal_friction")
    if friction is not None:
        model.geom_friction[:] = friction

    inertia = loka_state.get("nominal_inertia")
    if inertia is not None:
        model.body_inertia[:] = inertia

    mass = loka_state.get("nominal_mass")
    if mass is not None:
        model.body_mass[:] = mass

    ipos = loka_state.get("nominal_ipos")
    if ipos is not None:
        model.body_ipos[:] = ipos

    mass_changed = False
    com_changed = False
    for mut in loka_state.get("mutations", []):
        try:
            if mut["type"] == "actuator" and mut["attr"] == "gear":
                model.actuator_gear[mut["id"], 0] = float(mut["val"])
            elif mut["type"] == "geom" and mut["attr"] == "friction":
                row = model.geom_friction[mut["id"]]
                if isinstance(mut["val"], (list, tuple)):
                    row[: len(mut["val"])] = mut["val"]
                else:
                    row[0] = float(mut["val"])
            elif mut["type"] == "body" and mut["attr"] == "mass":
                idx = int(mut["id"])
                old_mass = float(model.body_mass[idx])
                new_mass = float(mut["val"])
                model.body_mass[idx] = new_mass
                if old_mass > 0.0:
                    model.body_inertia[idx] = model.body_inertia[idx] * (
                        new_mass / old_mass
                    )
                mass_changed = True
            elif mut["type"] == "body" and mut["attr"] in ("com", "ipos"):
                model.body_ipos[int(mut["id"])] = mutation_vec3(mut["val"])
                com_changed = True
        except Exception:
            pass

    if mass_changed or com_changed:
        mujoco.mj_setConst(model, mujoco.MjData(model))


def zero_dead_actuator_commands(actions, loka_state):
    """Zero control for actuators LOKA has declared dead in its internal model."""
    for mut in loka_state.get("mutations", []):
        try:
            if (
                mut["type"] == "actuator"
                and mut["attr"] == "gear"
                and float(mut["val"]) == 0.0
            ):
                actions[mut["id"]] = 0.0
        except Exception:
            pass
    return actions
