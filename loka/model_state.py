"""Runtime model state: LOKA mutations vs hidden physical faults."""

import mujoco
import numpy as np


def mutation_vec3(val):
    """Parse a Model_Mutations value as a 3-vector, or raise."""
    if isinstance(val, (list, tuple, np.ndarray)):
        arr = np.asarray(val, dtype=float).reshape(-1)
    else:
        raise TypeError("expected a 3-vector")
    if arr.size != 3:
        raise ValueError("expected exactly 3 values")
    return arr


def friction_from_mutation(val, old_row):
    """Absolute slide μ, or a 3-vector. Spin/roll scale with slide if a scalar."""
    old = np.asarray(old_row, dtype=float).reshape(-1)
    if isinstance(val, (list, tuple, np.ndarray)):
        arr = np.asarray(val, dtype=float).reshape(-1)
        out = np.array(old, copy=True)
        out[: min(out.size, arr.size)] = arr[: min(out.size, arr.size)]
        return out
    mu = float(val)
    old_slide = float(old[0]) if float(old[0]) > 0.0 else 1.0
    return old * (mu / old_slide)


def friction_mutation_targets(geom_id, loka_state):
    """Floor friction must land on every walking-contact geom.

    MuJoCo takes the element-wise *max* of the two geoms, so writing only the
    floor leaves the feet at 0.7 and contact μ never drops.
    """
    idx = int(geom_id)
    contact_ids = loka_state.get("friction_contact_ids")
    floor_id = loka_state.get("floor_geom_id")
    if (
        contact_ids
        and floor_id is not None
        and idx == int(floor_id)
    ):
        return [int(gid) for gid in contact_ids]
    return [idx]


def apply_loka_mutations(model, loka_state, nominal_gears=None):
    """Reset *belief* to nominal hardware, then apply LOKA Model_Mutations.

    This is the only path that may change the planner's internal model. Hidden
    plant faults must never be written here. ``mj_setConst`` uses a scratch
    ``MjData`` of the belief model — never plant data.
    """
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
                for gid in friction_mutation_targets(mut["id"], loka_state):
                    model.geom_friction[gid] = friction_from_mutation(
                        mut["val"], model.geom_friction[gid]
                    )
            elif mut["type"] == "body" and mut["attr"] == "mass":
                idx = int(mut["id"])
                old_mass = float(model.body_mass[idx])
                new_mass = float(mut["val"])
                model.body_mass[idx] = new_mass
                if inertia is not None and old_mass > 0.0:
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
