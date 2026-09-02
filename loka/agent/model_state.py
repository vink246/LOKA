"""Runtime model state: LOKA mutations vs hidden physical faults."""

import mujoco


def resolve_mjcf_name(model, obj_enum, name):
    """Resolve an MJCF object name, tolerating LLM casing mistakes.

    A mutation naming ``Left_Knee`` instead of ``left_knee`` is a spelling
    slip, not a diagnosis error, and dropping it would silently discard an
    otherwise sound belief update.
    """
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


def apply_loka_mutations(model, loka_state, nominal_gears):
    """Reset to nominal hardware, then apply LOKA's internal-model mutations."""
    model.actuator_gear[:, 0] = nominal_gears
    for mut in loka_state.get("mutations", []):
        try:
            if mut["type"] == "actuator" and mut["attr"] == "gear":
                model.actuator_gear[mut["id"], 0] = float(mut["val"])
            elif mut["type"] == "geom" and mut["attr"] == "friction":
                if isinstance(mut["val"], list):
                    model.geom_friction[mut["id"], : len(mut["val"])] = mut["val"]
                else:
                    model.geom_friction[mut["id"], 0] = float(mut["val"])
            elif mut["type"] == "body" and mut["attr"] == "mass":
                model.body_mass[mut["id"]] = float(mut["val"])
        except Exception:
            pass


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
