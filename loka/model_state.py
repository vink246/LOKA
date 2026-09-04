"""Runtime model state: LOKA mutations vs hidden physical faults."""

import mujoco


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

    mass = loka_state.get("nominal_mass")
    if mass is not None:
        model.body_mass[:] = mass

    mass_changed = False
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
                mass_changed = True
        except Exception:
            pass

    if mass_changed:
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
