"""Runtime model state: LOKA mutations vs hidden physical faults."""


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
