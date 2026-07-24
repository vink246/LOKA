"""Shared MuJoCo MJCF numeric helpers for LOKA."""

import mujoco

SUPPORTED_PLANNER_NUMERICS = frozenset({"agent_horizon", "sampling_exploration"})


def set_model_numeric(model, name, value):
    """Set an MJCF custom numeric on the MuJoCo model."""
    numeric_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, name)
    if numeric_id == -1:
        raise ValueError(f"Custom numeric '{name}' not found in model")
    addr = model.numeric_adr[numeric_id]
    model.numeric_data[addr] = float(value)


def get_model_numeric(model, name):
    """Read an MJCF custom numeric from the MuJoCo model."""
    numeric_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, name)
    if numeric_id == -1:
        return None
    addr = model.numeric_adr[numeric_id]
    return float(model.numeric_data[addr])
