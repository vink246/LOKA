"""Seeded initial-state noise for Walker suite trials.

MuJoCo's ``mj_step`` is a deterministic function of the model and state.
The suite makes trials differ by drawing a fresh initial ``qpos`` / ``qvel``
from a seeded RNG, then integrating the usual plant. The same seed always
rebuilds the same initial state. Trial ``k`` uses ``base_seed + k``, and that
seed is shared by every baseline on that trial so the comparison starts from
the same pose.
"""

from __future__ import annotations

import mujoco
import numpy as np

# Same half-width the DR-RL trainer uses for reset noise (meters and radians).
DEFAULT_INIT_NOISE = 0.005


def trial_seed(base_seed: int, trial: int) -> int:
    """Seed for trial ``k``. Shared across tests and baselines."""
    if int(trial) < 0:
        raise ValueError("trial must be >= 0")
    return int(base_seed) + int(trial)


def apply_seeded_state_noise(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    seed: int,
    scale: float,
) -> None:
    """Add ``Uniform(-scale, scale)`` to ``qpos0`` and to ``qvel``.

    ``scale == 0`` leaves the reset state untouched. ``qpos`` noise is applied
    on top of ``model.qpos0`` (root z/x in meters, pitch and joints in radians).
    ``qvel`` noise uses the same half-width in m/s and rad/s.
    """
    scale = float(scale)
    if scale < 0.0:
        raise ValueError("init_noise must be >= 0")
    if scale == 0.0:
        return
    rng = np.random.default_rng(int(seed))
    data.qpos[:] = np.asarray(model.qpos0, dtype=float) + rng.uniform(
        -scale, scale, size=model.nq
    )
    data.qvel[:] = rng.uniform(-scale, scale, size=model.nv)
    mujoco.mj_forward(model, data)
