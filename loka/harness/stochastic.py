"""Seeded initial-state noise for evaluation trials.

Trial ``k`` uses ``base_seed + k`` for every baseline, so a comparison starts
from the same pose. Noise is applied to the joint positions and the joint
velocities only. The floating-base pose from the stand keyframe, including the
foot-flat quaternion, is left alone.
"""

from __future__ import annotations

import mujoco
import numpy as np


def trial_seed(base_seed: int, trial: int) -> int:
    if int(trial) < 0:
        raise ValueError("trial must be >= 0")
    return int(base_seed) + int(trial)


def apply_seeded_state_noise(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    seed: int,
    scale: float,
) -> None:
    """Add ``Uniform(-scale, scale)`` to joint ``qpos`` and joint ``qvel``.

    ``scale == 0`` leaves the reset state untouched. Call this after the
    keyframe reset. The first 7 ``qpos`` (xyz and quaternion) and the first 6
    ``qvel`` (base twist) are not modified.
    """
    scale = float(scale)
    if scale < 0.0:
        raise ValueError("init_noise must be >= 0")
    if scale == 0.0:
        return
    rng = np.random.default_rng(int(seed))
    qpos = np.array(data.qpos, dtype=float)
    if model.nq > 7:
        qpos[7:] += rng.uniform(-scale, scale, size=model.nq - 7)
    data.qpos[:] = qpos
    qvel = np.array(data.qvel, dtype=float)
    if model.nv > 6:
        qvel[6:] += rng.uniform(-scale, scale, size=model.nv - 6)
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
