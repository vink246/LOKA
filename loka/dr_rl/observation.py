"""Proprioceptive observation for the MJPC Walker plant."""

from __future__ import annotations

import numpy as np

# qpos layout in models/walker/walker_modified.xml: rootz, rootx, rooty, joints...
ROOTZ_QPOS = 0
ROOTX_QPOS = 1
ROOTY_QPOS = 2
ROOTX_QVEL = 1


GAIT_PHASE_SIZE = 2


def gait_phase_features(gait_time: float, period: float) -> np.ndarray:
    """Open-loop walk clock as ``(sin 2π t/T, cos 2π t/T)``. Not noised."""
    frac = (float(gait_time) / max(float(period), 1e-6)) % 1.0
    ang = 2.0 * np.pi * frac
    return np.array([np.sin(ang), np.cos(ang)], dtype=np.float32)


def observation_size(
    model,
    *,
    include_last_action: bool = True,
    include_gait_phase: bool = False,
) -> int:
    size = (model.nq - 1) + model.nv
    if include_last_action:
        size += model.nu
    if include_gait_phase:
        size += GAIT_PHASE_SIZE
    return int(size)


def build_observation(
    data,
    last_action,
    *,
    include_last_action: bool = True,
    noise_rng: np.random.Generator | None = None,
    qpos_noise: float = 0.0,
    qvel_noise: float = 0.0,
    gait_phase: np.ndarray | None = None,
) -> np.ndarray:
    """qpos without rootx, full qvel, optional last action and gait phase."""
    qpos = np.asarray(data.qpos, dtype=np.float32)
    qvel = np.asarray(data.qvel, dtype=np.float32)
    qpos_wo_x = np.concatenate([qpos[:ROOTX_QPOS], qpos[ROOTX_QPOS + 1 :]])
    parts = [qpos_wo_x, qvel]
    if include_last_action:
        parts.append(np.asarray(last_action, dtype=np.float32).reshape(-1))
    if gait_phase is not None:
        parts.append(np.asarray(gait_phase, dtype=np.float32).reshape(-1))
    obs = np.concatenate(parts)
    if noise_rng is not None and (qpos_noise > 0.0 or qvel_noise > 0.0):
        noise = np.zeros_like(obs)
        n_qpos = qpos_wo_x.size
        n_qvel = qvel.size
        if qpos_noise > 0.0:
            noise[:n_qpos] = noise_rng.normal(0.0, qpos_noise, size=n_qpos)
        if qvel_noise > 0.0:
            noise[n_qpos : n_qpos + n_qvel] = noise_rng.normal(
                0.0, qvel_noise, size=n_qvel
            )
        obs = obs + noise
    return obs.astype(np.float32)
