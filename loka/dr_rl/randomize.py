"""In-distribution dynamics randomization for the Walker plant.

Ranges are chosen so LOKA suite faults stay out of distribution:
dead hip (gear=0), ice (mu=0.2), backpack (+25% torso mass), 0.5-bodyweight shove.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from loka.walker_suite.faults import (
    PlantSnapshot,
    capture_plant_snapshot,
    restore_plant_snapshot,
)


@dataclass
class NominalDynamics:
    plant: PlantSnapshot
    dof_damping: np.ndarray
    floor_id: int


@dataclass
class ResetRandomization:
    mu: float
    mass_scale: np.ndarray
    gear_scale: np.ndarray
    damping_scale: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "mu": float(self.mu),
            "mass_scale": self.mass_scale.tolist(),
            "gear_scale": self.gear_scale.tolist(),
            "damping_scale": self.damping_scale.tolist(),
        }


def capture_nominal_dynamics(model: mujoco.MjModel) -> NominalDynamics:
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0:
        raise ValueError("Walker model has no geom named 'floor'")
    return NominalDynamics(
        plant=capture_plant_snapshot(model),
        dof_damping=np.array(model.dof_damping, copy=True),
        floor_id=int(floor_id),
    )


def restore_nominal_dynamics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    nominal: NominalDynamics,
) -> None:
    restore_plant_snapshot(model, data, nominal.plant, set_const=False)
    model.dof_damping[:] = nominal.dof_damping
    mujoco.mj_setConst(model, data)


def sample_reset_randomization(
    rng: np.random.Generator,
    model: mujoco.MjModel,
    dr_cfg: dict[str, Any],
) -> ResetRandomization:
    mu_lo, mu_hi = (float(v) for v in dr_cfg["friction"])
    mass_lo, mass_hi = (float(v) for v in dr_cfg["mass_scale"])
    gear_lo, gear_hi = (float(v) for v in dr_cfg["gear_scale"])
    damp_lo, damp_hi = (float(v) for v in dr_cfg["damping_scale"])
    if gear_lo <= 0.0:
        raise ValueError("gear_scale lower bound must be > 0 (dead actuators are OOD)")

    mass_scale = np.ones(model.nbody, dtype=float)
    if model.nbody > 1:
        mass_scale[1:] = rng.uniform(mass_lo, mass_hi, size=model.nbody - 1)
    return ResetRandomization(
        mu=float(rng.uniform(mu_lo, mu_hi)),
        mass_scale=mass_scale,
        gear_scale=rng.uniform(gear_lo, gear_hi, size=model.nu),
        damping_scale=rng.uniform(damp_lo, damp_hi, size=model.nv),
    )


def apply_reset_randomization(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    nominal: NominalDynamics,
    sample: ResetRandomization,
) -> None:
    restore_nominal_dynamics(model, data, nominal)
    model.body_mass[:] = nominal.plant.body_mass * sample.mass_scale
    model.body_mass[0] = nominal.plant.body_mass[0]
    model.actuator_gear[:, 0] = nominal.plant.actuator_gear * sample.gear_scale
    model.dof_damping[:] = nominal.dof_damping * sample.damping_scale
    model.geom_friction[nominal.floor_id, 0] = sample.mu
    mujoco.mj_setConst(model, data)


def sample_training_push(
    rng: np.random.Generator,
    model: mujoco.MjModel,
    dr_cfg: dict[str, Any],
):
    """Maybe return a (onset_s, ResolvedPerturbation) mid-episode shove."""
    from loka.walker_suite.faults import resolve_perturbation

    if float(rng.random()) >= float(dr_cfg.get("push_prob", 0.0)):
        return None
    f_lo, f_hi = (float(v) for v in dr_cfg["push_force_frac"])
    d_lo, d_hi = (float(v) for v in dr_cfg["push_duration_s"])
    t_lo, t_hi = (float(v) for v in dr_cfg["push_onset_s"])
    frac = float(rng.uniform(f_lo, f_hi))
    if frac <= 0.0:
        return None
    direction = "forward" if rng.random() < 0.5 else "backward"
    resolved = resolve_perturbation(
        {
            "kind": "force",
            "force_frac": frac,
            "direction": direction,
            "duration_s": float(rng.uniform(d_lo, d_hi)),
        },
        model,
    )
    onset = float(rng.uniform(t_lo, t_hi))
    return onset, resolved
