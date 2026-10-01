"""Floor contact forces on the walker for the suite logs."""

from __future__ import annotations

import mujoco
import numpy as np

# The suite box is a separate body. Its floor contact is not a reaction on the robot.
_EXCLUDED_GEOM_NAMES = frozenset({"floor", "obstacle"})
_EXCLUDED_BODY_NAMES = frozenset({"obstacle_body"})
FOOT_GEOM_NAMES = frozenset({"right_foot", "left_foot"})


def _force_on_geom2(contact, force6: np.ndarray) -> np.ndarray:
    """World-frame contact force applied to ``contact.geom2``.

    ``mj_contactForce`` is in the contact frame (normal, then two tangents).
    Those axes are stored in ``contact.frame`` in world coordinates, and a
    positive normal component pushes geom2 along the normal. With the floor
    as geom1 the normal points up, so this vector is the ground reaction on
    the other geom.
    """
    axes = np.asarray(contact.frame, dtype=float).reshape(3, 3)
    force = np.asarray(force6, dtype=float).reshape(-1)
    return force[0] * axes[0] + force[1] * axes[1] + force[2] * axes[2]


def walker_ground_geom_ids(model: mujoco.MjModel) -> tuple[int, ...]:
    """Geoms on the robot that can carry a ground reaction.

    Feet, knees (leg capsules), thighs, and the torso all qualify. The floor
    and the parked suite box do not.
    """
    ids: list[int] = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name in _EXCLUDED_GEOM_NAMES:
            continue
        body_id = int(model.geom_bodyid[geom_id])
        if body_id <= 0:
            continue
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_name in _EXCLUDED_BODY_NAMES:
            continue
        ids.append(int(geom_id))
    return tuple(ids)


def nonfoot_geom_names(model: mujoco.MjModel) -> tuple[str, ...]:
    """Robot geoms other than the feet. Knees, thighs, and the torso live here."""
    names: list[str] = []
    for geom_id in walker_ground_geom_ids(model):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name and name not in FOOT_GEOM_NAMES:
            names.append(name)
    return tuple(names)


def sample_floor_contacts(
    model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Floor force on the walker and the non-foot geoms touching the floor.

    The name tuple is recorded at this step. Later plots read that log instead
    of guessing which body was down from the pose.
    """
    total = np.zeros(3, dtype=float)
    floor_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor"))
    if floor_id < 0:
        return total, ()
    robot_ids = set(walker_ground_geom_ids(model))
    if not robot_ids:
        return total, ()

    touching: set[str] = set()
    buf = np.zeros(6, dtype=float)
    for i in range(int(data.ncon)):
        contact = data.contact[i]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if floor_id not in (geom1, geom2):
            continue
        if geom1 not in robot_ids and geom2 not in robot_ids:
            continue
        mujoco.mj_contactForce(model, data, i, buf)
        force_on_geom2 = _force_on_geom2(contact, buf)
        if geom2 in robot_ids and geom1 == floor_id:
            total += force_on_geom2
            robot_geom = geom2
        elif geom1 in robot_ids and geom2 == floor_id:
            total -= force_on_geom2
            robot_geom = geom1
        else:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom) or ""
        if name and name not in FOOT_GEOM_NAMES:
            touching.add(name)
    return total, tuple(sorted(touching))


def ground_reaction_force(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """World-frame force the floor applies to the walker, shape ``(3,)``.

    Every robot geom in contact with the floor contributes: feet, knees, and
    the torso when it is down. Vertical +z is the ground pushing up. The
    horizontal part is shear from propulsion and from parts that stay planted
    and drag.
    """
    force, _touching = sample_floor_contacts(model, data)
    return force


def ground_reaction_channels(force: np.ndarray) -> dict[str, float]:
    """Magnitude, vertical component, and horizontal magnitude of a GRF vector."""
    vec = np.asarray(force, dtype=float).reshape(3)
    horizontal = float(np.hypot(vec[0], vec[1]))
    return {
        "grf_mag": float(np.linalg.norm(vec)),
        "grf_vertical": float(vec[2]),
        "grf_horizontal": horizontal,
    }
