"""MuJoCo viewer overlays: what the controller believes, drawn in the scene.

Every overlay here answers a question the numbers alone answer badly. A
foothold 4 cm to the left of where it should be is invisible in a plot of
lateral CoM error but obvious as a marker sitting outside the sole; a swing
arc that clips the ground reads as a torque spike and nothing else.

Overlays are drawn into ``viewer.user_scn``, which the caller owns. The first
overlay of a frame clears it -- see :func:`begin_frame` -- and the rest append,
so the draw order is the caller's to choose.
"""

from __future__ import annotations

import mujoco
import numpy as np

from loka.control.gait import swing_reference

G = 9.81
FORCE_ARROW_SCALE = 0.004  # m per Newton
WEIGHT_ARROW_SCALE = 0.002  # m per Newton of added weight
MAX_ARROW_LEN = 0.55

#: Overlay colours, RGBA.
FOOTHOLD = (0.10, 0.95, 1.00, 0.95)
SWING_DES = (1.00, 0.20, 0.85, 0.95)
SWING_ARC = (1.00, 0.90, 0.15, 0.80)
COM_REF = (0.20, 0.95, 0.30, 0.85)
DCM_REF = (1.00, 0.55, 0.10, 0.85)
PUSH = (0.10, 0.85, 1.00, 0.95)
ADDED_MASS = (1.00, 0.45, 0.05, 0.90)


def begin_frame(viewer) -> None:
    """Drop last frame's overlays. Call once, before the first draw."""
    viewer.user_scn.ngeom = 0


def add_arrow(scene, origin, vector, rgba, *, width: float = 0.02) -> None:
    length = float(np.linalg.norm(vector))
    if length < 1e-6 or scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom, mujoco.mjtGeom.mjGEOM_ARROW, width, origin, origin + vector
    )
    scene.ngeom += 1


def add_sphere(scene, center, radius: float, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]),
        np.asarray(center, dtype=float),
        np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def add_capsule(scene, p0, p1, radius: float, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        np.asarray(p0, dtype=float),
        np.asarray(p1, dtype=float),
    )
    scene.ngeom += 1


def _marker(scene, xy, ground_z: float, radius: float, rgba, *, lift: float) -> None:
    """A floor disc-ish sphere with a stem, so it reads against the ground."""
    base = np.array([float(xy[0]), float(xy[1]), ground_z + lift])
    add_sphere(scene, base, radius, rgba)
    add_capsule(
        scene,
        base,
        base + np.array([0.0, 0.0, 0.12]),
        0.008,
        (*rgba[:3], 0.55),
    )


def render_gait_overlays(viewer, controller) -> None:
    """Draw the footstep plan, swing arc and centroidal references.

    Markers:
      green  -- CoM xy reference        magenta -- commanded swing pose
      orange -- DCM reference           yellow  -- the swing arc, sampled
      cyan   -- the planned foothold
    """
    gait_out = getattr(controller, "_last_gait", None)
    if gait_out is None or not controller.gait.wants_walk():
        return

    scene = viewer.user_scn
    ground_z = float(getattr(controller, "_ground_height", 0.0))

    _marker(scene, gait_out.com_ref_xy, ground_z, 0.025, COM_REF, lift=0.01)
    _marker(scene, gait_out.dcm_ref, ground_z, 0.020, DCM_REF, lift=0.01)

    swing = gait_out.swing
    if not swing.active:
        return

    _marker(scene, swing.foothold[:2], ground_z, 0.035, FOOTHOLD, lift=0.015)
    add_sphere(scene, swing.des_pos, 0.028, SWING_DES)

    start = np.asarray(swing.start, dtype=float)
    end = np.asarray(swing.foothold, dtype=float)
    height = float(controller.gait.config.swing_height)
    previous = None
    for s in np.linspace(0.0, 1.0, 9):
        point, _, _ = swing_reference(start, end, float(s), swing_height=height)
        add_sphere(scene, point, 0.012, SWING_ARC)
        if previous is not None:
            add_capsule(scene, previous, point, 0.006, (*SWING_ARC[:3], 0.55))
        previous = point


def render_fault_overlays(viewer, model, data, viz) -> None:
    """Draw the pushes and added masses an operator has injected.

    ``viz`` is any object carrying ``external_force``, ``external_body_id`` and
    ``mass_loads``; see ``loka.agent.interactive.FaultVizState``.
    """
    scene = viewer.user_scn

    force = np.asarray(viz.external_force, dtype=float)
    body_id = viz.external_body_id
    if body_id < 0:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if body_id >= 0 and float(np.linalg.norm(force)) > 1.0:
        origin = data.xpos[body_id].copy()
        origin[2] += 0.05
        vector = force * FORCE_ARROW_SCALE
        norm = float(np.linalg.norm(vector))
        if norm > MAX_ARROW_LEN:
            vector *= MAX_ARROW_LEN / norm
        add_arrow(scene, origin, vector, PUSH, width=0.025)

    for load_body, delta_kg in viz.mass_loads:
        if delta_kg <= 0.05 or load_body < 0:
            continue
        center = data.xpos[load_body].copy()
        center[2] += 0.08
        add_sphere(scene, center, 0.045 + 0.008 * min(delta_kg, 15.0), ADDED_MASS)
        weight = np.array([0.0, 0.0, -delta_kg * G * WEIGHT_ARROW_SCALE])
        norm = float(np.linalg.norm(weight))
        if norm > MAX_ARROW_LEN:
            weight *= MAX_ARROW_LEN / norm
        add_arrow(scene, center, weight, (1.0, 0.25, 0.05, 0.95), width=0.028)
