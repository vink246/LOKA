"""Interactive plant-fault helpers and MuJoCo viewer overlays."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from loka.stand_loka.faults import FaultSpec, _FaultBackup

G = 9.81
FORCE_ARROW_SCALE = 0.004  # m per Newton
WEIGHT_ARROW_SCALE = 0.002  # m per Newton of added weight
MAX_ARROW_LEN = 0.55

MASS_BODY_ALIASES = {
    "torso": "torso_link",
    "torso_link": "torso_link",
    "chest": "torso_link",
    "left": "left_shoulder_roll_link",
    "left_shoulder": "left_shoulder_roll_link",
    "l": "left_shoulder_roll_link",
    "right": "right_shoulder_roll_link",
    "right_shoulder": "right_shoulder_roll_link",
    "r": "right_shoulder_roll_link",
}


@dataclass
class FaultVizState:
    """Live visualization / restore state for interactive faults."""

    # (body_id, delta_kg) for each active mass load (supports multi-shoulder).
    mass_loads: list[tuple[int, float]] = field(default_factory=list)
    friction_mu: float | None = None
    floor_geom_id: int = -1
    floor_rgba_nominal: np.ndarray | None = None
    dead_actuator: str | None = None
    backups: list[_FaultBackup] = field(default_factory=list)
    # Last external wrench applied to the plant this control step [N].
    external_force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    external_body_id: int = -1

    @property
    def mass_delta_kg(self) -> float:
        return float(sum(delta for _, delta in self.mass_loads))


def parse_fault_command(line: str) -> FaultSpec | str | None:
    """Parse an interactive fault / meta command.

    Returns
    -------
    FaultSpec
        Inject this fault now.
    ``\"clear\"``
        Restore plant mutations.
    ``\"help\"``
        Print help.
    None
        Not a fault command (treat as operator text).
    """
    text = line.strip()
    if not text:
        return None
    lower = text.lower()
    if lower in {"help", "?", "h"}:
        return "help"
    if lower in {"clear", "reset-faults", "unfault"}:
        return "clear"

    parts = lower.split()
    cmd = parts[0]

    if cmd in {"mass", "m", "weight"}:
        # mass [kg] [torso|left|right]
        # mass left [kg]   /   mass right 3
        kg = 5.0
        body_key = "torso"
        args = parts[1:]
        if args and args[0] in MASS_BODY_ALIASES:
            body_key = args[0]
            if len(args) > 1 and _is_float(args[1]):
                kg = float(args[1])
        elif args and _is_float(args[0]):
            kg = float(args[0])
            if len(args) > 1:
                body_key = args[1]
        body = MASS_BODY_ALIASES.get(body_key, body_key)
        return FaultSpec("mass", 0.0, {"delta_kg": kg, "body": body})

    if cmd in {"friction", "mu", "ice"}:
        if cmd == "ice":
            mu = 0.2
        else:
            mu = float(parts[1]) if len(parts) > 1 else 0.25
        return FaultSpec("friction", 0.0, {"mu": mu})

    if cmd in {"push", "p", "shove"}:
        impulse = 6.0
        direction = np.array([1.0, 0.0, 0.0])
        args = parts[1:]
        if args and _is_float(args[0]):
            impulse = float(args[0])
            args = args[1:]
        if not args:
            pass
        elif args[0] in {"forward", "fwd", "f", "+x"}:
            direction = np.array([1.0, 0.0, 0.0])
        elif args[0] in {"back", "backward", "b", "-x"}:
            direction = np.array([-1.0, 0.0, 0.0])
        elif args[0] in {"left", "l", "+y"}:
            direction = np.array([0.0, 1.0, 0.0])
        elif args[0] in {"right", "r", "-y"}:
            direction = np.array([0.0, -1.0, 0.0])
        elif len(args) >= 2 and _is_float(args[0]) and _is_float(args[1]):
            direction = np.array([float(args[0]), float(args[1]), 0.0])
            n = np.linalg.norm(direction)
            if n > 1e-9:
                direction /= n
        return FaultSpec(
            "push",
            0.0,
            {"impulse": impulse, "direction": tuple(direction.tolist())},
        )

    if cmd in {"dead", "kill", "amputate"}:
        name = parts[1] if len(parts) > 1 else "right_knee"
        return FaultSpec("actuator_dead", 0.0, {"actuator": name})

    if cmd == "fault" and len(parts) >= 2:
        return parse_fault_command(" ".join(parts[1:]))

    return None


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


FAULT_HELP = """\
Interactive commands (type then Enter):
  mass [kg] [torso|left|right]   add mass (default 5 kg on torso)
  mass left|right [kg]           shoulder-offset mass (asymmetric CoM)
  friction [mu] | ice            set floor friction (default 0.25; ice=0.2)
  push [N.s] [dir]               shove pelvis (default 6 forward)
                                 dir: forward|back|left|right  or  fx fy
  dead [actuator]                zero actuator (default right_knee)
  clear                          restore mass / friction / dead-actuator edits
  help                           this text
  <any other text>               operator request (LLM on only)

Viewer keys: 1=torso mass  2=ice  3=push  4=dead knee
             6=left shoulder mass  7=right shoulder mass  5=clear  H=help
"""


def _add_arrow(
    scene: mujoco.MjvScene,
    origin: np.ndarray,
    vector: np.ndarray,
    rgba: tuple[float, float, float, float],
    *,
    width: float = 0.02,
) -> None:
    length = float(np.linalg.norm(vector))
    if length < 1e-6 or scene.ngeom >= scene.maxgeom:
        return
    tip = origin + vector
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
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        width,
        origin,
        tip,
    )
    scene.ngeom += 1


def _add_sphere(
    scene: mujoco.MjvScene,
    center: np.ndarray,
    radius: float,
    rgba: tuple[float, float, float, float],
) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]),
        center,
        np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _add_capsule(
    scene: mujoco.MjvScene,
    p0: np.ndarray,
    p1: np.ndarray,
    radius: float,
    rgba: tuple[float, float, float, float],
) -> None:
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


def render_fault_overlays(
    viewer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viz: FaultVizState,
) -> None:
    """Draw push force arrows and added-mass markers into ``viewer.user_scn``."""
    scene = viewer.user_scn
    scene.ngeom = 0

    force = np.asarray(viz.external_force, dtype=float)
    body_id = viz.external_body_id
    if body_id < 0:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if body_id >= 0 and float(np.linalg.norm(force)) > 1.0:
        origin = data.xpos[body_id].copy()
        origin[2] += 0.05
        vec = force * FORCE_ARROW_SCALE
        n = float(np.linalg.norm(vec))
        if n > MAX_ARROW_LEN:
            vec *= MAX_ARROW_LEN / n
        _add_arrow(scene, origin, vec, (0.1, 0.85, 1.0, 0.95), width=0.025)

    for load_body, delta_kg in viz.mass_loads:
        if delta_kg <= 0.05 or load_body < 0:
            continue
        center = data.xpos[load_body].copy()
        center[2] += 0.08
        radius = 0.045 + 0.008 * min(delta_kg, 15.0)
        _add_sphere(scene, center, radius, (1.0, 0.45, 0.05, 0.9))
        weight = np.array([0.0, 0.0, -delta_kg * G * WEIGHT_ARROW_SCALE])
        wn = float(np.linalg.norm(weight))
        if wn > MAX_ARROW_LEN:
            weight *= MAX_ARROW_LEN / wn
        _add_arrow(scene, center, weight, (1.0, 0.25, 0.05, 0.95), width=0.028)


def render_gait_overlays(viewer, controller) -> None:
    """Draw Raibert foothold + Bézier swing path into ``viewer.user_scn``.

    Call *after* :func:`render_fault_overlays` (which clears ``ngeom``).
    Markers:
      - cyan sphere  = planned foothold (where the swing foot should land)
      - magenta sphere = current swing des_pos on the Bézier
      - yellow pearls = sampled swing path
      - green sphere  = CoM XY reference (at sole height)
    """
    gait = getattr(controller, "_last_gait", None)
    if gait is None:
        return
    from loka.control.gait import MODE_WALK, _bezier_swing_pos

    if float(controller.gait.config.mode) < MODE_WALK:
        return
    if float(controller.gait.config.speed) <= 1e-4 and controller.gait._cmd_speed <= 1e-4:
        return

    scene = viewer.user_scn
    ground_z = float(getattr(controller, "_ground_height", 0.0))

    # CoM reference (support-tied midline).
    com_ref = np.array(
        [gait.com_ref_xy[0], gait.com_ref_xy[1], ground_z + 0.01], dtype=float
    )
    _add_sphere(scene, com_ref, 0.025, (0.2, 0.95, 0.3, 0.85))

    swing = gait.swing
    if not swing.active:
        return

    fh = np.asarray(swing.foothold, dtype=float).copy()
    fh[2] = ground_z + 0.015
    # Foothold target — bright cyan, slightly larger.
    _add_sphere(scene, fh, 0.035, (0.1, 0.95, 1.0, 0.95))
    # Stem so it reads against the floor.
    _add_capsule(
        scene,
        fh,
        np.array([fh[0], fh[1], ground_z + 0.12]),
        0.008,
        (0.1, 0.85, 1.0, 0.7),
    )

    # Current desired swing pose.
    des = np.asarray(swing.des_pos, dtype=float).copy()
    _add_sphere(scene, des, 0.028, (1.0, 0.2, 0.85, 0.95))

    # Sampled Bézier path from lift-off to foothold.
    start = np.asarray(swing.start, dtype=float)
    end = np.asarray(swing.foothold, dtype=float)
    height = float(controller.gait.config.swing_height)
    prev = None
    for s in np.linspace(0.0, 1.0, 9):
        p = _bezier_swing_pos(start, end, float(s), swing_height=height, ground_z=ground_z)
        _add_sphere(scene, p, 0.012, (1.0, 0.9, 0.15, 0.8))
        if prev is not None:
            _add_capsule(scene, prev, p, 0.006, (1.0, 0.85, 0.1, 0.55))
        prev = p
