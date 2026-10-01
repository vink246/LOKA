"""Interactive plant-fault commands typed at a running simulation.

Rendering of the resulting pushes and masses lives in :mod:`loka.viz`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from loka.agent.faults import (
    FaultSpec,
    _FaultBackup,
    apply_plant_fault,
    clear_plant_faults,
    release_actuator,
    resolve_actuator,
)

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
    # actuator name -> fraction of nominal gear (1.0 is healthy).
    actuator_scales: dict[str, float] = field(default_factory=dict)
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

    if cmd in {"scale", "weak", "weaken"}:
        # scale [actuator] [fraction]   /   weak [actuator]
        # scale 0.5 right_knee          /   scale right_knee 0.5
        actuator = "right_knee"
        scale = 0.5
        args = parts[1:]
        if args and not _is_float(args[0]):
            actuator = args[0]
            args = args[1:]
        if args and _is_float(args[0]):
            scale = float(args[0])
            args = args[1:]
        if args and not _is_float(args[0]):
            actuator = args[0]
        return FaultSpec(
            "actuator_scale", 0.0, {"actuator": actuator, "scale": scale}
        )

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
  scale [actuator] [fraction]    deliver that fraction of nominal torque
  weak [actuator]                same as scale, default right_knee at 0.5
  clear                          restore mass / friction / actuator edits
  help                           this text
  <any other text>               locked mission directive (LLM pursues it)

Viewer keys (run_loka and the dashboard MuJoCo window, or the dashboard GUI):
  2  right knee at 50% torque       3  push forward
  4  ice                            5  +5 kg on the left shoulder
  6  +5 kg on the torso             7  +5 kg on the right shoulder
  8  clear plant edits              H  help
  Repeating 5/6/7 stacks another 5 kg. Repeating 2 stays at 50% (does not halve again).
"""


def _digit_key(keycode: int) -> str | None:
    """Map a GLFW / DearPyGui digit or numpad key to ``'0'``..``'9'``."""
    if ord("0") <= keycode <= ord("9"):
        return chr(keycode)
    # GLFW keypad: KP_0 = 320 ... KP_9 = 329. DearPyGui uses the same codes.
    if 320 <= keycode <= 329:
        return str(keycode - 320)
    return None


class KeyDebounce:
    """Drop OS key-repeat so holding ``1`` does not stack several masses."""

    def __init__(self, interval: float = 0.3) -> None:
        self.interval = interval
        self._last: dict[int, float] = {}

    def allow(self, keycode: int) -> bool:
        now = time.perf_counter()
        if now - self._last.get(keycode, -1e9) < self.interval:
            return False
        self._last[keycode] = now
        return True


def fault_for_keycode(keycode: int) -> FaultSpec | str | None:
    """Map a viewer key to a fault, ``\"clear\"``, ``\"help\"``, or None."""
    digit = _digit_key(keycode)
    if digit == "5":
        return FaultSpec(
            "mass", 0.0, {"delta_kg": 5.0, "body": "left_shoulder_roll_link"}
        )
    if digit == "2":
        return FaultSpec(
            "actuator_scale", 0.0, {"actuator": "right_knee", "scale": 0.5}
        )
    if digit == "3":
        return FaultSpec(
            "push", 0.0, {"impulse": 6.0, "direction": (1.0, 0.0, 0.0)}
        )
    if digit == "8":
        return "clear"
    if digit == "6":
        return FaultSpec("mass", 0.0, {"delta_kg": 5.0, "body": "torso_link"})
    if digit == "7":
        return FaultSpec(
            "mass", 0.0, {"delta_kg": 5.0, "body": "right_shoulder_roll_link"}
        )
    if digit == "4":
        return FaultSpec("friction", 0.0, {"mu": 0.2})
    if keycode in (ord("h"), ord("H")):
        return "help"
    return None


def _forget_actuator_viz(viz: FaultVizState, name: str | None) -> None:
    if not name:
        return
    viz.actuator_scales.pop(name, None)
    if viz.dead_actuator == name:
        viz.dead_actuator = None


def commit_fault(sim, viz: FaultVizState, fault: FaultSpec) -> tuple[bool, str]:
    """Apply one plant fault and record it on ``viz``.

    Actuator scale and dead edits replace any earlier edit of that same
    actuator, so pressing the knee key again stays at 50% of nominal torque
    instead of compounding to 25%.
    """
    from loka.sim import Push

    now = float(sim.data.time)
    fault = FaultSpec(fault.kind, now, dict(fault.params))
    if fault.kind == "push":
        impulse = float(fault.params.get("impulse", 6.0))
        direction = np.asarray(
            fault.params.get("direction", (1.0, 0.0, 0.0)), dtype=float
        )
        n = float(np.linalg.norm(direction[:2]))
        if n < 1e-9:
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = np.array([direction[0] / n, direction[1] / n, 0.0])
        sim.pushes.append(Push(impulse=impulse, direction=direction, time=now))
        msg = (
            f"[FAULT] push {impulse:.1f} N.s "
            f"dir=({direction[0]:+.1f},{direction[1]:+.1f}) at t={now:.2f}s"
        )
        return True, msg

    if fault.kind in ("actuator_scale", "actuator_dead"):
        resolved = resolve_actuator(
            sim.model, str(fault.params.get("actuator", "right_knee"))
        )
        if resolved is not None:
            released = release_actuator(sim.model, viz.backups, resolved[0])
            _forget_actuator_viz(viz, released)

    backup = apply_plant_fault(sim.model, fault)
    if backup is None:
        return False, f"[FAULT] failed to apply {fault.kind} params={fault.params}"
    viz.backups.append(backup)
    if backup.kind == "mass":
        delta = max(0.0, float(fault.params.get("delta_kg", 0.0)))
        body_id = int(backup.payload["id"])
        viz.mass_loads.append((body_id, delta))
        body_name = (
            mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            or fault.params.get("body", "?")
        )
        return True, f"[FAULT] +{delta:.1f} kg on {body_name} at t={now:.2f}s"
    if backup.kind == "friction":
        _record_friction_viz(sim, viz, backup, float(fault.params.get("mu", 0.25)))
    elif backup.kind == "actuator_dead":
        viz.dead_actuator = str(backup.payload.get("name", "?"))
        viz.actuator_scales.pop(viz.dead_actuator, None)
    elif backup.kind == "actuator_scale":
        name = str(backup.payload.get("name", "?"))
        scale = float(backup.payload.get("scale", 0.5))
        viz.actuator_scales[name] = scale
        if viz.dead_actuator == name:
            viz.dead_actuator = None
        return (
            True,
            f"[FAULT] {name} delivers {scale:.0%} of nominal torque "
            f"(gear {backup.payload['gear']:.3g} -> {backup.payload['applied_gear']:.3g}) "
            f"at t={now:.2f}s",
        )
    return True, f"[FAULT] {fault.kind} applied at t={now:.2f}s params={fault.params}"


def _record_friction_viz(sim, viz: FaultVizState, backup, mu: float) -> None:
    floor_id = int(backup.payload["id"])
    viz.floor_geom_id = floor_id
    if viz.floor_rgba_nominal is None:
        viz.floor_rgba_nominal = sim.model.geom_rgba[floor_id].copy()
    viz.friction_mu = float(mu)
    if mu < 0.6:
        sim.model.geom_rgba[floor_id] = np.array(
            [0.15, 0.75, 1.0, 1.0], dtype=np.float32
        )
    elif viz.floor_rgba_nominal is not None:
        sim.model.geom_rgba[floor_id] = viz.floor_rgba_nominal


def restore_faults(sim, viz: FaultVizState) -> str | None:
    """Undo mass, friction, and actuator edits. Pushes are left in place.

    Returns the log line, or None when there was nothing to clear.
    """
    if not viz.backups:
        return None
    clear_plant_faults(sim.model, viz.backups)
    if viz.floor_geom_id >= 0 and viz.floor_rgba_nominal is not None:
        sim.model.geom_rgba[viz.floor_geom_id] = viz.floor_rgba_nominal
    viz.backups.clear()
    viz.mass_loads.clear()
    viz.friction_mu = None
    viz.dead_actuator = None
    viz.actuator_scales.clear()
    return f"[FAULT] cleared plant mutations at t={sim.data.time:.2f}s"


class InteractivePlant:
    """Keyboard plant perturbations for a session that has no LOKA runtime.

    The dashboard uses this. ``run_loka`` goes through :class:`LokaRuntime`,
    which calls the same :func:`commit_fault` / :func:`fault_for_keycode`.
    """

    def __init__(self, sim) -> None:
        self.sim = sim
        self.viz = FaultVizState(external_body_id=sim.pelvis_id)
        self._keys = KeyDebounce()

    def inject(self, fault: FaultSpec) -> bool:
        ok, msg = commit_fault(self.sim, self.viz, fault)
        print(msg)
        return ok

    def clear(self) -> None:
        msg = restore_faults(self.sim, self.viz)
        print(msg or "[FAULT] nothing to clear")

    def handle_key(self, keycode: int) -> bool:
        """Apply the perturbation bound to ``keycode``. False if it is not one."""
        action = fault_for_keycode(keycode)
        if action is None:
            return False
        if not self._keys.allow(keycode):
            return True
        if action == "help":
            print(FAULT_HELP)
        elif action == "clear":
            self.clear()
        else:
            self.inject(action)
        return True
