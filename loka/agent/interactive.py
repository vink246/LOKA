"""Interactive plant-fault commands typed at a running simulation.

Rendering of the resulting pushes and masses lives in :mod:`loka.viz`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from loka.agent.faults import FaultSpec, _FaultBackup

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
  <any other text>               locked mission directive (LLM pursues it)

Viewer keys: 1=torso mass  2=ice  3=push  4=dead knee
             6=left shoulder mass  7=right shoulder mass  5=clear  H=help
"""
