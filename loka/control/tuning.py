"""The typed parameter surface shared by the GUI and the orchestrator LLM.

Everything a human slider or a language model is allowed to touch at runtime is
declared once here, with a range and a one-line rationale. Two properties make
this worth centralising rather than letting each caller poke at the config
dataclasses:

* **Names are unambiguous.** ``weight_force`` and ``friction_mu`` exist on both
  ``MPCConfig`` and ``WBCConfig`` with values three orders of magnitude apart,
  so a bare field name is not a safe address. Every entry here is a dotted
  path -- ``mpc.weight_force`` or ``wbc.weight_force`` -- and nothing else
  resolves.
* **Ranges are enforced.** A mutation arriving from a language model is
  clamped into a band that keeps the QPs well posed, so a hallucinated
  exponent degrades the stance instead of destroying the solver.

Applying an update never rebuilds an OSQP problem: the layers re-read their
weights each solve, and the few values baked into a matrix are refreshed in
place. A slider can therefore be dragged while the robot is standing on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class Tunable:
    """One runtime-mutable scalar."""

    path: str  # "wbc.kp_base_position"
    summary: str
    low: float
    high: float
    #: Which entries of a vector-valued field this scalar drives. ``None`` for
    #: a plain scalar field. Grouping roll+pitch or x+y behind one knob keeps
    #: the surface small and matches how the stance is actually symmetric.
    indices: tuple[int, ...] | None = None
    #: Slider (and LLM step) on a multiplicative scale. Weights span decades.
    log: bool = True
    #: Stacks whose ``update_weights`` accepts this path.
    stacks: tuple[str, ...] = ("legacy_dcm", "alip_footstep")

    @property
    def group(self) -> str:
        return self.path.split(".", 1)[0]

    @property
    def field(self) -> str:
        """The attribute this knob writes.

        Paths naming a slice of a vector carry a suffix that is part of the
        knob's identity but not of the attribute, so ``weight_position_xy``
        and ``weight_position_z`` both resolve to ``weight_position``.
        """
        name = self.path.split(".", 1)[1]
        if self.indices is not None:
            for suffix in ("_rp", "_yaw", "_xy", "_z"):
                if name.endswith(suffix):
                    return name[: -len(suffix)]
        return name


TUNABLES: tuple[Tunable, ...] = (
    # -- centroidal MPC: what the force plan cares about ------------------
    Tunable("mpc.weight_orientation_rp", "MPC: hold torso roll/pitch", 10.0, 5e3,
            indices=(0, 1)),
    Tunable("mpc.weight_orientation_yaw", "MPC: hold torso yaw", 10.0, 5e3,
            indices=(2,)),
    Tunable("mpc.weight_position_xy", "MPC: hold CoM over the feet", 10.0, 5e3,
            indices=(0, 1)),
    Tunable("mpc.weight_position_z", "MPC: hold stance height", 10.0, 2e4,
            indices=(2,)),
    Tunable("mpc.weight_angular_velocity", "MPC: damp torso rotation", 0.1, 500.0,
            indices=(0, 1, 2)),
    Tunable("mpc.weight_linear_velocity", "MPC: damp CoM drift", 1.0, 1e3,
            indices=(0, 1, 2)),
    Tunable("mpc.weight_force", "MPC: effort penalty; keep far below the state weights",
            1e-8, 1e-3),
    Tunable("mpc.friction_mu", "MPC: friction the force plan may assume",
            0.05, 1.2, log=False),
    # -- whole-body QP: how those forces become torques -------------------
    Tunable("wbc.weight_contact", "WBC: how hard to pin the planted feet", 100.0, 2e4),
    Tunable("wbc.weight_base_position", "WBC: priority of the CoM task", 0.1, 500.0),
    Tunable("wbc.weight_base_orientation", "WBC: priority of the torso task", 0.1, 1e3),
    Tunable("wbc.weight_posture_legs", "WBC: leg posture pull while planted", 1e-3, 10.0),
    Tunable("wbc.weight_posture_upper", "WBC: waist/arm posture pull", 0.1, 100.0),
    Tunable("wbc.weight_posture_swing", "WBC: leg posture pull while airborne", 0.1, 100.0),
    Tunable("wbc.weight_force", "WBC: deviation allowed from the MPC's forces", 1e-5, 1.0),
    Tunable("wbc.friction_mu", "WBC: friction the torque solve may assume",
            0.05, 1.2, log=False),
    Tunable("wbc.contact_kd", "WBC: damping on residual foot sliding", 0.0, 200.0,
            log=False),
    # -- task-space gains --------------------------------------------------
    Tunable("wbc.kp_base_position",
            "CoM stiffness. Above ~100 the demand exceeds what the soles can "
            "deliver and the response goes bang-bang", 1.0, 300.0, log=False),
    Tunable("wbc.kd_base_position", "CoM damping", 1.0, 100.0, log=False),
    Tunable("wbc.kp_base_orientation", "Torso stiffness", 10.0, 2e3, log=False),
    Tunable("wbc.kd_base_orientation", "Torso damping", 1.0, 200.0, log=False),
    Tunable("wbc.kp_posture_legs", "Planted-leg stiffness; 0 lets the legs bend freely",
            0.0, 200.0, log=False),
    Tunable("wbc.kd_posture_legs", "Planted-leg damping", 0.0, 50.0, log=False),
    Tunable("wbc.kp_posture_swing", "Airborne-leg stiffness", 0.0, 1e3, log=False),
    Tunable("wbc.kd_posture_swing", "Airborne-leg damping", 0.0, 100.0, log=False),
    Tunable("wbc.weight_swing_foot", "WBC: Cartesian swing-foot tracking weight. "
            "Must stay well above posture and in the same decade as contact "
            "or the QP will not lift the foot",
            1.0, 2e3),
    Tunable("wbc.kp_swing_foot", "Swing-foot Cartesian stiffness", 10.0, 2e3, log=False),
    Tunable("wbc.kd_swing_foot", "Swing-foot Cartesian damping", 1.0, 200.0, log=False),
    Tunable("wbc.weight_swing_orient", "WBC: late-swing sole-flat orientation weight",
            1.0, 500.0),
    Tunable("wbc.kp_swing_orient", "Swing-foot orientation stiffness", 10.0, 2e3, log=False),
    Tunable("wbc.kd_swing_orient", "Swing-foot orientation damping", 1.0, 200.0, log=False),
    Tunable("wbc.kp_posture_upper", "Waist/arm stiffness", 0.0, 1e3, log=False),
    Tunable("wbc.kd_posture_upper", "Waist/arm damping", 0.0, 100.0, log=False),
    # -- saturation --------------------------------------------------------
    Tunable("stand.max_linear_acc", "Cap on commanded CoM acceleration [m/s^2]",
            1.0, 100.0, log=False),
    Tunable("stand.max_angular_acc", "Cap on commanded torso acceleration [rad/s^2]",
            1.0, 200.0, log=False),
    Tunable("stand.max_joint_acc", "Cap on commanded joint acceleration [rad/s^2]",
            10.0, 2e3, log=False),
)


def paths_for_stack(stack: str) -> set[str]:
    """Dotted paths ``update_weights`` may write on ``stack``."""
    return {t.path for t in TUNABLES if stack in t.stacks}

BY_PATH: Mapping[str, Tunable] = {t.path: t for t in TUNABLES}


def _target(config: Any, tunable: Tunable) -> Any:
    """The dataclass owning ``tunable``. ``stand.*`` lives on the root."""
    return config if tunable.group == "stand" else getattr(config, tunable.group)


def read(config: Any, tunable: Tunable) -> float:
    value = getattr(_target(config, tunable), tunable.field)
    if tunable.indices is None:
        return float(value)
    return float(value[tunable.indices[0]])


def snapshot(config: Any) -> dict[str, float]:
    """Current value of every tunable, keyed by path."""
    return {t.path: read(config, t) for t in TUNABLES}


def catalogue() -> str:
    """Compact catalogue for an LLM prompt: one line per knob."""
    width = max(len(t.path) for t in TUNABLES)
    return "\n".join(
        f"{t.path:<{width}}  [{t.low:g}, {t.high:g}]  {t.summary}" for t in TUNABLES
    )


def clamp(path: str, value: float) -> float:
    tunable = BY_PATH.get(path)
    if tunable is None:
        raise KeyError(f"Unknown tunable {path!r}")
    return float(np.clip(value, tunable.low, tunable.high))


def apply(config: Any, updates: Mapping[str, float]) -> dict[str, float]:
    """Write clamped ``updates`` into ``config``; return what was applied.

    Vector-valued fields are copied to a list before the first write so a
    shared tuple default cannot be mutated out from under another config.
    """
    unknown = set(updates) - set(BY_PATH)
    if unknown:
        raise KeyError(f"Unknown tunables: {sorted(unknown)}")

    applied: dict[str, float] = {}
    for path, raw in updates.items():
        tunable = BY_PATH[path]
        value = clamp(path, float(raw))
        target = _target(config, tunable)
        if tunable.indices is None:
            setattr(target, tunable.field, value)
        else:
            current = list(getattr(target, tunable.field))
            for index in tunable.indices:
                current[index] = value
            setattr(target, tunable.field, current)
        applied[path] = value
    return applied


def paths(group: str | None = None) -> Iterable[str]:
    return [t.path for t in TUNABLES if group is None or t.group == group]
