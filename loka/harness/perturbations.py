"""Plant perturbations. Edits land on the simulator model, never the controller belief.

``prepare_model`` is a hook for disturbances that have to exist before the
first step (an obstacle geom, once vision is in the loop). The built-in
kinds do not use it.
"""

from __future__ import annotations

import numpy as np

from loka.agent.faults import FaultSpec
from loka.harness.config import PerturbationSpec
from loka.harness.registry import get_perturbation, register_perturbation


class Perturbation:
    kind = "none"

    def __init__(self, spec: PerturbationSpec) -> None:
        self.spec = spec
        self.params = dict(spec.params)

    def prepare_model(self, model) -> None:
        """Optional model edit applied once, before the episode starts."""
        return None

    def apply(self, agent) -> dict | None:
        """Mutate the plant. Return an overlay record, or None."""
        return None


@register_perturbation("none")
class NoPerturbation(Perturbation):
    kind = "none"


@register_perturbation("mass")
class MassPerturbation(Perturbation):
    kind = "mass"

    def __init__(self, spec: PerturbationSpec) -> None:
        super().__init__(spec)
        self.body = str(self.params.get("body", "torso_link"))
        self.delta_kg = float(self.params["delta_kg"])
        if self.delta_kg < 0:
            raise ValueError("mass delta_kg must be >= 0")

    def apply(self, agent) -> dict:
        agent.inject(FaultSpec("mass", 0.0, {"delta_kg": self.delta_kg, "body": self.body}))
        return {"kind": "mass", "body": self.body, "delta_kg": self.delta_kg}


@register_perturbation("actuator_gear")
class ActuatorGearPerturbation(Perturbation):
    """Scale an actuator gear. The controller's model is not told."""

    kind = "actuator_gear"

    def __init__(self, spec: PerturbationSpec) -> None:
        super().__init__(spec)
        self.actuator = str(self.params.get("actuator", "right_knee"))
        self.scale = float(self.params["scale"])
        if self.scale < 0:
            raise ValueError("actuator scale must be >= 0")

    def apply(self, agent) -> dict:
        agent.inject(
            FaultSpec(
                "actuator_scale",
                0.0,
                {"actuator": self.actuator, "scale": self.scale},
            )
        )
        return {
            "kind": "actuator_gear",
            "actuator": self.actuator,
            "scale": self.scale,
        }


@register_perturbation("friction")
class FrictionPerturbation(Perturbation):
    kind = "friction"

    def __init__(self, spec: PerturbationSpec) -> None:
        super().__init__(spec)
        self.mu = float(self.params["mu"])
        if self.mu < 0:
            raise ValueError("friction mu must be >= 0")

    def apply(self, agent) -> dict:
        agent.inject(FaultSpec("friction", 0.0, {"mu": self.mu}))
        return {"kind": "friction", "mu": self.mu}


@register_perturbation("push")
class PushPerturbation(Perturbation):
    kind = "push"

    def __init__(self, spec: PerturbationSpec) -> None:
        super().__init__(spec)
        self.force_n = float(self.params["force_n"])
        self.duration_s = float(self.params.get("duration_s", 0.05))
        direction = np.asarray(self.params.get("direction", (1.0, 0.0)), dtype=float)
        if direction.size == 2:
            direction = np.array([direction[0], direction[1], 0.0])
        if direction.size != 3:
            raise ValueError("push direction must have 2 or 3 components")
        flat = float(np.linalg.norm(direction[:2]))
        if flat < 1e-9:
            raise ValueError("push direction must have a horizontal component")
        self.direction = np.array([direction[0] / flat, direction[1] / flat, 0.0])
        if self.duration_s <= 0:
            raise ValueError("push duration_s must be positive")
        if self.force_n < 0:
            raise ValueError("push force_n must be >= 0")

    def apply(self, agent) -> dict:
        impulse = self.force_n * self.duration_s
        agent.inject(
            FaultSpec(
                "push",
                0.0,
                {
                    "impulse": impulse,
                    "direction": tuple(float(v) for v in self.direction),
                    "duration": self.duration_s,
                },
            )
        )
        force = (self.direction * self.force_n).tolist()
        return {
            "kind": "push",
            "force": [float(v) for v in force],
            "duration_s": self.duration_s,
            "body": "pelvis",
        }


def make_perturbation(spec: PerturbationSpec) -> Perturbation:
    return get_perturbation(spec.kind)(spec)


def make_push(force_n: float, direction, duration_s: float) -> PushPerturbation:
    """A push whose magnitude is chosen by the standing-force search."""
    spec = PerturbationSpec(
        name="push",
        kind="push",
        params={
            "force_n": float(force_n),
            "direction": list(direction),
            "duration_s": float(duration_s),
        },
    )
    return PushPerturbation(spec)
