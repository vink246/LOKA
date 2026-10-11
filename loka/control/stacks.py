"""Selectable locomotion stacks.

``legacy_dcm`` is the convex-MPC + DCM foothold plant. ``alip_footstep``
keeps that plant and replaces only the foothold law with the ALIP stepping
correction (Xiong and Ames) on the gait's own CoM reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from loka.agent.policy import LLM_STAND_CONTROLLER_ALLOWLIST
from loka.control.footholds import AlipStepFootholdPolicy
from loka.control.locomotion import LocomotionConfig, LocomotionController
from loka.control.tuning import paths_for_stack

STACK_NAMES = ("legacy_dcm", "alip_footstep")


@dataclass(frozen=True)
class StackSpec:
    name: str
    description: str
    factory: Callable[[LocomotionConfig], LocomotionController]
    tunable_paths: frozenset[str]
    llm_allowlist: frozenset[str]
    #: Knob names whose prompt text differs from the DCM catalogue.
    gait_knob_overrides: tuple[str, ...] = ()


def _legacy(config: LocomotionConfig) -> LocomotionController:
    config.stack = "legacy_dcm"
    return LocomotionController(config)


def _footstep(config: LocomotionConfig) -> LocomotionController:
    config.stack = "alip_footstep"
    # Nominal chain footholds plus ALIP feedback on the pre-impact error,
    # with the MPC on the DCM CoM reference it was tuned against.
    return LocomotionController(config, foothold_policy=AlipStepFootholdPolicy())


SPECS: dict[str, StackSpec] = {
    "legacy_dcm": StackSpec(
        name="legacy_dcm",
        description=(
            "The plant is a centroidal convex MPC over a whole-body QP. "
            "Footholds come from a DCM capture-point law. Friction while "
            "walking is mpc.friction_mu; the WBC copies it."
        ),
        factory=_legacy,
        tunable_paths=frozenset(paths_for_stack("legacy_dcm")),
        llm_allowlist=LLM_STAND_CONTROLLER_ALLOWLIST,
        gait_knob_overrides=(),
    ),
    "alip_footstep": StackSpec(
        name="alip_footstep",
        description=(
            "The plant is the same centroidal convex MPC and whole-body QP. "
            "Footholds come from the ALIP stepping law (Xiong and Ames): "
            "the state is horizontal CoM offset from the stance foot and "
            "angular momentum about that foot. Friction while walking is "
            "mpc.friction_mu."
        ),
        factory=_footstep,
        tunable_paths=frozenset(paths_for_stack("alip_footstep")),
        llm_allowlist=LLM_STAND_CONTROLLER_ALLOWLIST,
        gait_knob_overrides=("gait.capture_gain", "gait.foothold_retarget_s", "gait.turn_rate"),
    ),
}


def make(config: LocomotionConfig) -> LocomotionController:
    name = getattr(config, "stack", None) or "legacy_dcm"
    if name not in SPECS:
        raise ValueError(f"Unknown locomotion stack {name!r}; choose from {STACK_NAMES}")
    return SPECS[name].factory(config)


def add_stack_argument(parser, *, allow_all: bool = False) -> None:
    choices = list(STACK_NAMES) + (["all"] if allow_all else [])
    parser.add_argument(
        "--stack",
        choices=choices,
        default=None,
        help="Locomotion stack. Default is legacy_dcm (or the config file).",
    )


def apply_stack_arg(config: LocomotionConfig, args) -> LocomotionConfig:
    name = getattr(args, "stack", None)
    if name and name != "all":
        config.stack = name
    return config
