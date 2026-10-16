"""LOKA orchestration over the G1 MPC+WBC locomotion controller.

Dual-rate by construction: the plant runs at 500 Hz and never blocks, while
this package compresses telemetry, diagnoses in a background thread, and
applies typed parameter mutations whenever an answer arrives.

The ``*_stand_*`` names throughout still carry the Phase A framing, from
before the controller could walk.
"""

from loka.agent.apply import apply_stand_scratchpad
from loka.agent.compress import synthesize_stand_telemetry
from loka.agent.context import (
    build_stand_robot_context,
    format_stand_configuration,
)
from loka.agent.faults import FaultSpec, apply_plant_fault, clear_plant_faults

__all__ = [
    "synthesize_stand_telemetry",
    "apply_stand_scratchpad",
    "build_stand_robot_context",
    "format_stand_configuration",
    "FaultSpec",
    "apply_plant_fault",
    "clear_plant_faults",
]
