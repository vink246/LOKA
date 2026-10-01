"""LOKA orchestration over the G1 MPC+WBC locomotion controller.

Dual-rate by construction: the plant runs at 500 Hz and never blocks, while
this package compresses telemetry, diagnoses in a background thread, and
applies typed parameter mutations whenever an answer arrives.

The ``*_stand_*`` names throughout still carry the Phase A framing, from
before the controller could walk. ``docs/orchestrator.md`` §19 lists what is
stale here and why -- the mission criteria, anomaly scoring and compressor all
still assume a robot that is meant to hold still.
"""

from loka.agent.apply import apply_stand_scratchpad
from loka.agent.compress import StandTelemetrySections, synthesize_stand_telemetry
from loka.agent.context import (
    build_stand_capabilities,
    build_stand_robot_context,
    format_stand_configuration,
)
from loka.agent.error_defaults import default_stand_error_spec
from loka.agent.faults import FaultSpec, apply_plant_fault, clear_plant_faults
from loka.agent.policy import LLM_STAND_CONTROLLER_ALLOWLIST

__all__ = [
    "StandTelemetrySections",
    "synthesize_stand_telemetry",
    "apply_stand_scratchpad",
    "build_stand_capabilities",
    "build_stand_robot_context",
    "default_stand_error_spec",
    "format_stand_configuration",
    "FaultSpec",
    "apply_plant_fault",
    "clear_plant_faults",
    "LLM_STAND_CONTROLLER_ALLOWLIST",
]
