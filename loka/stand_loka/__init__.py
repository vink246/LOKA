"""Phase A: LOKA orchestration over the G1 standing MPC+WBC controller.

This package is intentionally separate from the MJPC Walker stack in root
``main.py``. Same dual-rate pattern (compressor → async LLM → typed apply),
different plant.
"""

from loka.stand_loka.apply import apply_stand_scratchpad
from loka.stand_loka.compress import StandTelemetrySections, synthesize_stand_telemetry
from loka.stand_loka.context import (
    build_stand_capabilities,
    build_stand_robot_context,
    format_stand_configuration,
)
from loka.stand_loka.error_defaults import default_stand_error_spec
from loka.stand_loka.faults import FaultSpec, apply_plant_fault, clear_plant_faults
from loka.stand_loka.policy import LLM_STAND_CONTROLLER_ALLOWLIST

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
