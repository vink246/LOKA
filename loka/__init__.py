"""LOKA — LLM-Orchestrated Kinematic Adaptation.

Two halves:

* ``loka.control`` / ``loka.sim`` -- the G1 standing controller (centroidal MPC
  feeding a whole-body QP) and its MuJoCo harness.
* ``loka.orchestrator`` and friends -- the LLM layer that compresses telemetry
  into semantic tags and mutates the controller's parameters at runtime.
"""

__all__ = ["control", "evaluate", "sim"]
