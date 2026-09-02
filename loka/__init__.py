"""LOKA — LLM-Orchestrated Kinematic Adaptation.

Two halves:

* ``loka.control`` / ``loka.sim`` -- the G1 locomotion controller (centroidal
  MPC feeding a whole-body QP, with a gait layer on top) and its MuJoCo
  harness. Nothing here waits on a language model.
* ``loka.agent`` -- the slow layer: compress proprioceptive anomalies into
  semantic tags, diagnose them with an LLM, and mutate the controller's typed
  parameters between control ticks.
"""

__all__ = ["agent", "control", "sim"]
