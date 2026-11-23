"""Whole-body trajectory optimisation for terrain (optional).

A Crocoddyl-style stack for challenging terrain (curb, stairs, gap): a contact
sequence planned on the terrain belief, a whole-body optimal control problem
over that sequence solved by feasibility-driven DDP on a Pinocchio twin of the
belief model, and a receding-horizon controller that tracks it on the plant
with the solver's Riccati gains.

Needs ``pinocchio`` and ``crocoddyl``. When those are missing, or when
``TERRAIN_STACK`` is ``classical`` / ``heuristic``, nothing here runs.
``TERRAIN_STACK=auto`` engages this stack for any non-trivial raised belief.
See :mod:`loka.control.wbto.session`.
"""

from loka.control.wbto.session import (
    WbtoSession,
    crocoddyl_available,
    should_use_wbto,
    wbto_stack_requested,
)

__all__ = [
    "WbtoSession",
    "crocoddyl_available",
    "should_use_wbto",
    "wbto_stack_requested",
]
