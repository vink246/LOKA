"""Touchdown-driven terrain refinement: the robot measures the heights.

The orchestrator declares terrain structure (a flight of stairs here, about
this many steps of about this height). Each footfall then measures one tread:
the sole's resting height comes from leg kinematics, and the touchdown itself
from the joint-torque load estimate. When a footfall lands on a believed
volume at a different height, that volume's top is set to the measurement,
and, for a declared flight, every step above it moves by the same amount,
because a flight's treads are cumulative.

A footfall the belief cannot explain (load on a sole over believed flat floor,
or a swing blocked on the way) is not refined here. It is reported, and the
runtime treats it as a fault for the orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from loka.agent.terrain_state import ensure_terrain_state, volumes_of
from loka.control.terrain import TerrainBelief

#: Touchdown height error that is refined rather than ignored [m].
REFINE_TOL = 0.01
#: A footfall this far above the floor where the belief has no volume is
#: unexplained terrain [m].
UNEXPLAINED_HEIGHT = 0.02


@dataclass
class RefineResult:
    """What one footfall did to the belief."""

    kind: str  # "refined", "unexplained", or "consistent"
    note: str
    error_m: float = 0.0


def _bump(loka_state: dict) -> None:
    loka_state["terrain_version"] = int(loka_state.get("terrain_version", 0)) + 1


def refine_from_touchdown(loka_state: dict, event: dict, floor_z: float = 0.0) -> RefineResult:
    """Fold one ``touchdown`` or ``early_contact`` event into the belief."""
    ensure_terrain_state(loka_state)
    xy = np.asarray(event["xy"], dtype=float)[:2]
    measured = float(event["measured_z"])
    belief = TerrainBelief()
    belief.set_volumes(volumes_of(loka_state))
    vol = belief.volume_at(xy)
    if vol is None:
        error = measured - float(floor_z)
        if error > UNEXPLAINED_HEIGHT:
            return RefineResult(
                "unexplained",
                f"sole came to rest at z={measured:.3f} m at ({xy[0]:+.2f}, {xy[1]:+.2f}), "
                f"{error:+.3f} m above the believed floor, where no volume is believed",
                error,
            )
        return RefineResult("consistent", "", error)
    error = measured - float(vol.top)
    if abs(error) <= REFINE_TOL:
        return RefineResult("consistent", "", error)
    volumes = loka_state["volumes"]
    moved = [vol.name]
    entry = volumes[vol.name]
    entry["top"] = round(measured, 4)
    entry["source"] = "refined"
    if vol.group is not None:
        for name, other in volumes.items():
            if other.get("group") == vol.group and int(other.get("index", 0)) > vol.index:
                other["top"] = round(float(other["top"]) + error, 4)
                other["source"] = "refined"
                moved.append(name)
    _bump(loka_state)
    tail = f"; {len(moved) - 1} higher step(s) of {vol.group} moved with it" if len(moved) > 1 else ""
    return RefineResult(
        "refined",
        f"terrain refine: {vol.name} top {vol.top:.3f} -> {measured:.3f} m "
        f"(touchdown {error:+.3f} m){tail}",
        error,
    )
