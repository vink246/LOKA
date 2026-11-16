"""Touchdown-driven terrain refinement: the robot measures the heights.

The orchestrator declares terrain structure (a flight of stairs here, about
this many steps of about this height). Each footfall then measures one tread:
the sole's resting height comes from leg kinematics, and the touchdown itself
from the joint-torque load estimate. When a footfall lands on a believed
volume at a different height, that volume's top is set to the measurement,
and, for a declared flight, every step above it moves by the same amount,
because a flight's treads are cumulative.

Guardrails keep flaky contacts from warping a flight:
- only ``touchdown`` events refine (callers must not pass ``early_contact``);
- a footfall whose measured height matches a sibling tread more closely than
  the volume under the sole is left alone (wrong-step contact);
- within a flight, tops only move up (a lower sole is treated as the wrong
  tread, not a shorter riser);
- after a cascade, each riser must stay inside ``STAIR_RISER_RANGE``.

A footfall the belief cannot explain (load on a sole over believed flat floor,
or a swing blocked on the way) is not refined here. It is reported, and the
runtime treats it as a fault for the orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from loka.agent.terrain_state import ensure_terrain_state, volumes_of
from loka.control.terrain import STAIR_RISER_RANGE, TerrainBelief, Volume

#: Touchdown height error that is refined rather than ignored [m].
REFINE_TOL = 0.01
#: A footfall this far above the floor where the belief has no volume is
#: unexplained terrain [m].
UNEXPLAINED_HEIGHT = 0.02


@dataclass
class RefineResult:
    """What one footfall did to the belief."""

    kind: str  # "refined", "unexplained", "consistent", or "skipped"
    note: str
    error_m: float = 0.0


def _bump(loka_state: dict) -> None:
    loka_state["terrain_version"] = int(loka_state.get("terrain_version", 0)) + 1


def _group_volumes(volumes: dict, group: str) -> list[tuple[str, dict]]:
    members = [
        (name, entry)
        for name, entry in volumes.items()
        if entry.get("group") == group
    ]
    members.sort(key=lambda item: int(item[1].get("index", 0)))
    return members


def _riser_spacings_ok(tops: list[float]) -> bool:
    """True when consecutive tops climb by a legal stair riser."""
    lo, hi = STAIR_RISER_RANGE
    for below, above in zip(tops, tops[1:]):
        rise = float(above) - float(below)
        if rise < lo or rise > hi:
            return False
    return True


def _better_sibling(vol: Volume, measured: float, belief: TerrainBelief) -> Volume | None:
    """Return a same-flight volume whose top is closer to ``measured`` than ``vol``."""
    if vol.group is None:
        return None
    best = vol
    best_err = abs(float(vol.top) - measured)
    for other in belief.volumes:
        if other.group != vol.group or other.name == vol.name:
            continue
        err = abs(float(other.top) - measured)
        if err + 1e-9 < best_err:
            best = other
            best_err = err
    return None if best.name == vol.name else best


def refine_from_touchdown(loka_state: dict, event: dict, floor_z: float = 0.0) -> RefineResult:
    """Fold one ``touchdown`` event into the belief."""
    if event.get("kind") not in (None, "touchdown"):
        return RefineResult(
            "skipped",
            f"terrain refine skipped: refusing kind={event.get('kind')!r}",
            0.0,
        )
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

    sibling = _better_sibling(vol, measured, belief)
    if sibling is not None:
        return RefineResult(
            "skipped",
            f"terrain refine skipped: measured z={measured:.3f} m matches {sibling.name} "
            f"(top {sibling.top:.3f}) more closely than {vol.name} (top {vol.top:.3f})",
            measured - float(vol.top),
        )

    error = measured - float(vol.top)
    if abs(error) <= REFINE_TOL:
        return RefineResult("consistent", "", error)

    # Within a flight, only raise tops. A sole that sits low is almost always
    # on a lower tread (or the face), not evidence that every higher riser
    # should shrink.
    if vol.group is not None and error < -REFINE_TOL:
        return RefineResult(
            "skipped",
            f"terrain refine skipped: refusing to lower {vol.name} "
            f"({vol.top:.3f} -> {measured:.3f} m) inside flight {vol.group}",
            error,
        )

    volumes = loka_state["volumes"]
    proposed: dict[str, float] = {vol.name: round(measured, 4)}
    if vol.group is not None:
        for name, other in volumes.items():
            if other.get("group") == vol.group and int(other.get("index", 0)) > vol.index:
                proposed[name] = round(float(other["top"]) + error, 4)
        group_tops = [
            proposed.get(name, float(entry["top"]))
            for name, entry in _group_volumes(volumes, vol.group)
        ]
        if not _riser_spacings_ok(group_tops):
            return RefineResult(
                "skipped",
                f"terrain refine skipped: cascading {error:+.3f} m from {vol.name} "
                f"would leave risers outside {STAIR_RISER_RANGE}",
                error,
            )

    moved = [vol.name]
    entry = volumes[vol.name]
    entry["top"] = proposed[vol.name]
    entry["source"] = "refined"
    if vol.group is not None:
        for name, other in volumes.items():
            if name in proposed and name != vol.name:
                other["top"] = proposed[name]
                other["source"] = "refined"
                moved.append(name)
    _bump(loka_state)
    tail = (
        f"; {len(moved) - 1} higher step(s) of {vol.group} moved with it"
        if len(moved) > 1
        else ""
    )
    return RefineResult(
        "refined",
        f"terrain refine: {vol.name} top {vol.top:.3f} -> {measured:.3f} m "
        f"(touchdown {error:+.3f} m){tail}",
        error,
    )
