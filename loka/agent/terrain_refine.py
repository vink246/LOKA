"""Touchdown-driven terrain refinement: the robot measures the heights.

The orchestrator declares terrain structure (a flight of stairs here, about
this many steps of about this height). Each footfall then measures one tread:
the sole's resting height comes from leg kinematics, and the touchdown itself
from the joint-torque load estimate.

When a footfall lands on a believed volume at a different height:

- a **uniform** stair flight (scalar ``riser`` in the declaration) has its
  riser rebuilt from the measurement: ``riser = (measured - floor) /
  (index + 1)``, and every tread top is rewritten as ``(i + 1) * riser``.
  Cascading an absolute delta would leave the wrong spacing when vision
  overestimated the riser itself (e.g. 0.13 m declared, 0.07 m true);
- an **irregular** flight (per-step riser list) or a lone volume still uses
  the absolute cascade: that tread's top becomes the measurement, and every
  higher step of the same group moves by the same amount.

Guardrails keep flaky contacts from warping a flight:
- only ``touchdown`` events refine (callers must not pass ``early_contact``);
- a footfall whose measured height matches a sibling tread more closely than
  the volume under the sole is left alone (wrong-step contact);
- irregular flights may move tops up freely, but may only move down by at
  most ``FLIGHT_DOWN_REFINE_MAX``;
- after a change, each riser must stay inside ``STAIR_RISER_RANGE``.

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
#: Within an irregular stair flight, allow lowering a tread by at most this
#: much [m]. Uniform flights rebuild the riser instead (see module docstring).
FLIGHT_DOWN_REFINE_MAX = 0.025
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


def _uniform_stairs_spec(loka_state: dict, group: str) -> dict | None:
    """Return the stairs declaration when it uses one scalar riser."""
    spec = (loka_state.get("stairs") or {}).get(group)
    if not isinstance(spec, dict):
        return None
    riser = spec.get("riser")
    if isinstance(riser, (list, tuple)):
        return None
    return spec


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

    volumes = loka_state["volumes"]
    uniform = _uniform_stairs_spec(loka_state, vol.group) if vol.group is not None else None
    if uniform is not None:
        return _refine_uniform_flight(
            loka_state, vol=vol, measured=measured, floor_z=float(floor_z), error=error
        )

    # Irregular flight or lone volume: absolute cascade.
    if vol.group is not None and error < -FLIGHT_DOWN_REFINE_MAX:
        return RefineResult(
            "skipped",
            f"terrain refine skipped: refusing to lower {vol.name} "
            f"({vol.top:.3f} -> {measured:.3f} m) inside flight {vol.group}",
            error,
        )

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
    old_top = float(vol.top)
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
        f"terrain refine: {vol.name} top {old_top:.3f} -> {measured:.3f} m "
        f"(touchdown {error:+.3f} m){tail}",
        error,
    )


def _refine_uniform_flight(
    loka_state: dict,
    *,
    vol: Volume,
    measured: float,
    floor_z: float,
    error: float,
) -> RefineResult:
    """Rebuild a scalar-riser flight from one measured tread height."""
    index = int(vol.index)
    denom = index + 1
    new_riser = (measured - floor_z) / float(denom)
    lo, hi = STAIR_RISER_RANGE
    if new_riser < lo or new_riser > hi:
        return RefineResult(
            "skipped",
            f"terrain refine skipped: uniform riser {new_riser:.3f} m from {vol.name} "
            f"is outside {STAIR_RISER_RANGE}",
            error,
        )
    members = _group_volumes(loka_state["volumes"], vol.group)
    tops = [round((i + 1) * new_riser, 4) for i in range(len(members))]
    if not _riser_spacings_ok(tops):
        return RefineResult(
            "skipped",
            f"terrain refine skipped: uniform riser {new_riser:.3f} m from {vol.name} "
            f"would leave risers outside {STAIR_RISER_RANGE}",
            error,
        )
    if all(abs(float(entry["top"]) - top) <= REFINE_TOL for (_, entry), top in zip(members, tops)):
        return RefineResult("consistent", "", error)

    old_top = float(vol.top)
    for (name, entry), top in zip(members, tops):
        entry["top"] = top
        entry["source"] = "refined"
    spec = loka_state["stairs"][vol.group]
    old_riser = float(spec.get("riser", new_riser))
    spec["riser"] = round(new_riser, 4)
    _bump(loka_state)
    return RefineResult(
        "refined",
        f"terrain refine: {vol.group} uniform riser {old_riser:.3f} -> {new_riser:.3f} m "
        f"from {vol.name} touchdown ({old_top:.3f} -> {measured:.3f} m); "
        f"{len(members)} step(s) rebuilt",
        error,
    )
