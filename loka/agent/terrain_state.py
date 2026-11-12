"""Terrain part of the LOKA belief: the volumes the orchestrator declared.

``loka_state["volumes"]`` maps a volume name to its dict, and is the source
of truth. The controller's :class:`~loka.control.terrain.TerrainBelief` is
rebuilt from it whenever ``loka_state["terrain_version"]`` moves, the same
reset-then-apply rule the mass and friction edits follow.
"""

from __future__ import annotations

from loka.control.terrain import (
    MAX_VOLUMES,
    Volume,
    clamp_box,
    clamp_stairs_spec,
    stairs_volumes,
)

TERRAIN_OBJECT_TYPES = ("volume", "stairs")


def ensure_terrain_state(loka_state: dict) -> None:
    loka_state.setdefault("volumes", {})
    loka_state.setdefault("stairs", {})
    loka_state.setdefault("terrain_version", 0)


def _bump(loka_state: dict) -> None:
    loka_state["terrain_version"] = int(loka_state.get("terrain_version", 0)) + 1


def volumes_of(loka_state: dict) -> list[Volume]:
    return [Volume.from_dict(v) for v in (loka_state.get("volumes") or {}).values()]


def _drop_group(loka_state: dict, group: str) -> None:
    volumes = loka_state["volumes"]
    for name in [n for n, v in volumes.items() if v.get("group") == group]:
        del volumes[name]


def apply_terrain_mutation(loka_state: dict, mut: dict, sim_time: float | None = None) -> str:
    """Apply one ``volume`` or ``stairs`` Model_Mutations entry.

    Returns a one-line description of the edit. Raises ``ValueError`` or
    ``TypeError`` on an edit that cannot be applied; nothing is changed then.
    """
    ensure_terrain_state(loka_state)
    obj_type = str(mut.get("object_type", "")).strip().lower()
    name = str(mut.get("name") or "").strip()
    attr = str(mut.get("attribute") or "shape").strip().lower()
    value = mut.get("value")
    if not name:
        raise ValueError(f"{obj_type} mutation needs a name")
    volumes = loka_state["volumes"]

    if obj_type == "volume":
        if attr == "remove":
            if name not in volumes:
                raise ValueError(f"no believed volume named {name!r}")
            del volumes[name]
            _bump(loka_state)
            return f"volume '{name}' removed"
        if attr == "top" and not isinstance(value, dict):
            value = {"top": value}
        if attr not in ("shape", "box", "top"):
            raise ValueError(f"volume attribute must be shape, top or remove, got {attr!r}")
        previous = Volume.from_dict(volumes[name]) if name in volumes else None
        if previous is None and len(volumes) >= MAX_VOLUMES:
            raise ValueError(f"terrain belief is full ({MAX_VOLUMES} volumes)")
        box = clamp_box(value, previous)
        vol = Volume(name=name, source="llm", **box)
        if previous is not None and previous.group is not None:
            vol.group, vol.index = previous.group, previous.index
        entry = vol.to_dict()
        entry["applied_at"] = sim_time
        volumes[name] = entry
        _bump(loka_state)
        return (
            f"volume '{name}' centre {entry['center']} size {entry['size']} "
            f"top {entry['top']:.3f} m yaw {entry['yaw']:.2f}"
        )

    if obj_type == "stairs":
        if attr == "remove":
            if name not in loka_state["stairs"]:
                raise ValueError(f"no believed stairs named {name!r}")
            _drop_group(loka_state, name)
            del loka_state["stairs"][name]
            _bump(loka_state)
            return f"stairs '{name}' removed"
        spec = clamp_stairs_spec(value)
        spec["name"] = name
        vols = stairs_volumes(spec, source="llm")
        others = sum(1 for v in volumes.values() if v.get("group") != name)
        if others + len(vols) > MAX_VOLUMES:
            raise ValueError(f"terrain belief is full ({MAX_VOLUMES} volumes)")
        _drop_group(loka_state, name)
        for vol in vols:
            entry = vol.to_dict()
            entry["applied_at"] = sim_time
            volumes[vol.name] = entry
        loka_state["stairs"][name] = {**spec, "applied_at": sim_time}
        _bump(loka_state)
        riser = spec["riser"]
        riser_txt = (
            ", ".join(f"{r:.3f}" for r in riser) if isinstance(riser, list) else f"{riser:.3f}"
        )
        return (
            f"stairs '{name}' from ({spec['x']:+.2f}, {spec['y']:+.2f}) heading {spec['yaw']:+.2f}: "
            f"{spec['steps']} steps, riser {riser_txt} m, tread {spec['tread']:.2f} m, "
            f"width {spec['width']:.2f} m"
        )

    raise ValueError(f"not a terrain object_type: {obj_type!r}")


def seed_oracle_terrain(loka_state: dict, specs) -> int:
    """Fill the terrain belief with the plant's own flights. Returns the volume count."""
    ensure_terrain_state(loka_state)
    loka_state["volumes"] = {}
    loka_state["stairs"] = {}
    for spec in specs or []:
        for vol in stairs_volumes(spec, source="oracle"):
            loka_state["volumes"][vol.name] = vol.to_dict()
    _bump(loka_state)
    return len(loka_state["volumes"])


def sync_controller_terrain(controller, loka_state: dict) -> bool:
    """Rebuild the controller's terrain when the belief changed. True if it did."""
    terrain = getattr(controller, "terrain", None)
    if terrain is None:
        return False
    version = int(loka_state.get("terrain_version", 0))
    if terrain.version == version:
        return False
    terrain.set_volumes(volumes_of(loka_state), version=version)
    return True


def format_terrain_belief(loka_state: dict) -> str:
    """The believed volumes, one line each; empty when the belief is flat."""
    volumes = volumes_of(loka_state)
    if not volumes:
        return ""
    lines = ["Believed terrain (solid blocks from the floor up to top z):"]
    for vol in volumes:
        lines.append(
            f"  - volume '{vol.name}': centre ({vol.center[0]:+.2f}, {vol.center[1]:+.2f}) m, "
            f"depth {vol.size[0]:.2f} x width {vol.size[1]:.2f} m, yaw {vol.yaw:+.2f} rad, "
            f"top z={vol.top:.3f} m ({vol.source})"
        )
    return "\n".join(lines)
