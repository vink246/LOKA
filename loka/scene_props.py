"""Plant model with props and the head camera, built with ``MjSpec``.

The props come from ``models/props/*.xml`` (transcribed from HumanoidBench).
Each prop file is a standalone MJCF with HumanoidBench's ``visual`` /
``collision`` default classes and one body. :func:`build_plant_model` copies
those geoms into a pool of slots added after the robot, so the robot's DOFs
stay the prefix ``qpos[:robot_nq]`` / ``qvel[:robot_nv]``.

Dynamic slots get a free joint and are parked far off-scene until they are
spawned. A parked prop has its collisions off and full gravity compensation,
so it floats in place and costs the solver nothing. Spawning teleports the free
joint, zeroes its velocity, and restores collisions and gravity, so the model
never recompiles and the viewer keeps its handle. Static kinds (the shelf)
have no joint; placing one edits ``model.body_pos``.

Only the plant gets this. The controller's belief model is loaded from the
plain MJCF in :class:`loka.control.robot.G1Model` and never sees a prop.
"""

from __future__ import annotations

import math
import re
import threading
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

PROPS_DIR = Path(__file__).resolve().parents[1] / "models" / "props"
PROP_KINDS = ("chair", "table", "box", "cube", "shelf")
#: Named prop layouts for ``run_loka --scene`` and ``scene save/load``.
SCENES_DIR = Path(__file__).resolve().parent / "config" / "scenes"
DEFAULT_STATIC_KINDS = ("shelf",)
#: Parked slots rest on the floor here, beyond the renderer's far plane.
PARK_X0 = -100.0
PARK_SPACING = 3.0
#: Clearance above the floor when a dynamic prop is spawned [m].
SPAWN_CLEARANCE = 0.002
#: A spawn closer than this to the pelvis is reported as overlapping the robot.
ROBOT_CLEARANCE = 0.45

_GEOM_TYPES = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
}
_SLOT_NAME = re.compile(r"^(" + "|".join(PROP_KINDS) + r")_(\d+)$")


# -- prop files ---------------------------------------------------------------


@dataclass(frozen=True)
class PropDef:
    kind: str
    textures: dict[str, Path]
    materials: dict[str, dict[str, str]]
    geoms: tuple[dict[str, str], ...]


def _collect_defaults(node: ET.Element, inherited: dict[str, str], out: dict) -> None:
    for child in node.findall("default"):
        attrs = dict(inherited)
        geom = child.find("geom")
        if geom is not None:
            attrs.update(geom.attrib)
        name = child.get("class")
        if name:
            out[name] = attrs
        _collect_defaults(child, attrs, out)


@lru_cache(maxsize=None)
def load_prop_def(kind: str) -> PropDef:
    """Parse ``models/props/<kind>.xml`` into geoms with their classes resolved."""
    path = PROPS_DIR / f"{kind}.xml"
    if not path.is_file():
        raise FileNotFoundError(f"no prop file for {kind!r}: {path}")
    root = ET.parse(path).getroot()
    defaults: dict[str, dict[str, str]] = {}
    for top in root.findall("default"):
        base = dict(top.find("geom").attrib) if top.find("geom") is not None else {}
        _collect_defaults(top, base, defaults)
    textures: dict[str, Path] = {}
    materials: dict[str, dict[str, str]] = {}
    for asset in root.findall("asset"):
        for tex in asset.findall("texture"):
            textures[tex.get("name")] = (path.parent / tex.get("file")).resolve()
        for mat in asset.findall("material"):
            materials[mat.get("name")] = dict(mat.attrib)
    bodies = root.find("worldbody").findall("body")
    if len(bodies) != 1:
        raise ValueError(f"{path}: expected one body, found {len(bodies)}")
    geoms = []
    for geom in bodies[0].findall("geom"):
        attrs = dict(defaults.get(geom.get("class", ""), {}))
        attrs.update(geom.attrib)
        attrs.pop("class", None)
        geoms.append(attrs)
    return PropDef(kind, textures, materials, tuple(geoms))


def _floats(text: str) -> list[float]:
    return [float(v) for v in text.split()]


def _asset_name(kind: str, name: str) -> str:
    return f"prop_{kind}_{name}"


def _add_prop_assets(spec, prop: PropDef) -> None:
    for name, file in prop.textures.items():
        tex = spec.add_texture()
        tex.name = _asset_name(prop.kind, name)
        tex.type = mujoco.mjtTexture.mjTEXTURE_2D
        tex.nchannel = 3
        tex.file = str(file)
    for name, attrs in prop.materials.items():
        mat = spec.add_material()
        mat.name = _asset_name(prop.kind, name)
        if "texture" in attrs:
            textures = list(mat.textures)
            textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = _asset_name(prop.kind, attrs["texture"])
            mat.textures = textures
        if "texuniform" in attrs:
            mat.texuniform = attrs["texuniform"].lower() == "true"
        if "texrepeat" in attrs:
            mat.texrepeat = _floats(attrs["texrepeat"])
        if "rgba" in attrs:
            mat.rgba = _floats(attrs["rgba"])


def _add_prop_geom(body, kind: str, slot: str, attrs: dict[str, str]) -> None:
    geom = body.add_geom()
    if "name" in attrs:
        geom.name = f"{slot}/{attrs['name']}"
    geom.type = _GEOM_TYPES[attrs.get("type", "sphere")]
    size = _floats(attrs["size"])
    geom.size = (size + [0.0, 0.0, 0.0])[:3]
    for key in ("pos", "quat", "rgba", "friction"):
        if key in attrs:
            setattr(geom, key, _floats(attrs[key]))
    for key in ("group", "contype", "conaffinity"):
        if key in attrs:
            setattr(geom, key, int(attrs[key]))
    if "density" in attrs:
        geom.density = float(attrs["density"])
    if "mass" in attrs:
        geom.mass = float(attrs["mass"])
    if "material" in attrs:
        geom.material = _asset_name(kind, attrs["material"])


# -- plant model --------------------------------------------------------------


def plant_recipe(
    model_path: str | Path,
    prop_slots: dict[str, int] | None = None,
    static_kinds=DEFAULT_STATIC_KINDS,
    camera: dict | None = None,
) -> dict:
    """Everything needed to rebuild the identical plant in another process."""
    slots = {str(k): int(v) for k, v in (prop_slots or {}).items() if int(v) > 0}
    for kind in slots:
        if kind not in PROP_KINDS:
            raise ValueError(f"unknown prop kind {kind!r}; known: {', '.join(PROP_KINDS)}")
    return {
        "model_path": str(model_path),
        "prop_slots": slots,
        "static_kinds": [str(k) for k in static_kinds],
        "camera": dict(camera) if camera else None,
    }


def recipe_is_plain(recipe: dict | None) -> bool:
    return not recipe or (not recipe.get("prop_slots") and not recipe.get("camera"))


@dataclass
class PropSlot:
    name: str
    kind: str
    static: bool
    body_id: int
    qpos_adr: int  # -1 for a static slot
    dof_adr: int
    park: tuple[float, float, float]
    #: Collision geoms and their compiled contype / conaffinity.
    collision_geoms: tuple[tuple[int, int, int], ...] = ()
    active: bool = False


@dataclass
class PropLayout:
    slots: list[PropSlot] = field(default_factory=list)
    robot_nq: int = 0
    robot_nv: int = 0
    robot_root: int = -1

    @property
    def prop_body_ids(self) -> dict[int, PropSlot]:
        return {slot.body_id: slot for slot in self.slots}

    def by_name(self, name: str) -> PropSlot | None:
        for slot in self.slots:
            if slot.name == name:
                return slot
        return None


def _layout(model, names: list[tuple[str, str, bool, tuple]]) -> PropLayout:
    slots = []
    for name, kind, static, park in names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        qpos_adr = dof_adr = -1
        if not static:
            joint = model.body_jntadr[body_id]
            qpos_adr = int(model.jnt_qposadr[joint])
            dof_adr = int(model.jnt_dofadr[joint])
        collision = tuple(
            (g, int(model.geom_contype[g]), int(model.geom_conaffinity[g]))
            for g in range(model.ngeom)
            if int(model.geom_bodyid[g]) == body_id
            and (model.geom_contype[g] or model.geom_conaffinity[g])
        )
        slots.append(
            PropSlot(name, kind, static, int(body_id), qpos_adr, dof_adr, park, collision)
        )
    dynamic = [slot for slot in slots if not slot.static]
    robot_nq = min((s.qpos_adr for s in dynamic), default=int(model.nq))
    robot_nv = min((s.dof_adr for s in dynamic), default=int(model.nv))
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    root = int(model.body_rootid[pelvis]) if pelvis >= 0 else -1
    return PropLayout(slots, int(robot_nq), int(robot_nv), root)


def build_plant_model(recipe: dict):
    """Compile the plant described by :func:`plant_recipe`. Returns (model, layout)."""
    if recipe_is_plain(recipe):
        model = mujoco.MjModel.from_xml_path(recipe["model_path"])
        return model, _layout(model, [])

    from loka.vision import EgoCameraSpec, add_ego_camera, fit_offscreen_buffer

    spec = mujoco.MjSpec()
    spec.from_file(recipe["model_path"])
    camera = EgoCameraSpec.from_config(recipe["camera"]) if recipe.get("camera") else None
    if camera is not None:
        add_ego_camera(spec, camera)

    static_kinds = set(recipe.get("static_kinds") or ())
    names: list[tuple[str, str, bool, tuple]] = []
    park_qpos: list[float] = []
    index = 0
    for kind, count in recipe.get("prop_slots", {}).items():
        prop = load_prop_def(kind)
        _add_prop_assets(spec, prop)
        static = kind in static_kinds
        for i in range(count):
            name = f"{kind}_{i}"
            park = (PARK_X0 - PARK_SPACING * index, 0.0, 0.0)
            body = spec.worldbody.add_body()
            body.name = name
            body.pos = list(park)
            if not static:
                joint = body.add_freejoint()
                joint.name = f"{name}/free"
                # Compiled on so MuJoCo keeps the gravcomp pass; the manager
                # turns it off when the prop is spawned.
                body.gravcomp = 1.0
                park_qpos.extend([*park, 1.0, 0.0, 0.0, 0.0])
            for attrs in prop.geoms:
                _add_prop_geom(body, kind, name, attrs)
            names.append((name, kind, static, park))
            index += 1
    for key in spec.key:
        key.qpos = list(key.qpos) + park_qpos
    model = spec.compile()
    if camera is not None:
        fit_offscreen_buffer(model, camera)
    return model, _layout(model, names)


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def quat_yaw(quat) -> float:
    w, x, y, z = (float(v) for v in quat)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def plant_snapshot(model, data, layout: PropLayout | None) -> dict:
    """Copy of everything a renderer needs to reproduce this instant."""
    static = {}
    if layout is not None:
        for slot in layout.slots:
            if slot.static:
                static[slot.body_id] = (
                    model.body_pos[slot.body_id].copy(),
                    model.body_quat[slot.body_id].copy(),
                )
    return {"time": float(data.time), "qpos": data.qpos.copy(), "static": static}


def apply_snapshot(model, data, snapshot: dict) -> None:
    data.qpos[:] = snapshot["qpos"]
    for body_id, (pos, quat) in snapshot.get("static", {}).items():
        model.body_pos[int(body_id)] = pos
        model.body_quat[int(body_id)] = quat
    data.time = float(snapshot.get("time", 0.0))
    mujoco.mj_forward(model, data)


def prop_contacts(model, data, layout: PropLayout | None) -> list[dict]:
    """Robot-to-prop contacts this step: prop, summed normal force, robot links."""
    if layout is None or not layout.slots:
        return []
    props = layout.prop_body_ids
    touching: dict[str, dict] = {}
    force = np.zeros(6)
    for i in range(data.ncon):
        con = data.contact[i]
        b1 = int(model.geom_bodyid[con.geom1])
        b2 = int(model.geom_bodyid[con.geom2])
        if b1 in props and int(model.body_rootid[b2]) == layout.robot_root:
            slot, link = props[b1], b2
        elif b2 in props and int(model.body_rootid[b1]) == layout.robot_root:
            slot, link = props[b2], b1
        else:
            continue
        mujoco.mj_contactForce(model, data, i, force)
        entry = touching.setdefault(
            slot.name, {"prop": slot.name, "kind": slot.kind, "force_n": 0.0, "links": set()}
        )
        entry["force_n"] += abs(float(force[0]))
        entry["links"].add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, link) or str(link))
    out = []
    for entry in touching.values():
        entry["links"] = sorted(entry["links"])
        out.append(entry)
    return sorted(out, key=lambda e: -e["force_n"])


# -- runtime prop placement ---------------------------------------------------


class PropManager:
    """Spawn, move, and remove props on a live plant.

    Call the methods directly from the simulation thread. Other threads (the
    stdin reader) use :meth:`submit`; :meth:`apply_pending` runs those on the
    simulation thread between control steps.
    """

    def __init__(self, sim) -> None:
        self.sim = sim
        self.layout: PropLayout = sim.prop_layout
        self._pending: deque = deque()
        self._lock = threading.Lock()

    @property
    def kinds(self) -> list[str]:
        return sorted({slot.kind for slot in self.layout.slots})

    def reset(self) -> None:
        """Park every slot. The keyframe already put dynamic slots at their park."""
        for slot in self.layout.slots:
            self._park(slot)

    # -- cross-thread ---------------------------------------------------------

    def submit(self, op: str, *args) -> None:
        with self._lock:
            self._pending.append((op, args))

    def apply_pending(self) -> list[str]:
        with self._lock:
            if not self._pending:
                return []
            ops = list(self._pending)
            self._pending.clear()
        messages = []
        for op, args in ops:
            try:
                result = getattr(self, op)(*args)
            except (ValueError, KeyError, OSError) as exc:
                result = f"[PROPS] {exc}"
            if isinstance(result, list):
                messages.extend(result)
            elif result:
                messages.append(result)
        return messages

    # -- placement ------------------------------------------------------------

    def _slot(self, name: str) -> PropSlot:
        slot = self.layout.by_name(name)
        if slot is None:
            raise ValueError(f"no prop named {name!r}")
        return slot

    def _place_static(self, slot: PropSlot, x: float, y: float, yaw: float) -> None:
        model = self.sim.model
        model.body_pos[slot.body_id] = (x, y, 0.0)
        model.body_quat[slot.body_id] = yaw_quat(yaw)

    def _set_live(self, slot: PropSlot, live: bool) -> None:
        """Collisions and gravity on for a placed prop, off while parked."""
        model = self.sim.model
        for geom, contype, conaffinity in slot.collision_geoms:
            model.geom_contype[geom] = contype if live else 0
            model.geom_conaffinity[geom] = conaffinity if live else 0
        if not slot.static:
            model.body_gravcomp[slot.body_id] = 0.0 if live else 1.0

    def _place(self, slot: PropSlot, x: float, y: float, yaw: float) -> None:
        model, data = self.sim.model, self.sim.data
        if slot.static:
            self._place_static(slot, x, y, yaw)
        else:
            data.qpos[slot.qpos_adr:slot.qpos_adr + 3] = (x, y, SPAWN_CLEARANCE)
            data.qpos[slot.qpos_adr + 3:slot.qpos_adr + 7] = yaw_quat(yaw)
            data.qvel[slot.dof_adr:slot.dof_adr + 6] = 0.0
        mujoco.mj_forward(model, data)

    def _robot_note(self, x: float, y: float) -> str:
        pelvis = np.asarray(self.sim.data.qpos[:2], dtype=float)
        dist = float(np.hypot(x - pelvis[0], y - pelvis[1]))
        if dist < ROBOT_CLEARANCE:
            return f" (warning: {dist:.2f} m from the pelvis, overlapping the robot)"
        return ""

    def spawn(self, kind: str, x: float, y: float, yaw: float = 0.0) -> str:
        if kind not in self.kinds:
            raise ValueError(f"no {kind!r} slots; available kinds: {', '.join(self.kinds)}")
        free = [s for s in self.layout.slots if s.kind == kind and not s.active]
        if not free:
            total = sum(1 for s in self.layout.slots if s.kind == kind)
            raise ValueError(f"all {total} {kind} slots are in use; remove one first")
        slot = free[0]
        self._set_live(slot, True)
        self._place(slot, float(x), float(y), float(yaw))
        slot.active = True
        return (
            f"[PROPS] spawned {slot.name} at ({x:+.2f}, {y:+.2f}) yaw {yaw:+.2f} rad"
            f"{self._robot_note(float(x), float(y))}"
        )

    def move(self, name: str, x: float, y: float, yaw: float = 0.0) -> str:
        slot = self._slot(name)
        self._set_live(slot, True)
        self._place(slot, float(x), float(y), float(yaw))
        slot.active = True
        return (
            f"[PROPS] moved {slot.name} to ({x:+.2f}, {y:+.2f}) yaw {yaw:+.2f} rad"
            f"{self._robot_note(float(x), float(y))}"
        )

    def remove(self, name: str) -> str:
        slot = self._slot(name)
        if not slot.active:
            return f"[PROPS] {name} is not placed"
        self._park(slot)
        return f"[PROPS] removed {name}"

    def _park(self, slot: PropSlot) -> None:
        self._set_live(slot, False)
        self._place(slot, slot.park[0], slot.park[1], 0.0)
        slot.active = False

    def clear(self) -> str:
        placed = [s for s in self.layout.slots if s.active]
        for slot in placed:
            self._park(slot)
        return f"[PROPS] cleared {len(placed)} prop(s)" if placed else "[PROPS] nothing placed"

    def pose(self, slot: PropSlot) -> tuple[float, float, float, float]:
        """Live x, y, z, yaw. A dynamic prop may have been pushed since it spawned."""
        if slot.static:
            pos = self.sim.model.body_pos[slot.body_id]
            quat = self.sim.model.body_quat[slot.body_id]
        else:
            pos = self.sim.data.qpos[slot.qpos_adr:slot.qpos_adr + 3]
            quat = self.sim.data.qpos[slot.qpos_adr + 3:slot.qpos_adr + 7]
        return float(pos[0]), float(pos[1]), float(pos[2]), quat_yaw(quat)

    def save_scene(self, name: str) -> str:
        placed = []
        for slot in self.layout.slots:
            if slot.active:
                x, y, _, yaw = self.pose(slot)
                placed.append((slot.kind, x, y, yaw))
        if not placed:
            return "[PROPS] nothing placed; scene not saved"
        path = save_scene(name, placed)
        return f"[PROPS] saved {len(placed)} prop(s) to {path}"

    def load_scene(self, name: str) -> list[str]:
        spawns = load_scene(name)
        lines = [self.clear()]
        for kind, x, y, yaw in spawns:
            try:
                lines.append(self.spawn(kind, x, y, yaw))
            except ValueError as exc:
                lines.append(f"[PROPS] {exc}")
        return lines

    def scenes(self) -> str:
        names = list_scenes()
        return "[PROPS] scenes: " + (", ".join(names) if names else f"none in {SCENES_DIR}")

    def list(self) -> list[str]:
        lines = []
        for slot in self.layout.slots:
            if not slot.active:
                continue
            x, y, z, yaw = self.pose(slot)
            kind = "static" if slot.static else "free"
            lines.append(
                f"[PROPS] {slot.name:8s} {kind:6s} xy ({x:+.2f}, {y:+.2f}) z {z:+.2f} yaw {yaw:+.2f}"
            )
        free = {
            kind: sum(1 for s in self.layout.slots if s.kind == kind and not s.active)
            for kind in self.kinds
        }
        lines.append(
            "[PROPS] free slots: " + ", ".join(f"{k}={n}" for k, n in free.items())
        )
        return lines


def parse_props_spec(text: str) -> list[tuple[str, float, float, float]]:
    """``chair:2,0.5,1.57;box:1.5,-0.3`` -> [(kind, x, y, yaw), ...]."""
    out = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"bad prop {part!r}; expected kind:x,y[,yaw]")
        kind, coords = part.split(":", 1)
        kind = kind.strip().lower()
        if kind not in PROP_KINDS:
            raise ValueError(f"unknown prop kind {kind!r}; known: {', '.join(PROP_KINDS)}")
        values = [float(v) for v in coords.replace(" ", "").split(",") if v]
        if len(values) not in (2, 3):
            raise ValueError(f"bad prop {part!r}; expected kind:x,y[,yaw]")
        out.append((kind, values[0], values[1], values[2] if len(values) == 3 else 0.0))
    return out


def scene_path(name: str | Path) -> Path:
    """``name`` in :data:`SCENES_DIR` (``.yaml`` optional), or a path to a file."""
    path = Path(name)
    if path.suffix in (".yaml", ".yml") or path.parent != Path("."):
        return path
    return SCENES_DIR / f"{name}.yaml"


def list_scenes() -> list[str]:
    return sorted(p.stem for p in SCENES_DIR.glob("*.yaml"))


def load_scene(name: str | Path) -> list[tuple[str, float, float, float]]:
    """Read a scene file -> [(kind, x, y, yaw), ...].

    The file holds ``props:``, a list of ``{kind, x, y, yaw}`` (yaw optional,
    radians), and an optional ``description``.
    """
    import yaml

    path = scene_path(name)
    if not path.is_file():
        known = ", ".join(list_scenes()) or "none"
        raise ValueError(f"no scene {str(name)!r} ({path}); saved scenes: {known}")
    data = yaml.safe_load(path.read_text()) or {}
    out = []
    for item in data.get("props") or []:
        kind = str(item.get("kind", "")).strip().lower()
        if kind not in PROP_KINDS:
            raise ValueError(f"{path}: unknown prop kind {kind!r}; known: {', '.join(PROP_KINDS)}")
        out.append((kind, float(item["x"]), float(item["y"]), float(item.get("yaw", 0.0))))
    return out


def save_scene(name: str | Path, placed, description: str = "") -> Path:
    """Write ``placed`` [(kind, x, y, yaw), ...] as a scene file; return its path."""
    import yaml

    path = scene_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "description": description or f"Saved from run_loka ({len(placed)} props)",
        "props": [
            {"kind": kind, "x": round(x, 3), "y": round(y, 3), "yaw": round(yaw, 3)}
            for kind, x, y, yaw in placed
        ],
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=None))
    return path


def is_slot_name(text: str) -> bool:
    return _SLOT_NAME.match(text) is not None
