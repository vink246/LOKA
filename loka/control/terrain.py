"""Terrain the controller believes in: a list of solid boxes on the floor.

The belief ``MjModel`` never runs collision, so terrain does not need to be
compiled into it. Each :class:`Volume` is a yaw-rotated box that rises from
the floor to ``top``. The gait planner asks two questions of it: how high is
the ground under a sole, and is that sole fully on one surface. With no
volumes the controller is the flat-floor controller, bit for bit.

The plant builds its stairs from the same :func:`stairs_volumes`, so an
oracle belief is exactly the plant geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: Sole extent about the foot-centre site [m]. Heel and toe spheres sit at
#: -0.05 and +0.12 from the ankle, the centre site at +0.035.
SOLE_HALF_LENGTH = 0.085
SOLE_HALF_WIDTH = 0.035
#: Clearance kept between the sole and a tread edge [m].
EDGE_MARGIN = 0.02
#: The same near a volume the orchestrator declared [m]. A camera estimate
#: put a curb's edge 5 cm early, and a 3 cm margin then landed the heel on
#: the real nosing.
UNVERIFIED_EDGE_MARGIN = 0.05
#: Largest height change one step may take [m].
STEP_UP_MAX = 0.16
#: Shallowest tread the planner will put a whole sole on [m].
MIN_TREAD = 0.25
#: Terrain-mode controller variant. ``heuristic``: the tuned stack (cruise
#: stride snapped onto treads, MPC forces, torso lean). ``classical``: the
#: humanoid stair stack of the literature (IHMC Atlas, Caron's HRP-4, AIST's
#: BaselineWalkingController): one foot per tread with distance-scaled
#: swings, 3D DCM with a heel-to-toe VRP, DCM feedback to a CMP and a contact
#: wrench distribution in place of the MPC, upright torso with angular
#: momentum damping, lateral step adjustment. ``wbto``: Crocoddyl whole-body
#: FDDP over a contact plan (needs pinocchio+crocoddyl). ``auto``: WBTO for
#: any challenging raised belief (curb, stairs, gap) when the deps are
#: installed; flat stays classical. Read at run time (``--terrain-stack`` on
#: run_loka and the bench scripts). Classical is the default flat regress
#: stack; use ``auto`` or ``wbto`` for terrain (docs/LOKA.md 4.9).
TERRAIN_STACK = "classical"
TERRAIN_STACK_CHOICES = ("classical", "heuristic", "auto", "wbto")

#: Tops at or above this count as challenging raised ground for WBTO [m].
WBTO_RAISED_TOP = 0.015


def classical_stack() -> bool:
    """True when the reduced-order terrain path should use the classical DCM stack.

    ``auto`` / ``wbto`` fall back to classical whenever the whole-body session
    is not engaged (flat belief or missing Crocoddyl).
    """
    return TERRAIN_STACK != "heuristic"


def terrain_wants_wbto(belief: "TerrainBelief") -> bool:
    """Challenging terrain for ``auto``: any non-trivial raised belief.

    Flat (no volumes / only dust-height tops) stays classical. A curb, a
    flight of stairs, or separated pads (gap) — including declarations that
    almost touch so a gap scan would miss — all select WBTO.
    """
    if not belief.active:
        return False
    return any(float(vol.top) >= WBTO_RAISED_TOP for vol in belief.volumes)


def _belief_has_gap(belief: "TerrainBelief") -> bool:
    """True when a scan through the volumes sees raised → floor → raised."""
    if len(belief.volumes) < 2:
        return False
    y = float(np.mean([float(vol.center[1]) for vol in belief.volumes]))
    xs: list[float] = []
    for vol in belief.volumes:
        half = 0.5 * float(vol.size[0])
        xs.extend([float(vol.center[0]) - half, float(vol.center[0]) + half])
    x0, x1 = min(xs) - 0.05, max(xs) + 0.05
    n = max(int(round((x1 - x0) / 0.02)), 3)
    pts = np.column_stack([np.linspace(x0, x1, n), np.full(n, y)])
    raised = belief.heights_at(pts, 0.0) >= 0.02
    i = 0
    while i < n and not raised[i]:
        i += 1
    while i < n and raised[i]:
        i += 1
    gap_start = i
    while i < n and not raised[i]:
        i += 1
    if i >= n or gap_start >= i:
        return False
    width = (i - gap_start) * (x1 - x0) / (n - 1)
    return 0.04 <= width <= 0.65


#: Longest stride a terrain snap may produce, from the other foot [m].
STAIR_STRIDE_MAX = 0.40
#: Where a tread-by-tread plan puts the sole centre past the edge of a deep
#: surface, such as a landing [m]: half of a 0.30 m tread.
TREAD_SETBACK = 0.15
#: A height difference below this is the same surface [m].
SAME_SURFACE_TOL = 0.005
#: Height change at which a step counts as a stair step [m].
STAIR_STEP_DZ = 0.02
#: Cap on the number of believed volumes.
MAX_VOLUMES = 16

#: Clamps on what an orchestrator may write.
TOP_RANGE = (0.0, 1.0)
SIZE_RANGE = (0.05, 5.0)
CENTER_RANGE = (-20.0, 20.0)
STAIR_STEPS_RANGE = (1, 8)
#: Sole z vs believed height under a planted foot [m]. Larger means the
#: belief left the stance on the floor while the plant put it on a pad.
STANCE_HEIGHT_TOL = 0.02
#: Extra xy when growing a volume under the soles [m].
STANCE_XY_MARGIN = 0.12
#: If the nearest same-height pad is farther than this from a sole, insert a
#: stance pad instead of extending — avoids filling a real gap [m].
STANCE_EXTEND_REACH = 0.55
#: Max |top − sole_z| treated as the same surface (vision height error) [m].
#: Kept below a typical stair riser so adjacent treads are not flattened.
STANCE_TOP_SNAP_MAX = 0.05
STAIR_RISER_RANGE = (0.02, 0.20)
STAIR_TREAD_RANGE = (0.15, 1.0)
STAIR_WIDTH_RANGE = (0.3, 3.0)
STAIR_LANDING_RANGE = (0.2, 3.0)


def _clip(value, lo, hi) -> float:
    return float(np.clip(float(value), lo, hi))


@dataclass
class Volume:
    """A solid box from the floor up to ``top``, yawed about world z."""

    name: str
    center: np.ndarray  # (2,) world xy
    size: np.ndarray  # (2,) full depth (local x) and width (local y) [m]
    top: float  # world z of the top face [m]
    yaw: float = 0.0
    source: str = "llm"
    #: The ``stairs`` declaration this volume was expanded from, if any.
    group: str | None = None
    #: Position within that declaration, 0 at the bottom step.
    index: int = 0

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float).reshape(2)
        self.size = np.asarray(self.size, dtype=float).reshape(2)
        self.top = float(self.top)
        self.yaw = float(self.yaw)

    def local(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 2) - self.center
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.stack([c * pts[:, 0] + s * pts[:, 1], -s * pts[:, 0] + c * pts[:, 1]], axis=1)

    def contains(self, points: np.ndarray) -> np.ndarray:
        loc = self.local(points)
        half = 0.5 * self.size
        return (np.abs(loc[:, 0]) <= half[0]) & (np.abs(loc[:, 1]) <= half[1])

    def to_dict(self) -> dict:
        out = {
            "name": self.name,
            "center": [round(float(v), 4) for v in self.center],
            "size": [round(float(v), 4) for v in self.size],
            "top": round(self.top, 4),
            "yaw": round(self.yaw, 4),
            "source": self.source,
        }
        if self.group is not None:
            out["group"] = self.group
            out["index"] = int(self.index)
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "Volume":
        return cls(
            name=str(data["name"]),
            center=data["center"],
            size=data["size"],
            top=float(data["top"]),
            yaw=float(data.get("yaw", 0.0)),
            source=str(data.get("source", "llm")),
            group=data.get("group"),
            index=int(data.get("index", 0)),
        )


def stairs_volumes(spec: dict, *, source: str = "oracle") -> list[Volume]:
    """Volumes of one straight flight of stairs.

    ``x``, ``y`` is the foot of the first riser, on the centre line. The
    flight climbs along ``yaw``. ``riser`` is one height or a list, one per
    step; ``landing`` is the depth of the top step. A ``wall`` spec is not
    ground and has none.
    """
    if str(spec.get("kind", "stairs")).lower() == "wall":
        return []
    name = str(spec.get("name", "stairs"))
    steps = int(spec.get("steps", 3))
    risers = spec.get("riser", 0.10)
    if isinstance(risers, (list, tuple)):
        risers = [float(r) for r in risers]
        steps = len(risers)
    else:
        risers = [float(risers)] * steps
    tread = float(spec.get("tread", 0.30))
    width = float(spec.get("width", 1.0))
    landing = float(spec.get("landing", 1.0))
    yaw = float(spec.get("yaw", 0.0))
    origin = np.array([float(spec.get("x", 0.0)), float(spec.get("y", 0.0))])
    forward = np.array([math.cos(yaw), math.sin(yaw)])
    out = []
    top = float(spec.get("base", 0.0))
    for k in range(steps):
        top += risers[k]
        depth = landing if k == steps - 1 else tread
        start = k * tread
        center = origin + forward * (start + 0.5 * depth)
        out.append(
            Volume(
                name=f"{name}_{k}",
                center=center,
                size=np.array([depth, width]),
                top=top,
                yaw=yaw,
                source=source,
                group=name,
                index=k,
            )
        )
    return out


def clamp_stairs_spec(raw: dict) -> dict:
    """Bounded copy of an orchestrator's ``stairs`` declaration."""
    if not isinstance(raw, dict):
        raise TypeError("stairs value must be a mapping")
    start = raw.get("start", raw.get("xy", [raw.get("x", 0.0), raw.get("y", 0.0)]))
    start = np.asarray(start, dtype=float).reshape(-1)
    if start.size != 2:
        raise ValueError("stairs start must be [x, y]")
    riser = raw.get("riser", 0.10)
    if isinstance(riser, (list, tuple)):
        riser = [_clip(r, *STAIR_RISER_RANGE) for r in riser][: STAIR_STEPS_RANGE[1]]
        steps = len(riser)
    else:
        riser = _clip(riser, *STAIR_RISER_RANGE)
        steps = int(np.clip(int(raw.get("steps", 3)), *STAIR_STEPS_RANGE))
    return {
        "x": _clip(start[0], *CENTER_RANGE),
        "y": _clip(start[1], *CENTER_RANGE),
        "yaw": float(raw.get("heading", raw.get("yaw", 0.0))),
        "steps": steps,
        "riser": riser,
        "tread": _clip(raw.get("tread", 0.30), *STAIR_TREAD_RANGE),
        "width": _clip(raw.get("width", 1.0), *STAIR_WIDTH_RANGE),
        "landing": _clip(raw.get("landing", 1.0), *STAIR_LANDING_RANGE),
    }


def clamp_box(raw: dict, previous: Volume | None = None) -> dict:
    """Bounded ``box`` fields, merged over an existing volume of that name."""
    if not isinstance(raw, dict):
        raise TypeError("box value must be a mapping")
    base = previous.to_dict() if previous is not None else {
        "center": [0.0, 0.0], "size": [0.3, 1.0], "top": 0.1, "yaw": 0.0,
    }
    center = np.asarray(raw.get("center", base["center"]), dtype=float).reshape(-1)
    size = np.asarray(raw.get("size", base["size"]), dtype=float).reshape(-1)
    if center.size != 2 or size.size != 2:
        raise ValueError("box center and size must be 2-vectors")
    return {
        "center": [_clip(v, *CENTER_RANGE) for v in center],
        "size": [_clip(v, *SIZE_RANGE) for v in size],
        "top": _clip(raw.get("top", base["top"]), *TOP_RANGE),
        "yaw": float(raw.get("yaw", base["yaw"])),
    }


def footprint_points(center, yaw: float, margin: float = 0.0) -> np.ndarray:
    """Sole sample points (corners, edge midpoints, centre), world xy."""
    c, s = math.cos(yaw), math.sin(yaw)
    fx, fy = np.array([c, s]), np.array([-s, c])
    length = SOLE_HALF_LENGTH + margin
    width = SOLE_HALF_WIDTH
    pts = []
    for a in (-length, -0.5 * length, 0.0, 0.5 * length, length):
        for b in (-width, 0.0, width):
            pts.append(np.asarray(center, dtype=float)[:2] + a * fx + b * fy)
    return np.array(pts)


def _volume_outside_distance(vol: Volume, point) -> float:
    """Distance from ``point`` to the volume AABB in the volume frame (0 inside)."""
    loc = vol.local(np.asarray(point, dtype=float)[:2])[0]
    half = 0.5 * vol.size
    dx = max(0.0, abs(float(loc[0])) - float(half[0]))
    dy = max(0.0, abs(float(loc[1])) - float(half[1]))
    return float(math.hypot(dx, dy))


def _extend_volume_to_cover(vol: Volume, points: np.ndarray, margin: float) -> None:
    """Grow ``vol`` in its local frame so every sole sample lies inside."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    samples = []
    for p in pts:
        for dx in (-margin - SOLE_HALF_LENGTH, 0.0, margin + SOLE_HALF_LENGTH):
            for dy in (-margin - SOLE_HALF_WIDTH, 0.0, margin + SOLE_HALF_WIDTH):
                samples.append(p + np.array([dx, dy]))
    loc = vol.local(np.asarray(samples, dtype=float))
    half = 0.5 * np.asarray(vol.size, dtype=float)
    new_lo = np.minimum(-half, loc.min(axis=0))
    new_hi = np.maximum(half, loc.max(axis=0))
    new_half = 0.5 * (new_hi - new_lo)
    new_center_local = 0.5 * (new_lo + new_hi)
    c, s = math.cos(vol.yaw), math.sin(vol.yaw)
    offset = np.array(
        [
            c * new_center_local[0] - s * new_center_local[1],
            s * new_center_local[0] + c * new_center_local[1],
        ]
    )
    vol.center = np.asarray(vol.center, dtype=float) + offset
    vol.size = np.clip(2.0 * new_half, SIZE_RANGE[0], SIZE_RANGE[1])


def _stance_volumes_near_feet(
    belief: "TerrainBelief",
    pts: np.ndarray,
    sole_z: float,
    *,
    snap_max: float,
    extend_reach: float,
) -> list[Volume]:
    """Volumes under the soles, or nearby pads at about the sole height."""
    covering = [vol for vol in belief.volumes if any(vol.contains(p)[0] for p in pts)]
    if covering:
        return covering
    nearby = []
    for vol in belief.volumes:
        if abs(float(vol.top) - sole_z) > snap_max:
            continue
        if max(_volume_outside_distance(vol, p) for p in pts) <= extend_reach:
            nearby.append(vol)
    return nearby


def ensure_stance_support(
    belief: "TerrainBelief",
    feet_xyz: np.ndarray,
    *,
    height_tol: float = STANCE_HEIGHT_TOL,
    xy_margin: float = STANCE_XY_MARGIN,
    extend_reach: float = STANCE_EXTEND_REACH,
    snap_max: float = STANCE_TOP_SNAP_MAX,
) -> bool:
    """Make believed ground under the soles match where they currently rest.

    Vision often (1) starts the near pad ahead of the robot or (2) overestimates
    pad height. Whole-body planning then fights the plant and collapses.
    Snaps coplanar tops to the measured sole height, then extends the nearest
    pad under the feet (or inserts a local stance pad) without filling a real
    gap. Does not bump ``belief.version`` — sync would otherwise reload the
    short LLM pads every tick. Returns True when the belief changed.
    """
    feet = np.asarray(feet_xyz, dtype=float).reshape(2, 3)
    pts = feet[:, :2]
    sole_z = float(np.median(feet[:, 2]))
    changed = False

    # Height: vision tops of 0.12–0.13 m on a 0.10 m plant must snap before plan.
    seed = _stance_volumes_near_feet(
        belief, pts, sole_z, snap_max=snap_max, extend_reach=extend_reach
    )
    snap_keys = {
        round(float(vol.top), 3)
        for vol in seed
        if abs(float(vol.top) - sole_z) > height_tol
    }
    if snap_keys:
        for vol in belief.volumes:
            if round(float(vol.top), 3) in snap_keys:
                vol.top = sole_z
                changed = True

    under = belief.heights_at(pts, 0.0)
    if bool(np.all(np.abs(under - sole_z) <= height_tol)):
        return changed

    candidates = [
        vol for vol in belief.volumes if abs(float(vol.top) - sole_z) <= height_tol
    ]
    mid = pts.mean(axis=0)
    if candidates:
        vol = min(candidates, key=lambda v: float(np.linalg.norm(v.center - mid)))
        reach = max(_volume_outside_distance(vol, p) for p in pts)
        if reach <= extend_reach:
            _extend_volume_to_cover(vol, pts, xy_margin)
            return True

    half = np.array(
        [
            max(0.25, float(np.ptp(pts[:, 0]) * 0.5) + xy_margin + SOLE_HALF_LENGTH),
            max(0.20, float(np.ptp(pts[:, 1]) * 0.5) + xy_margin + SOLE_HALF_WIDTH),
        ]
    )
    pad = Volume(
        name="stance_support",
        center=mid.copy(),
        size=2.0 * half,
        top=sole_z,
        yaw=0.0,
        source="seed",
    )
    belief.volumes = list(belief.volumes) + [pad]
    return True


@dataclass
class TerrainBelief:
    """The controller's terrain: volumes over a flat floor."""

    volumes: list[Volume] = field(default_factory=list)
    version: int = 0

    @property
    def active(self) -> bool:
        return bool(self.volumes)

    def set_volumes(self, volumes, version: int | None = None) -> None:
        self.volumes = [v if isinstance(v, Volume) else Volume.from_dict(v) for v in volumes]
        self.version = self.version + 1 if version is None else int(version)

    def heights_at(self, points: np.ndarray, floor: float = 0.0) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        out = np.full(len(pts), float(floor))
        for vol in self.volumes:
            inside = vol.contains(pts)
            if inside.any():
                out[inside] = np.maximum(out[inside], vol.top)
        return out

    def height_at(self, xy, floor: float = 0.0) -> float:
        return float(self.heights_at(np.asarray(xy, dtype=float)[:2], floor)[0])

    def volume_at(self, xy) -> Volume | None:
        """The highest volume under ``xy``, or None over the floor."""
        best = None
        pt = np.asarray(xy, dtype=float)[:2]
        for vol in self.volumes:
            if vol.contains(pt)[0] and (best is None or vol.top > best.top):
                best = vol
        return best

    def support_height(self, center, yaw: float, floor: float = 0.0, margin: float = 0.0) -> float:
        """Highest ground under the sole."""
        return float(self.heights_at(footprint_points(center, yaw, margin), floor).max())

    def edge_margin(self, center) -> float:
        """Edge clearance near ``center``: wider beside a declared volume.

        Touchdowns measure a tread's height, not where its edges are, so a
        refined volume's edges are still the orchestrator's estimate.
        """
        pt = np.asarray(center, dtype=float)[:2]
        for vol in self.volumes:
            if vol.source == "oracle":
                continue
            loc = vol.local(pt)[0]
            reach = 0.5 * vol.size + SOLE_HALF_LENGTH + UNVERIFIED_EDGE_MARGIN
            if abs(loc[0]) <= reach[0] and abs(loc[1]) <= reach[1]:
                return UNVERIFIED_EDGE_MARGIN
        return EDGE_MARGIN

    def footprint_uniform(self, center, yaw: float, floor: float = 0.0, margin: float | None = None) -> bool:
        """True when the sole, grown by ``margin`` along its length, is on one surface."""
        if margin is None:
            margin = self.edge_margin(center)
        heights = self.heights_at(footprint_points(center, yaw, margin), floor)
        return float(heights.max() - heights.min()) <= SAME_SURFACE_TOL

    def next_tread_center(self, anchor, forward, anchor_z: float, floor: float = 0.0,
                          reach: float = 0.8):
        """Distance along ``forward`` to the middle of the next surface ahead.

        Scans the line from ``anchor`` for the first height change (a riser),
        then for the one after it. The foothold goes to the middle of that
        tread; on a deep surface (a landing) it goes as close to the edge as
        a sole plus margin allows, the way a tread-by-tread stair plan puts
        one foot on each step. A drop that rises again within ``reach`` (a
        gap in a raised walkway) is stepped over onto that far surface, not
        into the hole. Returns ``(distance, height)`` or None.
        """
        anchor = np.asarray(anchor, dtype=float).reshape(2)
        fwd = np.asarray(forward, dtype=float).reshape(2)
        fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-9)
        ds = np.arange(0.0, reach, 0.005)
        h = self.heights_at(anchor[None, :] + ds[:, None] * fwd[None, :], floor)
        # Riser test against the belief at the start of the scan: ``anchor_z``
        # is the measured stance height, which a refinement can move off it.
        base = float(h[0])
        rise = np.nonzero(np.abs(h - base) >= STAIR_STEP_DZ)[0]
        if rise.size == 0:
            return None
        start = int(rise[0])
        top = float(h[min(start + 1, len(h) - 1)])
        # Same-height gap: the first change is a drop, then the walkway resumes.
        gap_cross = False
        if top < base - STAIR_STEP_DZ:
            recover = np.nonzero(h[start:] >= base - SAME_SURFACE_TOL)[0]
            if recover.size == 0:
                return None
            start = start + int(recover[0])
            top = float(h[min(start + 1, len(h) - 1)])
            gap_cross = True
        if abs(top - base) > STEP_UP_MAX or abs(top - float(anchor_z)) > STEP_UP_MAX + STAIR_STEP_DZ:
            return None
        after = np.nonzero(np.abs(h[start:] - top) > SAME_SURFACE_TOL)[0]
        set_in = SOLE_HALF_LENGTH + self.edge_margin(anchor + ds[start] * fwd)
        if after.size == 0:
            # After a gap, plant just past the far edge; a deep-landing setback
            # would push the stride past STAIR_PITCH_MAX on a short step-over.
            inset = set_in if gap_cross else max(set_in, TREAD_SETBACK)
            along = float(ds[start]) + inset
        else:
            depth = float(ds[start + int(after[0])] - ds[start])
            if depth < 2.0 * set_in:
                return None
            along = float(ds[start]) + max(0.5 * depth, set_in)
        return along, top

    def snap_foothold(
        self,
        pos,
        yaw: float,
        forward,
        anchor,
        anchor_z: float,
        floor: float = 0.0,
        nearest: bool = False,
    ) -> np.ndarray:
        """Move a foothold along ``forward`` until its sole sits on one tread.

        ``anchor`` is the other foot. Backing off is preferred while it still
        makes progress past the anchor; otherwise the foot steps onto the
        next surface, if that step is short and low enough. ``nearest`` (a
        turn in place, which makes no progress) takes the closer clear spot.
        A foothold that is already clear of every edge is returned unchanged.
        """
        pos = np.asarray(pos, dtype=float).reshape(2)
        if not self.active or self.footprint_uniform(pos, yaw, floor):
            return pos
        fwd = np.asarray(forward, dtype=float).reshape(2)
        fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-9)
        anchor = np.asarray(anchor, dtype=float).reshape(2)

        def first_clear(sign: float):
            for d in np.arange(0.01, 0.45, 0.01):
                cand = pos + sign * d * fwd
                if self.footprint_uniform(cand, yaw, floor):
                    return cand
            return None

        back = first_clear(-1.0)
        ahead = first_clear(1.0)
        progress = 0.02
        if ahead is not None:
            rise = abs(self.support_height(ahead, yaw, floor) - float(anchor_z))
            stride = float(np.dot(ahead - anchor, fwd))
            if rise > STEP_UP_MAX or stride > STAIR_STRIDE_MAX:
                ahead = None
        if nearest and back is not None and ahead is not None:
            closer_back = np.linalg.norm(back - pos) <= np.linalg.norm(ahead - pos)
            return back if closer_back else ahead
        if back is not None and float(np.dot(back - anchor, fwd)) >= progress:
            return back
        if ahead is not None:
            return ahead
        if back is not None:
            return back
        return pos

    def clamp_to_surface(self, pos, nominal, yaw: float, z: float, floor: float = 0.0) -> np.ndarray:
        """Keep a corrected foothold on the surface its nominal was planned on."""
        pos = np.asarray(pos, dtype=float).reshape(2)
        if not self.active:
            return pos
        half = 0.5 * self.edge_margin(pos)
        ok = (
            self.footprint_uniform(pos, yaw, floor, margin=half)
            and abs(self.support_height(pos, yaw, floor) - float(z)) <= SAME_SURFACE_TOL
        )
        if ok:
            return pos
        nominal = np.asarray(nominal, dtype=float).reshape(2)
        c, s = math.cos(yaw), math.sin(yaw)
        fwd = np.array([c, s])
        lateral = pos - nominal - float(np.dot(pos - nominal, fwd)) * fwd
        cand = nominal + lateral
        if (
            self.footprint_uniform(cand, yaw, floor, margin=half)
            and abs(self.support_height(cand, yaw, floor) - float(z)) <= SAME_SURFACE_TOL
        ):
            return cand
        return nominal.copy()

    def describe(self) -> list[str]:
        lines = []
        for vol in self.volumes:
            lines.append(
                f"{vol.name}: centre ({vol.center[0]:+.2f}, {vol.center[1]:+.2f}) m, "
                f"size {vol.size[0]:.2f} x {vol.size[1]:.2f} m, yaw {vol.yaw:+.2f} rad, "
                f"top z={vol.top:.3f} m ({vol.source})"
            )
        return lines


def plant_terrain_height(specs, xy) -> float:
    """Ground height under ``xy`` in a plant built from ``specs``."""
    if not specs:
        return 0.0
    belief = TerrainBelief()
    vols = []
    for spec in specs:
        vols.extend(stairs_volumes(spec))
    belief.set_volumes(vols)
    return belief.height_at(xy)
