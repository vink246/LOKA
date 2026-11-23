"""Contact sequence over a terrain belief: footholds, timings and swing paths.

The planner works along the straight line from the feet to the goal. Each
foot keeps its own lateral line; a foothold is a distance along that line.
A foothold is admissible when the whole sole, grown by an edge margin, lies on
one believed surface. Steps are chosen greedily at the nominal stride and
moved to the nearest admissible distance, which on a flight of stairs lands
one foot per tread and on a gap reaches for the far side.

A swing follows a lift, carry, lower profile whose apex and horizontal window
are chosen so that the flat sole stays clear of every believed edge it passes
over.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from loka.control.terrain import TerrainBelief

LEFT, RIGHT = 0, 1


@dataclass
class StepParams:
    """Geometry and timing of the contact sequence."""

    stride: float = 0.22  # nominal advance of the swing foot past the stance foot [m]
    reach_max: float = 0.46  # farthest foothold ahead of the stance foot [m]
    reach_min: float = -0.02  # closest foothold relative to the stance foot [m]
    rise_max: float = 0.22  # largest height change between consecutive footholds [m]
    edge_margin: float = 0.025  # sole clearance to a believed edge, along the sole [m]
    side_margin: float = 0.01  # the same across the sole [m]
    sole_half_length: float = 0.085
    sole_half_width: float = 0.03
    width_range: tuple[float, float] = (0.20, 0.26)  # lateral foot separation [m]
    #: An admissible interval shorter than this is a tread: step to its middle [m].
    tread_max: float = 0.20
    #: Ground lower than the surfaces on both sides of it, and no wider than this, is a gap
    #: to step over rather than into [m]. Zero allows footholds in gaps.
    gap_max: float = 0.60
    gap_depth: float = 0.03  # how much lower than both sides counts as a gap [m]
    clearance: float = 0.05  # apex height above the highest ground under the swing [m]
    edge_clearance: float = 0.025  # sole height over an edge it passes [m]
    swing_base: float = 0.40  # swing time: base + per metre + per metre of height [s]
    swing_per_m: float = 0.45
    swing_per_dz: float = 1.0
    swing_range: tuple[float, float] = (0.50, 1.00)
    transfer: float = 0.36  # double support between swings [s]
    transfer_terrain: float = 0.50  # the same before a swing that changes height or spans a gap [s]
    first_transfer: float = 0.60  # double support before the first swing [s]
    settle: float = 1.0  # double support after the last swing [s]


@dataclass
class Foothold:
    foot: int
    pos: np.ndarray  # (3,) sole centre, world
    yaw: float


@dataclass
class Step:
    """One swing: the foot, where it leaves and lands, and its path."""

    foot: int
    start: Foothold
    end: Foothold
    duration: float
    transfer: float  # double support before this swing [s]
    apex: float = 0.0
    window: tuple[float, float] = (0.0, 1.0)  # swing phase over which the foot travels
    lift: tuple[float, float] = (0.4, 0.6)  # end of the rise, start of the descent

    def point(self, tau: float) -> tuple[np.ndarray, np.ndarray]:
        """Sole-centre position and velocity at swing phase ``tau`` in [0, 1]."""
        tau = float(np.clip(tau, 0.0, 1.0))
        h0, h1 = self.window
        a, da = _smooth((tau - h0) / (h1 - h0))
        da /= (h1 - h0) * self.duration
        delta = self.end.pos[:2] - self.start.pos[:2]
        xy = self.start.pos[:2] + a * delta
        up, down = self.lift
        z0, z1 = float(self.start.pos[2]), float(self.end.pos[2])
        if tau < up:
            b, db = _smooth(tau / up)
            z = z0 + (self.apex - z0) * b
            dz = (self.apex - z0) * db / (up * self.duration)
        elif tau > down:
            b, db = _smooth((tau - down) / (1.0 - down))
            z = self.apex + (z1 - self.apex) * b
            dz = (z1 - self.apex) * db / ((1.0 - down) * self.duration)
        else:
            z, dz = self.apex, 0.0
        return np.array([xy[0], xy[1], z]), np.array([da * delta[0], da * delta[1], dz])

    def yaw(self, tau: float) -> float:
        a, _ = _smooth(tau)
        return self.start.yaw + a * _wrap(self.end.yaw - self.start.yaw)


@dataclass
class ContactPlan:
    """Initial stance plus the swings that follow it."""

    stance: list[Foothold]  # [left, right] before the first swing
    steps: list[Step] = field(default_factory=list)
    heading: float = 0.0
    params: StepParams = field(default_factory=StepParams)

    @property
    def duration(self) -> float:
        return sum(s.transfer + s.duration for s in self.steps) + self.params.settle

    def final_stance(self) -> list[Foothold]:
        feet = list(self.stance)
        for step in self.steps:
            feet[step.foot] = step.end
        return feet


def _smooth(u: float) -> tuple[float, float]:
    """Quintic smoothstep and its derivative; zero velocity and acceleration at both ends."""
    u = float(np.clip(u, 0.0, 1.0))
    return 10 * u**3 - 15 * u**4 + 6 * u**5, 30 * u**2 - 60 * u**3 + 30 * u**4


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class _Line:
    """One foot's line of travel through the belief."""

    def __init__(self, belief: TerrainBelief, origin, forward, lateral, offset: float,
                 params: StepParams, floor: float) -> None:
        self.belief = belief
        self.origin = np.asarray(origin, dtype=float)
        self.fwd = np.asarray(forward, dtype=float)
        self.lat = np.asarray(lateral, dtype=float)
        self.offset = float(offset)
        self.p = params
        self.floor = float(floor)

    def xy(self, s: float) -> np.ndarray:
        return self.origin + s * self.fwd + self.offset * self.lat

    def _span(self, s: float, margin: float, side: float) -> np.ndarray:
        """Ground heights under the sole centred at ``s``, grown by the margins."""
        half = self.p.sole_half_length + margin
        wide = self.p.sole_half_width + side
        pts = [
            self.xy(s) + a * self.fwd + b * self.lat
            for a in np.linspace(-half, half, 9)
            for b in (-wide, 0.0, wide)
        ]
        return self.belief.heights_at(np.array(pts), self.floor)

    def height(self, s: float) -> float:
        return float(self.belief.heights_at(self.xy(s), self.floor)[0])

    def highest_under(self, s: float, margin: float) -> float:
        return float(self._span(s, margin, 0.0).max())

    def admissible(self, s: float) -> bool:
        h = self._span(s, self.p.edge_margin, self.p.side_margin)
        return bool(h.max() - h.min() <= 0.005) and not self.in_gap(s)

    def in_gap(self, s: float) -> bool:
        """Whether ``s`` lies in a trench: higher ground behind and ahead, ``gap_max`` apart at most."""
        if self.p.gap_max <= 0.0:
            return False
        level = self.height(s) + self.p.gap_depth
        offsets = np.arange(0.02, self.p.gap_max + 1e-9, 0.02)
        behind = next((d for d in offsets if self.height(s - d) > level), None)
        if behind is None:
            return False
        ahead = next((d for d in offsets if self.height(s + d) > level), None)
        return ahead is not None and behind + ahead <= self.p.gap_max + 0.04


def plan_contacts(
    belief: TerrainBelief,
    feet_xyz: np.ndarray,
    goal_xy,
    params: StepParams | None = None,
    floor: float = 0.0,
    heading: float | None = None,
) -> ContactPlan:
    """Footholds and swings from the current stance to ``goal_xy``.

    ``feet_xyz`` is the ``(2, 3)`` array of sole centres, left first. The feet
    end side by side with their midpoint at the goal (or as close to it as an
    admissible foothold allows).
    """
    p = params or StepParams()
    feet_xyz = np.asarray(feet_xyz, dtype=float).reshape(2, 3)
    goal = np.asarray(goal_xy, dtype=float).reshape(2)
    mid = feet_xyz[:, :2].mean(axis=0)
    if heading is None:
        heading = math.atan2(goal[1] - mid[1], goal[0] - mid[0])
    fwd = np.array([math.cos(heading), math.sin(heading)])
    lat = np.array([-fwd[1], fwd[0]])
    width = float(np.clip((feet_xyz[LEFT, :2] - feet_xyz[RIGHT, :2]) @ lat, *p.width_range))
    lines = [
        _Line(belief, mid, fwd, lat, +0.5 * width, p, floor),
        _Line(belief, mid, fwd, lat, -0.5 * width, p, floor),
    ]
    s = [float((feet_xyz[i, :2] - mid) @ fwd) for i in (LEFT, RIGHT)]
    z = [float(feet_xyz[i, 2]) for i in (LEFT, RIGHT)]
    stance = [Foothold(i, feet_xyz[i].copy(), heading) for i in (LEFT, RIGHT)]
    s_goal = float((goal - mid) @ fwd)
    plan = ContactPlan(stance=stance, heading=heading, params=p)
    current = list(stance)

    swing = LEFT if s[LEFT] <= s[RIGHT] else RIGHT
    stalled = 0
    for _ in range(80):
        support = 1 - swing
        line = lines[swing]
        target = min(s[support] + p.stride, s_goal)
        if abs(s[support] - s_goal) < 0.02:
            target = s[support]  # the other foot is home: close the stance beside it
        grid = np.arange(s[support] + p.reach_min, s[support] + p.reach_max + 1e-9, 0.005)
        ok = np.array(
            [
                line.admissible(c)
                and abs(line.height(c) - z[support]) <= p.rise_max
                for c in grid
            ]
        )
        if not ok.any():
            raise ValueError(
                f"no admissible foothold within reach of the stance foot at s={s[support]:.2f} m"
            )
        best = int(np.argmin(np.where(ok, np.abs(grid - target), np.inf)))
        lo = hi = best
        while lo > 0 and ok[lo - 1]:
            lo -= 1
        while hi < len(grid) - 1 and ok[hi + 1]:
            hi += 1
        choice = float(grid[best])
        # A short admissible interval that the grid does not run off the end of is a tread.
        if lo > 0 and hi < len(grid) - 1 and grid[hi] - grid[lo] <= p.tread_max:
            choice = 0.5 * float(grid[lo] + grid[hi])
        if abs(choice - s[swing]) < 0.01:
            # Staying put is the nearest option: take the nearest one ahead instead, if any.
            ahead = ok & (grid > max(s[swing], s[support]) + 0.02)
            if ahead.any():
                choice = float(grid[int(np.argmin(np.where(ahead, np.abs(grid - target), np.inf)))])
        if abs(choice - s[swing]) < 0.01:
            stalled += 1
            if stalled >= 2:
                if abs(s[LEFT] - s_goal) < 0.05 and abs(s[RIGHT] - s_goal) < 0.05:
                    break
                raise ValueError(
                    f"contact plan stalled at s={s[swing]:.2f} m, {s_goal - s[swing]:.2f} m short "
                    "of the goal (terrain ahead is out of reach)"
                )
            swing = support
            continue
        stalled = 0
        end = Foothold(swing, np.array([*line.xy(choice), line.height(choice)]), heading)
        step = Step(foot=swing, start=current[swing], end=end, duration=0.0, transfer=0.0)
        _shape_swing(step, line, s[swing], choice, p)
        terrain_step = (
            abs(end.pos[2] - step.start.pos[2]) > 0.01
            or step.apex - max(end.pos[2], step.start.pos[2]) > p.clearance + 0.005
            or _crosses_gap(line, s[swing], choice, min(end.pos[2], step.start.pos[2]))
        )
        step.transfer = (
            p.first_transfer if not plan.steps
            else p.transfer_terrain if terrain_step else p.transfer
        )
        plan.steps.append(step)
        current[swing] = end
        s[swing], z[swing] = choice, float(end.pos[2])
        if abs(s[LEFT] - s_goal) < 0.02 and abs(s[RIGHT] - s_goal) < 0.02:
            break
        if abs(s[LEFT] - s[RIGHT]) < 0.01 and abs(target - s_goal) < 1e-6:
            break  # feet together at the closest admissible spot to the goal
        swing = support
    else:
        raise ValueError("contact plan did not reach the goal in 80 steps")
    return plan


def _crosses_gap(line: _Line, s0: float, s1: float, level: float) -> bool:
    samples = np.linspace(s0, s1, 25)
    return bool(min(line.height(c) for c in samples) < level - 0.02)


def _shape_swing(step: Step, line: _Line, s0: float, s1: float, p: StepParams) -> None:
    """Set the swing's duration, apex and travel window so the sole clears every edge."""
    z0, z1 = float(step.start.pos[2]), float(step.end.pos[2])
    dist = abs(s1 - s0)
    step.duration = float(
        np.clip(p.swing_base + p.swing_per_m * dist + p.swing_per_dz * abs(z1 - z0), *p.swing_range)
    )
    samples = np.linspace(s0, s1, 41)
    ground = max(line.highest_under(c, 0.0) for c in samples)
    apex = max(ground, z0, z1) + p.clearance
    windows = [
        (0.0, 1.0), (0.1, 1.0), (0.0, 0.9), (0.1, 0.9), (0.2, 1.0), (0.2, 0.9), (0.0, 0.8),
        (0.3, 1.0), (0.1, 0.8), (0.3, 0.9), (0.2, 0.8), (0.3, 0.8), (0.35, 0.75),
    ]
    for extra in np.arange(0.0, 0.101, 0.01):
        step.apex = apex + float(extra)
        for window in windows:
            step.window = window
            if _swing_clear(step, line, s0, s1, p):
                return
    # No profile clears: keep the tallest, tightest one and let the solver trade it off.
    step.window = windows[-1]


def _swing_clear(step: Step, line: _Line, s0: float, s1: float, p: StepParams) -> bool:
    z0, z1 = float(step.start.pos[2]), float(step.end.pos[2])
    for tau in np.linspace(0.0, 1.0, 61):
        pos, _ = step.point(tau)
        a, _ = _smooth((tau - step.window[0]) / (step.window[1] - step.window[0]))
        under = line.highest_under(s0 + a * (s1 - s0), 0.015)
        leaving = tau <= 0.5 and under <= z0 + 1e-3
        landing = tau >= 0.5 and under <= z1 + 1e-3
        if leaving or landing:
            if pos[2] < under - 0.002:
                return False
            continue
        if pos[2] - under < p.edge_clearance:
            return False
    return True


def describe(plan: ContactPlan) -> list[str]:
    """One line per swing, for logs."""
    lines = []
    t = 0.0
    for i, step in enumerate(plan.steps):
        t += step.transfer
        lines.append(
            f"step {i:2d} {'LR'[step.foot]} t={t:5.2f}s +{step.duration:.2f}s  "
            f"({step.start.pos[0]:+.3f},{step.start.pos[1]:+.3f},{step.start.pos[2]:.3f}) -> "
            f"({step.end.pos[0]:+.3f},{step.end.pos[1]:+.3f},{step.end.pos[2]:.3f})  "
            f"apex {step.apex:.3f} window {step.window[0]:.2f}-{step.window[1]:.2f}"
        )
        t += step.duration
    return lines
