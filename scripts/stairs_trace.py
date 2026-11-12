"""Trace one oracle stair run around the first step-up, and label the fall.

    python scripts/stairs_trace.py --riser 0.10 --from 5.5 --to 9
    python scripts/stairs_trace.py --scene curb --to 20 --quiet

:class:`FallAttribution` is the diagnostic. It reads the simulator freely
(contact forces, joint ranges) because it only grades a run; nothing it
computes reaches the controller. Each step it checks a set of constraints;
at a fall it reports which one broke first in the window before the fall.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loka.scene_props import load_scene_terrain  # noqa: E402
from loka.sim import Simulation  # noqa: E402
from stairs_gate import controller_config, landing_goal  # noqa: E402

GRAVITY = 9.81
#: Fraction of the torque limit that counts as saturated.
SATURATION = 0.97
#: Distance from a joint range end that counts as at the limit [rad].
JOINT_LIMIT_BAND = 0.02
COP_EDGE_M = 0.005
DCM_ERROR_M = 0.06
#: Vertical / horizontal MPC-vs-WBC contact force gap, as a fraction of weight.
FORCE_GAP_Z = 0.25
FORCE_GAP_XY = 0.15
TILT_DEG = 12.0
LATERAL_FOOTHOLD_M = 0.03
#: A violation must last this long to count [s].
SUSTAIN_S = 0.03
#: How far before the fall to look for the first violation [s].
WINDOW_S = 1.5
#: Leg joints plus the waist.
TRACKED_JOINTS = 15


def _hull(points: np.ndarray) -> np.ndarray:
    pts = sorted({(round(float(p[0]), 5), round(float(p[1]), 5)) for p in points})
    if len(pts) < 3:
        return np.asarray(pts, dtype=float).reshape(-1, 2)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def polygon_margin(point: np.ndarray, hull: np.ndarray) -> float:
    """Signed distance from ``point`` to the edge of a CCW hull; positive inside."""
    if len(hull) < 3:
        return 0.0
    margin = np.inf
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        edge = b - a
        length = float(np.linalg.norm(edge))
        if length < 1e-9:
            continue
        rel = point - a
        margin = min(margin, float(edge[0] * rel[1] - edge[1] * rel[0]) / length)
    return float(margin)


class FallAttribution:
    """Per-step constraint checks and a first-violation label at the fall."""

    def __init__(self, sim: Simulation) -> None:
        self.sim = sim
        model = sim.model
        ctrl = sim.controller
        self.limit = np.asarray(ctrl.robot.torque_limit[:TRACKED_JOINTS], dtype=float)
        self.names = []
        self.ranges = np.full((TRACKED_JOINTS, 2), [-np.inf, np.inf])
        for i in range(TRACKED_JOINTS):
            jid = i + 1
            self.names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid))
            if model.jnt_limited[jid]:
                self.ranges[i] = model.jnt_range[jid]
        foot_bodies = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("left_ankle_roll_link", "right_ankle_roll_link")
        }
        self.foot_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] in foot_bodies}
        self.weight = float(ctrl.robot.total_mass) * GRAVITY
        self.t: list[float] = []
        self.flags: dict[str, list[tuple[float, bool]]] = {}
        self.sat_steps = np.zeros(TRACKED_JOINTS)
        self.limit_steps = np.zeros(TRACKED_JOINTS)
        self.steps = 0
        self.peak_torque = np.zeros(TRACKED_JOINTS)
        self.cop_margins: list[float] = []
        self.lateral_errors: list[float] = []
        self.max_tilt = 0.0
        self.max_dcm_error = 0.0
        self._foothold: dict[int, tuple[np.ndarray, float]] = {}
        self._qp_failures: dict[str, int] = {}
        self.qp_failures: dict[str, int] = {}

    def _flag(self, name: str, t: float, on: bool) -> None:
        self.flags.setdefault(name, []).append((t, bool(on)))

    def _cop_margin(self) -> float | None:
        model, data = self.sim.model, self.sim.data
        points, forces = [], []
        wrench = np.zeros(6)
        for i in range(data.ncon):
            con = data.contact[i]
            if con.geom1 not in self.foot_geoms and con.geom2 not in self.foot_geoms:
                continue
            mujoco.mj_contactForce(model, data, i, wrench)
            if wrench[0] <= 1.0:
                continue
            points.append(con.pos[:2].copy())
            forces.append(float(wrench[0]))
        if not points:
            return None
        pts = np.asarray(points)
        w = np.asarray(forces)
        cop = (pts * w[:, None]).sum(axis=0) / w.sum()
        return polygon_margin(cop, _hull(pts))

    def update(self) -> None:
        sim, ctrl = self.sim, self.sim.controller
        tel, g = ctrl.telemetry, ctrl.last_gait
        t = float(sim.data.time)
        self.t.append(t)
        self.steps += 1

        torque = np.abs(np.asarray(tel.torque)[:TRACKED_JOINTS])
        self.peak_torque = np.maximum(self.peak_torque, torque)
        sat = torque >= SATURATION * self.limit
        self.sat_steps += sat
        q = sim.data.qpos[7:7 + TRACKED_JOINTS]
        near = (q - self.ranges[:, 0] < JOINT_LIMIT_BAND) | (self.ranges[:, 1] - q < JOINT_LIMIT_BAND)
        self.limit_steps += near
        for i in range(TRACKED_JOINTS):
            self._flag(f"torque_saturation:{self.names[i]}", t, sat[i])
            if np.isfinite(self.ranges[i]).all():
                self._flag(f"joint_limit:{self.names[i]}", t, near[i])

        margin = self._cop_margin()
        if margin is not None:
            self.cop_margins.append(margin)
        self._flag("cop_at_edge", t, margin is not None and margin < COP_EDGE_M)

        com = np.asarray(tel.com)
        height = max(float(com[2]) - float(getattr(g, "support_z", 0.0) or 0.0), 0.3)
        omega = np.sqrt(GRAVITY / height)
        dcm = com[:2] + np.asarray(tel.com_velocity)[:2] / omega
        err = float(np.linalg.norm(dcm - np.asarray(g.dcm_ref)[:2])) if g.walking else 0.0
        self.max_dcm_error = max(self.max_dcm_error, err)
        self._flag("dcm_error", t, err > DCM_ERROR_M)

        wbc = getattr(ctrl, "_wbc_terrain", None) or ctrl.wbc
        for name, owner in (("wbc_qp_failed", wbc), ("mpc_qp_failed", ctrl.mpc)):
            qp = getattr(owner, "_qp", None)
            count = int(getattr(qp, "failures", 0)) if qp is not None else 0
            self._flag(name, t, count > self._qp_failures.get(name, 0))
            if count > self._qp_failures.get(name, 0):
                self.qp_failures[name] = self.qp_failures.get(name, 0) + 1
            self._qp_failures[name] = count

        des = np.asarray(tel.desired_forces).reshape(-1, 3).sum(axis=0)
        sol = np.asarray(tel.contact_forces).reshape(-1, 3).sum(axis=0)
        gap_z = abs(des[2] - sol[2]) / self.weight
        gap_xy = float(np.linalg.norm(des[:2] - sol[:2])) / self.weight
        self._flag("mpc_wbc_force_gap", t, gap_z > FORCE_GAP_Z or gap_xy > FORCE_GAP_XY)

        tilt = float(np.degrees(np.abs(np.asarray(tel.rpy)[:2]).max()))
        self.max_tilt = max(self.max_tilt, tilt)
        self._flag("torso_tilt", t, tilt > TILT_DEG)

        feet = ctrl.robot.foot_center_positions()
        if g.swing.active:
            self._foothold[g.swing.leg] = (np.asarray(g.swing.foothold[:2]).copy(),
                                           float(g.swing.foothold_yaw))
        lateral = False
        for event in ctrl.terrain_events:
            planned = self._foothold.get(event.get("leg"))
            if event.get("kind") != "touchdown" or planned is None:
                continue
            target, yaw = planned
            side = np.array([-np.sin(yaw), np.cos(yaw)])
            error = float(abs((feet[event["leg"], :2] - target) @ side))
            self.lateral_errors.append(error)
            lateral = lateral or error > LATERAL_FOOTHOLD_M
        self._flag("lateral_foothold", t, lateral)

    def _onset(self, series: list[tuple[float, bool]], t0: float, t1: float) -> float | None:
        need = SUSTAIN_S
        start = None
        for t, on in series:
            if t < t0 or t > t1:
                continue
            if on:
                start = t if start is None else start
                if t - start >= need or series[-1][0] == t:
                    return start
            else:
                start = None
        return None

    def summary(self, fell: bool) -> dict:
        t_end = self.t[-1] if self.t else 0.0
        out = {
            "torque_saturation_fraction": {
                self.names[i]: round(float(self.sat_steps[i] / max(self.steps, 1)), 4)
                for i in range(TRACKED_JOINTS) if self.sat_steps[i]
            },
            "joint_limit_fraction": {
                self.names[i]: round(float(self.limit_steps[i] / max(self.steps, 1)), 4)
                for i in range(TRACKED_JOINTS) if self.limit_steps[i]
            },
            "peak_torque_ratio": {
                self.names[i]: round(float(self.peak_torque[i] / self.limit[i]), 3)
                for i in range(TRACKED_JOINTS) if self.peak_torque[i] >= 0.8 * self.limit[i]
            },
            "min_cop_margin_m": round(min(self.cop_margins), 4) if self.cop_margins else None,
            "max_lateral_foothold_error_m": round(max(self.lateral_errors), 3) if self.lateral_errors else 0.0,
            "max_tilt_deg": round(self.max_tilt, 1),
            "max_dcm_error_m": round(self.max_dcm_error, 3),
            "qp_failed_steps": dict(self.qp_failures),
        }
        if fell:
            onsets = []
            for name, series in self.flags.items():
                onset = self._onset(series, t_end - WINDOW_S, t_end)
                if onset is not None:
                    onsets.append((onset, name))
            onsets.sort()
            out["fall_label"] = onsets[0][1] if onsets else "none_detected"
            out["fall_sequence"] = [f"{t:.2f} {n}" for t, n in onsets[:6]]
        return out


def _print_state(sim: Simulation) -> None:
    ctrl = sim.controller
    g = ctrl.last_gait
    tel = ctrl.telemetry
    feet = ctrl.robot.foot_center_positions()
    gs = ctrl.gait
    steps = " ".join(
        f"[{i}:{gs._steps[i].leg}@{gs._steps[i].pos[0]:.2f},z{(gs._steps[i].z or 0):.2f}{'S' if gs._steps[i].stair else ''}]"
        for i in range(gs.step_index, gs.step_index + 4) if i in gs._steps
    )
    des = np.asarray(tel.desired_forces)
    sol = np.asarray(tel.contact_forces)
    q = sim.data.qpos[7:]
    tau = np.asarray(tel.torque)
    sw = g.swing
    swing = (f"swing={'L' if sw.leg == 0 else 'R'}{sw.s:.2f} des=({sw.des_pos[0]:.2f},{sw.des_pos[2]:.3f})"
             if sw.active else "DS")
    print(
        f"t={sim.data.time:5.2f} tau={gs._tau:.2f}/{gs._current_duration(0):.2f} "
        f"com=({tel.com[0]:.2f},{tel.com[1]:+.2f},{tel.com[2]:.3f}) ref=({tel.com_reference[0]:.2f},"
        f"{tel.com_reference[1]:+.2f},{tel.com_reference[2]:.3f}) "
        f"feet=({feet[0,0]:.2f},{feet[0,1]:+.2f},{feet[0,2]:.3f} | {feet[1,0]:.2f},{feet[1,1]:+.2f},{feet[1,2]:.3f}) "
        f"mpcFz=({des[:4,2].sum():.0f},{des[4:,2].sum():.0f}) wbcFz=({sol[:4,2].sum():.0f},{sol[4:,2].sum():.0f}) "
        f"knee=({q[3]:.2f},{q[9]:.2f}) ankP=({q[4]:.2f},{q[10]:.2f}) tauK=({tau[3]:.0f},{tau[9]:.0f}) "
        f"tauA=({tau[4]:.0f},{tau[10]:.0f}) rpy=({np.degrees(tel.rpy[0]):+.0f},{np.degrees(tel.rpy[1]):+.0f}) "
        f"mask={''.join('1' if m else '0' for m in tel.contact_mask)} {swing} {steps}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="stairs_regular")
    parser.add_argument("--riser", type=float, default=0.10)
    parser.add_argument("--speed", type=float, default=0.15)
    parser.add_argument("--from", dest="t0", type=float, default=5.5)
    parser.add_argument("--to", dest="t1", type=float, default=9.0)
    parser.add_argument("--every", type=float, default=0.04)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--belief", choices=("oracle", "empty"), default="oracle")
    args = parser.parse_args()
    terrain = load_scene_terrain(args.scene)
    for flight in terrain:
        flight["riser"] = args.riser
    sim = Simulation(controller_config(), terrain=terrain)
    sim.reset()
    ctrl = sim.controller
    if args.belief == "oracle":
        ctrl.terrain.set_volumes(sim.plant_terrain.volumes)
    goal, _ = landing_goal(terrain[0])
    while sim.data.time < 0.5:
        sim.step()
    ctrl.set_task_targets({"gait.mode": "walk", "gait.speed": args.speed,
                           "gait.goal_x": float(goal[0]), "gait.goal_y": float(goal[1]),
                           "gait.goal_tolerance": 0.2})
    diag = FallAttribution(sim)
    next_t = args.t0
    while sim.data.time < args.t1 and not sim.fell:
        sim.step()
        diag.update()
        if not args.quiet:
            for e in ctrl.terrain_events:
                print("   EVENT", {k: (np.round(v, 3).tolist() if isinstance(v, (list, np.ndarray)) else v)
                                   for k, v in e.items()})
        ctrl.terrain_events.clear()
        if args.quiet or sim.data.time < next_t:
            continue
        next_t += args.every
        _print_state(sim)
    print("fell" if sim.fell else "up", round(float(sim.data.time), 2), np.round(sim.data.qpos[:3], 3))
    print(json.dumps(diag.summary(sim.fell), indent=1))


if __name__ == "__main__":
    main()
