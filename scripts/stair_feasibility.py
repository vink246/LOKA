"""Static feasibility of the G1 step-up pose (plan checkpoint K1).

For a trailing foot on the floor and a lead foot on a tread ``riser`` up and
``step`` ahead, solve leg inverse kinematics for a CoM placed a fraction of
the way toward the lead foot at a given height, with a given torso (waist)
pitch. Then find, by linear program over the eight sole-corner forces, the
smallest achievable peak of |torque| / limit over the leg and waist joints
in static equilibrium (gravity only, friction 0.5). A pose is feasible when
the legs reach, every joint is inside its range, and that ratio is <= 1.

Three phases: mid transfer (CoM halfway), end of transfer (CoM over the
lead foot, both feet down), and lead-leg single support.

    python scripts/stair_feasibility.py
    python scripts/stair_feasibility.py --riser 0.15 --step 0.26
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "g1" / "scene.xml"
LEG = 6
#: Leg and waist joints, the ones a stance load passes through.
LOADED = 15
MU = 0.5
REACH_TOL = 0.002
FOOT_SITES = ("left_foot_center", "right_foot_center")
CORNERS = [f"{side}_foot_c{i}" for side in ("left", "right") for i in range(4)]
#: Sole centre to toe line, along the foot [m] (MJCF sites at x=0.035 and 0.12).
TOE_X = 0.085


class Pose:
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(MODEL))
        self.data = mujoco.MjData(self.model)
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        self.q0 = self.model.key_qpos[key].copy()
        self.site = {n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n)
                     for n in FOOT_SITES + tuple(CORNERS)}
        self.ranges = self.model.jnt_range[1:1 + LOADED].copy()
        self.limit = np.abs(self.model.actuator_ctrlrange[:LOADED, 1])
        self.names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j + 1)
                      for j in range(LOADED)]
        waist = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "waist_pitch_joint")
        self.waist_pitch = int(self.model.jnt_qposadr[waist])
        self.total_mass = float(self.model.body_mass.sum())

    def _leg_ik(self, q: np.ndarray, leg: int, target: np.ndarray, rot_target: np.ndarray) -> float:
        m, d = self.model, self.data
        sid = self.site[FOOT_SITES[leg]]
        cols = slice(6 + leg * LEG, 6 + (leg + 1) * LEG)
        qcols = slice(7 + leg * LEG, 7 + (leg + 1) * LEG)
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        lo, hi = self.ranges[leg * LEG:(leg + 1) * LEG].T
        err = np.inf
        for _ in range(300):
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            pos = d.site_xpos[sid]
            rot = d.site_xmat[sid].reshape(3, 3)
            e_pos = target - pos
            e_rot = 0.5 * sum(np.cross(rot[:, i], rot_target[:, i]) for i in range(3))
            err = float(np.linalg.norm(e_pos)) + 0.05 * float(np.linalg.norm(e_rot))
            if err < 1e-5:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, sid)
            jac = np.vstack([jacp[:, cols], jacr[:, cols]])
            step = np.linalg.solve(jac.T @ jac + 1e-4 * np.eye(LEG), jac.T @ np.concatenate([e_pos, e_rot]))
            norm = float(np.linalg.norm(step))
            if norm > 0.05:
                step *= 0.05 / norm
            q[qcols] = np.clip(q[qcols] + step, lo, hi)
        return err

    def solve(self, feet: np.ndarray, rots: list[np.ndarray], com_target: np.ndarray,
              waist_pitch: float):
        """Pelvis placement and leg IK so the CoM lands on ``com_target``."""
        m, d = self.model, self.data
        q = self.q0.copy()
        q[self.waist_pitch] = waist_pitch
        q[0:2] = com_target[:2]
        q[2] = com_target[2] + 0.07
        errs = (np.inf, np.inf)
        for _ in range(30):
            errs = tuple(self._leg_ik(q, leg, feet[leg], rots[leg]) for leg in (0, 1))
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            delta = com_target - d.subtree_com[0]
            if np.linalg.norm(delta) < 5e-4:
                break
            q[0:3] += 0.8 * delta
        return q, max(errs)

    def static_load(self, q: np.ndarray, planted: list[bool]):
        """Least achievable max |τ|/limit in static equilibrium, and the joint."""
        m, d = self.model, self.data
        d.qpos[:] = q
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        bias = d.qfrc_bias.copy()
        jacp = np.zeros((3, m.nv))
        jac_cols = []
        for name in CORNERS:
            mujoco.mj_jacSite(m, d, jacp, None, self.site[name])
            jac_cols.append(jacp.copy())
        jac = np.vstack(jac_cols)  # (24, nv)
        nf = jac.shape[0]
        # Variables: f (24), t.
        a_eq = np.hstack([jac[:, :6].T, np.zeros((6, 1))])
        b_eq = bias[:6]
        jj = jac[:, 6:6 + LOADED].T  # (15, 24)
        lim = self.limit[:, None]
        a_ub = [np.hstack([-jj, -lim]), np.hstack([jj, -lim])]
        b_ub = [-bias[6:6 + LOADED], bias[6:6 + LOADED]]
        for i in range(8):
            fx, fy, fz = 3 * i, 3 * i + 1, 3 * i + 2
            for axis in (fx, fy):
                for sign in (1.0, -1.0):
                    row = np.zeros(nf + 1)
                    row[axis] = sign
                    row[fz] = -MU
                    a_ub.append(row[None])
                    b_ub.append(np.zeros(1))
        bounds = []
        for i in range(8):
            down = planted[i]
            bounds += [(None, None), (None, None), (0.0, None)] if down else [(0, 0)] * 3
        bounds.append((0.0, None))
        cost = np.zeros(nf + 1)
        cost[-1] = 1.0
        res = linprog(cost, A_ub=np.vstack(a_ub), b_ub=np.concatenate(b_ub), A_eq=a_eq,
                      b_eq=b_eq, bounds=bounds, method="highs")
        if not res.success:
            return np.inf, "no_equilibrium"
        f = res.x[:nf]
        tau = bias[6:6 + LOADED] - jj @ f
        ratio = np.abs(tau) / self.limit
        worst = int(np.argmax(ratio))
        return float(ratio[worst]), self.names[worst]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--riser", type=float, nargs="+", default=[0.10])
    parser.add_argument("--step", type=float, nargs="+", default=[0.22, 0.26])
    parser.add_argument("--width", type=float, default=0.18)
    parser.add_argument("--heights", type=float, nargs="+", default=[0.58, 0.62, 0.66, 0.70],
                        help="CoM height above the lead tread [m].")
    parser.add_argument("--waist", type=float, nargs="+", default=[0.0, 0.15, 0.26],
                        help="Waist pitch (torso lean) [rad].")
    parser.add_argument("--toe", type=float, nargs="+", default=[0.0, 0.25, 0.5],
                        help="Trailing-foot toe-off pitch options [rad].")
    parser.add_argument("--out", type=Path, default=ROOT / "logs" / "stairs" / "feasibility.json")
    args = parser.parse_args()
    pose = Pose()
    rows = []
    flat = np.eye(3)
    for riser, step, height, waist in itertools.product(args.riser, args.step, args.heights, args.waist):
        lead = np.array([step, -0.5 * args.width, riser])
        trail_flat = np.array([0.0, 0.5 * args.width, 0.0])
        for phase, frac, lead_only in (("mid_transfer", 0.5, False),
                                       ("end_transfer", 1.0, False),
                                       ("lead_single", 1.0, True)):
            best = None
            for toe in ([0.0] if lead_only else args.toe):
                # Trailing foot pitched heel-up by ``toe`` about its toe line.
                axis = np.array([np.cos(toe), 0.0, -np.sin(toe)])
                trail = trail_flat + np.array([TOE_X, 0.0, 0.0]) - TOE_X * axis
                rot_trail = np.array([[np.cos(toe), 0, np.sin(toe)], [0, 1, 0],
                                      [-np.sin(toe), 0, np.cos(toe)]])
                if lead_only:
                    # The other leg is in the air, passing the lead foot at swing height.
                    trail = np.array([lead[0] - 0.10, trail_flat[1], riser + 0.07])
                xy = trail_flat[:2] + frac * (lead[:2] - trail_flat[:2])
                com = np.array([xy[0], xy[1], riser + height])
                q, reach = pose.solve(np.stack([trail, lead]), [rot_trail, flat], com, waist)
                margin = float(np.min(np.minimum(q[7:7 + LOADED] - pose.ranges[:, 0],
                                                 pose.ranges[:, 1] - q[7:7 + LOADED])))
                trail_corners = [False] * 4 if lead_only else (
                    [True] * 4 if toe == 0.0 else [False, False, True, True])
                ratio, joint = pose.static_load(q, trail_corners + [True] * 4)
                ok = reach < REACH_TOL and margin > 1e-3 and ratio <= 1.0
                row = {"riser": riser, "step": step, "com_height": height, "waist": waist,
                       "phase": phase, "toe_off": toe, "reach_err": round(reach, 4),
                       "joint_margin": round(margin, 3), "torque_ratio": round(ratio, 3),
                       "worst": joint, "knees": [round(float(q[10]), 2), round(float(q[16]), 2)],
                       "feasible": ok}
                if best is None or (ok, -ratio) > (best["feasible"], -best["torque_ratio"]):
                    best = row
            rows.append(best)
    for r in rows:
        print(f"riser={r['riser']:.2f} step={r['step']:.2f} h={r['com_height']:.2f} waist={r['waist']:.2f} "
              f"{r['phase']:13s} toe={r['toe_off']:.1f} reach={r['reach_err']*1000:5.1f}mm "
              f"margin={r['joint_margin']:+.3f} tau/lim={r['torque_ratio']:.2f} ({r['worst']}) "
              f"knees={r['knees']} {'OK' if r['feasible'] else '--'}")
    for riser in args.riser:
        for phase in ("mid_transfer", "end_transfer", "lead_single"):
            ok = [r for r in rows if r["riser"] == riser and r["phase"] == phase and r["feasible"]]
            best = min(ok, key=lambda r: r["torque_ratio"]) if ok else None
            print(f"[K1] riser {riser:.2f} {phase}: "
                  + (f"{len(ok)} feasible; best tau/lim {best['torque_ratio']:.2f} at h={best['com_height']}, "
                     f"waist={best['waist']}, step={best['step']}" if best else "NO feasible pose"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
