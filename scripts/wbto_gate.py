"""Gate for the whole-body trajectory-optimisation terrain controller (prototype).

The counterpart of ``stairs_gate.py``: no orchestrator, the terrain belief is
the plant's own geometry (optionally distorted), and the robot walks to a
goal on the terrain. The existing controller stands for the first 0.5 s; the
whole-body MPC (``loka/control/wbto``) then plans a contact sequence on the
belief and tracks it. Success is the test ``stairs_gate.py`` uses: pelvis
within 0.20 m of the goal with both soles on the goal surface for 1 s.

Runs in the environment that has ``pinocchio`` and ``crocoddyl``:

    .venv-croc/bin/python scripts/wbto_gate.py --scene stairs_regular --riser 0.10 --seeds 0 1 2
    .venv-croc/bin/python scripts/wbto_gate.py --scene platform_gap --gap 0.30 --goal 2.5 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loka.control.terrain import TerrainBelief, stairs_volumes  # noqa: E402
from loka.control.wbto.controller import MpcParams, WholeBodyMpc  # noqa: E402
from loka.control.wbto.footsteps import LEFT, RIGHT, StepParams, describe, plan_contacts  # noqa: E402
from loka.control.wbto.model import WholeBodyModel  # noqa: E402
from loka.control.wbto.ocp import OcpBuilder, OcpParams, build_knots  # noqa: E402
from loka.harness.stochastic import apply_seeded_state_noise  # noqa: E402
from loka.scene_props import load_scene_terrain  # noqa: E402
from loka.sim import Simulation  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "loka" / "config" / "g1.yaml"


def scene_terrain(scene: str, riser: float | None, gap: float | None) -> list[dict]:
    terrain = load_scene_terrain(scene)
    if riser is not None:
        for flight in terrain:
            flight["riser"] = float(riser)
    if gap is not None:
        if len(terrain) < 2:
            raise ValueError("--gap needs a scene with two pads")
        near = stairs_volumes(terrain[0])[-1]
        terrain[1]["x"] = float(near.center[0] + 0.5 * near.size[0] + gap)
    return terrain


def believed_volumes(terrain: list[dict], args) -> list:
    """The belief: the plant's terrain, with any requested estimation error."""
    volumes = []
    for spec in terrain:
        spec = dict(spec)
        if args.belief_riser is not None:
            spec["riser"] = float(args.belief_riser)
        if args.belief_tread is not None:
            spec["tread"] = float(args.belief_tread)
        spec["x"] = float(spec.get("x", 0.0)) + args.belief_shift
        volumes.extend(stairs_volumes(spec))
    return volumes


def run(args, seed: int) -> dict:
    from loka.control.locomotion import LocomotionConfig

    terrain = scene_terrain(args.scene, args.riser, args.gap)
    sim = Simulation(LocomotionConfig.from_yaml(CONFIG), terrain=terrain)
    believed = believed_volumes(terrain, args)
    sim.controller.terrain.set_volumes(believed)  # the stand phase needs the ground height
    sim.reset()
    apply_seeded_state_noise(sim.model, sim.data, seed, args.noise)
    if args.goal is not None:
        goal = np.asarray(args.goal, dtype=float)
    else:
        goal = stairs_volumes(terrain[0])[-1].center.copy()
    top_z = float(sim.plant_terrain.height_at(goal))

    while sim.data.time < 0.5:
        sim.step()

    belief = TerrainBelief()
    belief.set_volumes(believed)
    wb = WholeBodyModel(sim.config.model_path)
    nq, nv = wb.mj_model.nq, wb.mj_model.nv
    x0 = wb.state_from_mujoco(sim.data.qpos[:nq], sim.data.qvel[:nv])
    feet = np.array([m.translation for m in wb.foot_placements(x0[: wb.nq])])
    step_params = StepParams(stride=args.stride, reach_max=args.reach)
    plan = plan_contacts(belief, feet, goal, step_params)
    if args.verbose:
        print("\n".join(describe(plan)), flush=True)
    builder = OcpBuilder(wb, OcpParams(**_overrides(args.ocp)))
    knots = build_knots(plan, wb.com(x0[: wb.nq]), builder.p)
    mpc = WholeBodyMpc(
        wb, builder, knots, plan.heading,
        MpcParams(horizon=args.horizon, max_iter=args.iters, feedback=not args.no_feedback,
                  latency=args.latency, feedback_scale=args.feedback_scale, predict=not args.no_predict),
    )

    control_dt = sim.config.control_dt
    per_knot = max(1, int(round(builder.p.dt / control_dt)))
    feet_sites = [
        mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in ("left_foot_center", "right_foot_center")
    ]
    foot_bodies = [int(sim.model.site_bodyid[site]) for site in feet_sites]
    t_start = float(sim.data.time)
    deadline = t_start + plan.duration + args.hold
    wall0 = time.perf_counter()
    outcome, held, max_z, torque_peak = "timeout", None, 0.0, 0.0
    log_q = []
    tick = 0
    next_print = t_start
    while sim.data.time < deadline:
        qpos, qvel = sim.data.qpos[:nq].copy(), sim.data.qvel[:nv].copy()
        if tick % per_knot == 0:
            mpc.update(wb.state_from_mujoco(qpos, qvel))
        tau = mpc.actuator_torque(qpos, qvel, (tick % per_knot) / per_knot)
        tick += 1
        torque_peak = max(torque_peak, float(np.max(np.abs(tau) / wb.mj_model.actuator_ctrlrange[:, 1])))
        for _ in range(sim.steps_per_control):
            sim.data.ctrl[: wb.mj_model.nu] = tau
            mujoco.mj_step(sim.model, sim.data)
        soles = sim.data.site_xpos[feet_sites]
        if args.save is not None and tick % 5 == 0:
            knot = mpc.knot(mpc.tick)
            target = knot.swing[1].translation if knot.swing is not None else np.full(3, np.nan)
            log_q.append(np.concatenate([
                [sim.data.time], sim.data.qpos[:nq], soles.ravel(), foot_loads(sim, foot_bodies),
                [float(LEFT in knot.support), float(RIGHT in knot.support)], target,
                np.abs(tau) / wb.mj_model.actuator_ctrlrange[:, 1],
            ]))
        max_z = max(max_z, float(soles[:, 2].min()))
        if sim.fell:
            outcome = "fell"
            break
        on_goal = (
            float(np.linalg.norm(sim.data.qpos[:2] - goal)) <= 0.20
            and float(soles[:, 2].min()) >= top_z - 0.02
        )
        if on_goal:
            held = held if held is not None else float(sim.data.time)
            if sim.data.time - held >= 1.0:
                outcome = "success"
                break
        else:
            held = None
        if args.verbose and sim.data.time >= next_print:
            next_print += 1.0
            knot = mpc.knot(mpc.tick)
            print(
                f"  t={sim.data.time:5.1f} pelvis=({sim.data.qpos[0]:+.2f},{sim.data.qpos[1]:+.2f},"
                f"{sim.data.qpos[2]:.3f}) soles_z=({soles[0, 2]:.3f},{soles[1, 2]:.3f}) "
                f"step={knot.step}/{len(plan.steps)} solve={np.mean(mpc.solve_ms[-50:]):.1f}ms",
                flush=True,
            )
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save.with_name(f"{args.save.stem}_seed{seed}.npz"),
            log=np.array(log_q), terrain=json.dumps(terrain),
        )
    solves = np.array(mpc.solve_ms[1:] or [0.0])
    return {
        "scene": args.scene,
        "riser": args.riser,
        "gap": args.gap,
        "seed": seed,
        "latency_knots": args.latency,
        "ocp": args.ocp,
        "belief": {"riser": args.belief_riser, "tread": args.belief_tread, "shift": args.belief_shift},
        "outcome": outcome,
        "t_end": round(float(sim.data.time), 2),
        "steps": len(plan.steps),
        "plan_s": round(plan.duration, 2),
        "highest_both_feet_z": round(max_z, 3),
        "top_z": top_z,
        "goal_error_m": round(float(np.linalg.norm(sim.data.qpos[:2] - goal)), 3),
        "yaw_deg": round(float(np.degrees(_yaw(sim.data.qpos[3:7]))), 1),
        "torque_peak_frac": round(torque_peak, 2),
        "cold_solve_ms": round(mpc.solve_ms[0], 1),
        "solve_ms_mean": round(float(solves.mean()), 1),
        "solve_ms_p95": round(float(np.percentile(solves, 95)), 1),
        "solver_failures": mpc.failures,
        "wall_s": round(time.perf_counter() - wall0, 1),
    }


def foot_loads(sim, bodies) -> np.ndarray:
    """Normal force the plant puts on each foot [N] (for the trace only)."""
    loads = np.zeros(len(bodies))
    wrench = np.zeros(6)
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        for k, body in enumerate(bodies):
            if body in (sim.model.geom_bodyid[contact.geom1], sim.model.geom_bodyid[contact.geom2]):
                mujoco.mj_contactForce(sim.model, sim.data, i, wrench)
                loads[k] += wrench[0]
    return loads


def _overrides(pairs: list[str]) -> dict:
    """``name=value`` pairs for scalar fields of :class:`OcpParams`."""
    return {name: float(value) for name, value in (pair.split("=", 1) for pair in pairs)}


def _yaw(quat) -> float:
    w, x, y, z = (float(v) for v in quat)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default="stairs_regular")
    parser.add_argument("--riser", type=float, default=None, help="Override every riser of the scene [m].")
    parser.add_argument("--gap", type=float, default=None, help="Width of the gap between two pads [m].")
    parser.add_argument("--goal", type=float, nargs=2, default=None,
                        help="World xy to stand at (default: middle of the first flight's landing).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--noise", type=float, default=0.005)
    parser.add_argument("--stride", type=float, default=StepParams.stride)
    parser.add_argument("--reach", type=float, default=StepParams.reach_max)
    parser.add_argument("--horizon", type=int, default=MpcParams.horizon)
    parser.add_argument("--iters", type=int, default=MpcParams.max_iter)
    parser.add_argument("--latency", type=int, default=1, choices=(0, 1),
                        help="Knot periods (20 ms) between the state measurement and its solution taking effect.")
    parser.add_argument("--feedback-scale", type=float, default=MpcParams.feedback_scale)
    parser.add_argument("--ocp", nargs="*", default=[], metavar="NAME=VALUE",
                        help="Override scalar OcpParams fields, e.g. w_swing=5e3.")
    parser.add_argument("--no-predict", action="store_true")
    parser.add_argument("--no-feedback", action="store_true", help="Feed-forward torque only between solves.")
    parser.add_argument("--hold", type=float, default=3.0, help="Time allowed after the plan ends [s].")
    parser.add_argument("--belief-riser", type=float, default=None, help="Riser the controller believes [m].")
    parser.add_argument("--belief-tread", type=float, default=None, help="Tread the controller believes [m].")
    parser.add_argument("--belief-shift", type=float, default=0.0,
                        help="Shift of the believed terrain along x [m].")
    parser.add_argument("--save", type=Path, default=None, help="Write a 100 Hz trace of the run to this .npz stem.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--quiet", dest="verbose", action="store_false")
    args = parser.parse_args()
    rows = []
    for seed in args.seeds:
        row = run(args, seed)
        print(json.dumps(row), flush=True)
        rows.append(row)
    wins = sum(r["outcome"] == "success" for r in rows)
    print(f"[wbto gate] {wins}/{len(rows)} reached the goal")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
