"""Gate 1: the low-level controller climbs stairs it already believes in.

No orchestrator. The terrain belief is the plant's own stairs (``oracle``)
or empty (``empty``, the flat-floor controller). The robot walks to the
middle of the top landing.

    python scripts/stairs_gate.py --riser 0.10 --seeds 0 1 2 3 4
    python scripts/stairs_gate.py --riser 0.15 --seeds 0 --belief empty
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loka.control.terrain import stairs_volumes  # noqa: E402
from loka.harness.stochastic import apply_seeded_state_noise  # noqa: E402
from loka.scene_props import load_scene_terrain  # noqa: E402
from loka.sim import Simulation  # noqa: E402


CONFIG = Path(__file__).resolve().parents[1] / "loka" / "config" / "g1.yaml"


def controller_config():
    """The controller configuration the evaluation harness runs."""
    from loka.control.locomotion import LocomotionConfig

    return LocomotionConfig.from_yaml(CONFIG)


def landing_goal(flight: dict) -> tuple[np.ndarray, float]:
    top = stairs_volumes(flight)[-1]
    return top.center.copy(), float(top.top)


def run(scene: str, riser: float | None, seed: int, belief: str, speed: float,
        timeout: float, noise: float, verbose: bool, diagnose=None, goal_xy=None) -> dict:
    """One gate run. ``diagnose(sim)`` builds a per-step recorder with
    ``update()`` and ``summary(fell)``, e.g. ``stairs_trace.FallAttribution``."""
    terrain = load_scene_terrain(scene)
    if riser is not None:
        for flight in terrain:
            flight["riser"] = float(riser)
    sim = Simulation(controller_config(), terrain=terrain)
    sim.reset()
    apply_seeded_state_noise(sim.model, sim.data, seed, noise)
    ctrl = sim.controller
    if belief == "oracle":
        ctrl.terrain.set_volumes(sim.plant_terrain.volumes)
    goal, top_z = landing_goal(terrain[0])
    if goal_xy is not None:
        # Success is standing at this point, on whatever surface is there.
        goal = np.asarray(goal_xy, dtype=float).reshape(2)
        top_z = float(sim.plant_terrain.height_at(goal))
    t0 = time.perf_counter()
    while sim.data.time < 0.5:
        sim.step()
    ctrl.set_task_targets({
        "gait.mode": "walk",
        "gait.speed": float(speed),
        "gait.goal_x": float(goal[0]),
        "gait.goal_y": float(goal[1]),
        "gait.goal_tolerance": 0.20,
    })
    outcome = "timeout"
    max_z = 0.0
    held = None
    events = []
    next_print = 1.0
    recorder = diagnose(sim) if diagnose is not None else None
    while sim.data.time < timeout:
        sim.step()
        if recorder is not None:
            recorder.update()
        events.extend(ctrl.terrain_events)
        ctrl.terrain_events.clear()
        feet = ctrl.robot.foot_center_positions()
        max_z = max(max_z, float(feet[:, 2].min()))
        if sim.fell:
            outcome = "fell"
            break
        on_top = (
            float(np.linalg.norm(sim.data.qpos[:2] - goal)) <= 0.20
            and float(feet[:, 2].min()) >= top_z - 0.02
        )
        if on_top:
            held = held if held is not None else float(sim.data.time)
            if sim.data.time - held >= 1.0:
                outcome = "success"
                break
        else:
            held = None
        if verbose and sim.data.time >= next_print:
            next_print += 1.0
            g = ctrl.last_gait
            print(
                f"  t={sim.data.time:5.1f} pelvis=({sim.data.qpos[0]:+.2f},{sim.data.qpos[1]:+.2f},"
                f"{sim.data.qpos[2]:.3f}) feet_z=({feet[0,2]:.3f},{feet[1,2]:.3f}) "
                f"leg_z=({ctrl._leg_z[0]:.3f},{ctrl._leg_z[1]:.3f}) step={ctrl.gait.step_index} "
                f"walking={g.walking}",
                flush=True,
            )
    touchdowns = [e for e in events if e["kind"] == "touchdown"]
    early = [e for e in events if e["kind"] == "early_contact"]
    diagnosis = recorder.summary(outcome == "fell") if recorder is not None else None
    return {
        "scene": scene,
        "riser": riser,
        "seed": seed,
        "belief": belief,
        "outcome": outcome,
        "t_end": round(float(sim.data.time), 2),
        "highest_both_feet_z": round(max_z, 3),
        "top_z": top_z,
        "touchdowns": len(touchdowns),
        "early_contacts": len(early),
        "max_touchdown_error": round(max(
            [abs(e["measured_z"] - e["believed_z"]) for e in touchdowns if e.get("believed_z") is not None]
            or [0.0]
        ), 3),
        "wall_s": round(time.perf_counter() - t0, 1),
        **({"diagnosis": diagnosis} if diagnosis is not None else {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="stairs_regular")
    parser.add_argument("--riser", type=float, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--belief", choices=("oracle", "empty"), default="oracle")
    parser.add_argument("--speed", type=float, default=0.15)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--noise", type=float, default=0.005)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--goal", type=float, nargs=2, default=None,
                        help="Walk to this world xy instead of the first flight's landing.")
    parser.add_argument("--diagnose", action="store_true",
                        help="Attach stairs_trace.FallAttribution and report the fall label.")
    parser.add_argument("--terrain-stack", choices=("heuristic", "classical"), default="classical",
                        help="Terrain-mode controller variant (loka.control.terrain.TERRAIN_STACK).")
    args = parser.parse_args()
    from loka.control import terrain

    terrain.TERRAIN_STACK = args.terrain_stack
    diagnose = None
    if args.diagnose:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from stairs_trace import FallAttribution as diagnose
    rows = []
    for seed in args.seeds:
        row = run(args.scene, args.riser, seed, args.belief, args.speed, args.timeout,
                  args.noise, not args.quiet, diagnose=diagnose, goal_xy=args.goal)
        print(json.dumps(row), flush=True)
        rows.append(row)
    wins = sum(r["outcome"] == "success" for r in rows)
    print(f"[gate] {wins}/{len(rows)} reached the landing")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
