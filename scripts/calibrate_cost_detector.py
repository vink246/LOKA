#!/usr/bin/env python3
"""Set the cost_detector ablation's thresholds from the healthy plant.

The residual detector opens a failure episode after ``consecutive_steps``
ticks in a row above threshold, and its thresholds sit just above the healthy
plant's noise. This script applies the same rule to the MPC tracking cost: a
threshold is the smallest level the healthy plant never stays above for
``consecutive_steps`` ticks, times a margin, measured across the motions the
eval tasks use:

  quiet_stand     standing from the keyframe
  stop_mid_walk   a walk stopped by a stand command, as stand-on-fault does,
                  then standing while the stance settles
  walk_to_goal    resuming the walk after that stop (the resume LOKA commands
                  after every fix), the arrival stop and the stand
  matched_payload the same protocol with a 10 kg torso payload that the belief
                  already knows: a correct model is a healthy plant, so a fixed
                  fault must be able to fall back under the thresholds
  turn_in_place   a 1 rad turn on the spot and the stand after it; reported,
                  and used for thresholds only with --include-turns

It then checks the two things a threshold must allow:
  * recovery: with a torso payload and the correct mass written into the
    belief, the standing cost must fall back under the standing threshold, or
    no failure episode can ever close;
  * detection: for each goto_goal perturbation, whether the fault holds the
    walking cost above threshold for ``consecutive_steps`` ticks.

    python scripts/calibrate_cost_detector.py                 # report only
    python scripts/calibrate_cost_detector.py --write         # also update loka_ablations.yaml
    python scripts/calibrate_cost_detector.py --seeds 5 --margin 1.5 --include-turns

CPU only; about 10 minutes on a laptop.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loka.agent.ablations import DEFAULT_ABLATION_PATH, AblationSet, load_ablation_file  # noqa: E402
from loka.agent.faults import FaultSpec  # noqa: E402
from loka.agent.runtime import LokaConfig, LokaRuntime  # noqa: E402
from loka.harness.config import load_suite_yaml  # noqa: E402
from loka.harness.perturbations import make_perturbation  # noqa: E402
from loka.harness.stochastic import apply_seeded_state_noise, trial_seed  # noqa: E402
from loka.sim import Simulation  # noqa: E402

#: Placeholder thresholds while recording: the detector must never trip.
_NEVER = 1e12
HEALTHY_SEGMENTS = ("quiet_stand", "stop_mid_walk", "walk_to_goal", "matched_payload")
PAYLOAD_KG = 10.0


class _Agent:
    """The perturbation API (``inject``) over a bare runtime."""

    def __init__(self, runtime: LokaRuntime) -> None:
        self.runtime = runtime

    def inject(self, fault) -> None:
        if not self.runtime.inject_fault_now(fault):
            raise RuntimeError(f"fault {fault} was not applied")


def _runtime(params: dict, seed: int, init_noise: float) -> LokaRuntime:
    cost = dict(params)
    cost.update(stand_threshold=_NEVER, walk_threshold=_NEVER)
    ablations = AblationSet(names=("cost_detector",), params={"cost_detector": cost})
    sim = Simulation()
    runtime = LokaRuntime(
        sim,
        LokaConfig(enable_llm=False, enable_session_log=False, print_compressor=False, ablations=ablations),
    )
    runtime.effort_observer.printout = False
    sim.reset()
    apply_seeded_state_noise(sim.model, sim.data, seed, init_noise)
    return runtime


def _record(runtime: LokaRuntime, seconds: float) -> tuple[list[tuple[bool, float]], bool]:
    """``[(walking, windowed cost)]`` for every tick with a full window, and False on a fall."""
    obs = runtime.effort_observer
    trace = []
    end = float(runtime.sim.data.time) + seconds
    while float(runtime.sim.data.time) < end:
        runtime.step()
        if runtime.sim.fell:
            return trace, False
        if obs.ready and obs.trigger_value is not None:
            trace.append((bool(obs.is_walking), float(obs.trigger_value)))
    return trace, True


def sustained_level(trace: list[tuple[bool, float]], walking: bool, steps: int) -> float | None:
    """Highest level the cost stayed above for ``steps`` ticks in a row, in one mode.

    A threshold at or above it never opens a failure episode on this trace.
    """
    best = None
    run: list[float] = []
    for mode, value in [*trace, (None, None)]:
        if mode == walking:
            run.append(value)
            continue
        if len(run) >= steps:
            level = float(np.lib.stride_tricks.sliding_window_view(np.asarray(run), steps).min(axis=1).max())
            best = level if best is None else max(best, level)
        run = []
    return best


def _walk(runtime: LokaRuntime, goal, speed: float, radius: float) -> None:
    runtime.controller.set_task_targets(
        {
            "gait.mode": "walk",
            "gait.speed": float(speed),
            "gait.goal_x": float(goal[0]),
            "gait.goal_y": float(goal[1]),
            "gait.goal_tolerance": float(radius),
        }
    )


def _stand(runtime: LokaRuntime) -> None:
    runtime.controller.set_task_targets({"gait.mode": "stand", "gait.speed": 0.0})


def _matched_payload(runtime: LokaRuntime, delta_kg: float) -> float:
    """Add ``delta_kg`` to the plant torso and write the same mass into the belief."""
    import mujoco

    runtime.inject_fault_now(FaultSpec("mass", 0.0, {"delta_kg": delta_kg, "body": "torso_link"}))
    body = mujoco.mj_name2id(runtime.sim.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    true_mass = float(runtime.sim.model.body_mass[body])
    runtime.loka_state["mutations"].append({"type": "body", "attr": "mass", "id": body, "val": true_mass})
    return true_mass


def healthy_segments(
    params: dict, seed: int, init_noise: float, goal, speed: float, radius: float, payload_kg: float = 0.0
) -> dict:
    """Traces of a healthy plant (belief matches plant), by segment."""
    runtime = _runtime(params, seed, init_noise)
    if payload_kg:
        _matched_payload(runtime, payload_kg)
    segments = {}
    segments["quiet_stand"], _ = _record(runtime, 4.0)
    _walk(runtime, goal, speed, radius)
    walk1, _ = _record(runtime, 6.0)
    _stand(runtime)
    stop, _ = _record(runtime, 8.0)
    segments["stop_mid_walk"] = walk1[-1:] + stop  # the stand that follows a walk
    _walk(runtime, goal, speed, radius)
    to_goal, _ = _record(runtime, 18.0)
    segments["walk_to_goal"] = walk1 + to_goal
    runtime.controller.set_task_targets({"gait.mode": "walk", "gait.speed": 0.0, "gait.heading": 1.0})
    turn, _ = _record(runtime, 6.0)
    _stand(runtime)
    after, _ = _record(runtime, 6.0)
    segments["turn_in_place"] = turn + after
    return segments


def recovery_check(params: dict, seed: int, init_noise: float, goal, speed: float, radius: float, steps: int) -> dict:
    """Torso payload while walking, a stop, the correct mass written into the belief, then standing.

    This is the order an eval episode follows (fault, stand on fault, search),
    unlike matched_payload, where the belief knows the payload from the start.
    """
    runtime = _runtime(params, seed, init_noise)
    _walk(runtime, goal, speed, radius)
    _record(runtime, 5.0)
    runtime.inject_fault_now(FaultSpec("mass", 0.0, {"delta_kg": PAYLOAD_KG, "body": "torso_link"}))
    _record(runtime, 1.0)
    _stand(runtime)
    _record(runtime, 1.0)
    import mujoco

    body = mujoco.mj_name2id(runtime.sim.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    true_mass = float(runtime.sim.model.body_mass[body])
    runtime.loka_state["mutations"].append({"type": "body", "attr": "mass", "id": body, "val": true_mass})
    trace, survived = _record(runtime, 8.0)
    standing = [v for walking, v in trace if not walking]
    tail = standing[-steps:] if len(standing) >= steps else standing
    return {
        "true_mass_kg": true_mass,
        "survived": survived,
        "standing_cost_final_max": float(max(tail)) if tail else None,
        "standing_cost_median": float(np.median(standing)) if standing else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default=str(ROOT / "loka" / "config" / "eval_suite.yaml"))
    parser.add_argument("--ablation-config", default=str(DEFAULT_ABLATION_PATH))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0, help="Base seed (trial seeds follow the harness).")
    parser.add_argument("--init-noise", type=float, default=0.005)
    parser.add_argument("--margin", type=float, default=1.5)
    parser.add_argument(
        "--include-turns",
        action="store_true",
        help="Also keep the healthy turn in place under the thresholds (a much less sensitive detector).",
    )
    parser.add_argument("--after-fault-s", type=float, default=6.0, help="Seconds observed after each perturbation.")
    parser.add_argument("--no-perturbations", action="store_true")
    parser.add_argument("--write", action="store_true", help="Write the thresholds into the ablation config.")
    parser.add_argument("--out", default=str(ROOT / "results" / "cost_calibration"))
    args = parser.parse_args(argv)

    params = dict(load_ablation_file(args.ablation_config)["ablations"].get("cost_detector") or {})
    steps = int(params.get("consecutive_steps", 250))
    suite = load_suite_yaml(args.suite)
    goto = next((t for t in suite.tasks if t.type == "goto_goal"), None)
    goal_cfg = (goto.raw.get("goal") or {}) if goto else {}
    goal = goal_cfg.get("xy", [3.0, 0.0])
    speed = float(goal_cfg.get("speed", 0.15))
    radius = float(goal_cfg.get("radius", 0.2))

    used = HEALTHY_SEGMENTS + (("turn_in_place",) if args.include_turns else ())
    levels: dict[str, dict[str, list[float]]] = {}
    def collect(name_of, payload_kg: float) -> None:
        segments = healthy_segments(params, seed, args.init_noise, goal, speed, radius, payload_kg)
        for name, trace in segments.items():
            name = name_of(name)
            if name is None:
                continue
            for mode, walking in (("stand", False), ("walk", True)):
                level = sustained_level(trace, walking, steps)
                if level is not None:
                    levels.setdefault(name, {}).setdefault(mode, []).append(level)

    for trial in range(args.seeds):
        seed = trial_seed(args.seed, trial)
        collect(lambda name: name, 0.0)
        # Every motion again with a payload the belief knows; turns stay separate.
        collect(lambda name: None if name == "turn_in_place" else "matched_payload", PAYLOAD_KG)
        print(f"[calibrate] healthy seed {seed} recorded")

    report: dict = {"rule": f"max sustained ({steps} ticks) healthy cost x {args.margin}", "segments": {}}
    thresholds = {}
    for mode in ("stand", "walk"):
        floor = max(max(levels[s][mode]) for s in used if mode in levels.get(s, {}))
        thresholds[mode] = float(f"{floor * args.margin:.4g}")
    for name, modes in levels.items():
        report["segments"][name] = {mode: max(v) for mode, v in modes.items()}
        flag = "" if name in used else "   (not used for thresholds)"
        text = ", ".join(f"{mode} {max(v):.4g}" for mode, v in modes.items())
        print(f"[calibrate] sustained healthy cost, {name:14s}: {text}{flag}")
    print(f"[calibrate] thresholds: stand {thresholds['stand']}, walk {thresholds['walk']} (margin {args.margin})")
    turn = report["segments"].get("turn_in_place", {})
    if not args.include_turns and turn.get("walk", 0.0) > thresholds["walk"]:
        print(
            f"[calibrate] note: a healthy turn in place sustains {turn['walk']:.4g} > walk threshold; "
            "the cost detector will open a fault on intentional turns (pass --include-turns to avoid it)"
        )

    rec = recovery_check(params, trial_seed(args.seed, 0), args.init_noise, goal, speed, radius, steps)
    rec["recovers"] = bool(rec["survived"] and rec["standing_cost_final_max"] is not None
                           and rec["standing_cost_final_max"] <= thresholds["stand"])
    report["recovery_check"] = rec
    print(
        f"[calibrate] recovery check (torso +10 kg, belief corrected to {rec['true_mass_kg']:.2f} kg): "
        f"standing cost {rec['standing_cost_final_max']:.4g} vs stand threshold {thresholds['stand']} -> "
        f"{'recovers' if rec['recovers'] else 'DOES NOT RECOVER'}"
    )

    if goto is not None and not args.no_perturbations:
        fault_time = float(goto.raw.get("perturbation_time_s", 5.0))
        for spec in goto.perturbations:
            if spec.kind == "none":
                continue
            runtime = _runtime(params, trial_seed(args.seed, 0), args.init_noise)
            _walk(runtime, goal, speed, radius)
            _record(runtime, fault_time)
            make_perturbation(spec).apply(_Agent(runtime))
            trace, survived = _record(runtime, args.after_fault_s)
            walk_level = sustained_level(trace, True, steps)
            values = [v for walking, v in trace if walking]
            trips = walk_level is not None and walk_level > thresholds["walk"]
            report["perturbations"] = report.get("perturbations", {})
            report["perturbations"][spec.name] = {
                "peak_walking_cost": float(max(values)) if values else None,
                "sustained_walking_cost": walk_level,
                "trips": trips,
                "fell": not survived,
            }
            peak = f"{max(values):.4g}" if values else "n/a"
            print(
                f"[calibrate] {spec.name}: peak walking cost {peak}, sustained "
                f"{'n/a' if walk_level is None else f'{walk_level:.4g}'} -> "
                f"{'trips' if trips else 'does not trip'}{', fell' if not survived else ''}"
            )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report["thresholds"] = thresholds
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[calibrate] report {path}")
    if args.write:
        if not rec["recovers"]:
            print("[calibrate] not writing: a corrected model would never close its failure episode", file=sys.stderr)
            return 1
        write_thresholds(Path(args.ablation_config), thresholds["stand"], thresholds["walk"])
        print(f"[calibrate] wrote stand_threshold={thresholds['stand']} walk_threshold={thresholds['walk']} "
              f"to {args.ablation_config}")
    return 0


def write_thresholds(path: Path, stand: float, walk: float) -> None:
    """Replace the two threshold values in place, keeping the file's comments."""
    text = path.read_text(encoding="utf-8")
    for key, value in (("stand_threshold", stand), ("walk_threshold", walk)):
        pattern = re.compile(rf"^(\s+{key}:)[^\n#]*", re.MULTILINE)
        if len(pattern.findall(text)) != 1:
            raise ValueError(f"{path} must have exactly one {key} line")
        text = pattern.sub(lambda m, v=value: f"{m.group(1)} {v:.4g}", text)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main(argv=None))
