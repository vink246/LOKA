#!/usr/bin/env python3
"""Set the cost_detector ablation's thresholds from the healthy plant.

The residual detector's thresholds sit just above the noise floor of the
healthy plant. This script applies the same rule to the MPC tracking cost: it
runs the nominal plant with the orchestrator off (stand, then walk to a goal)
for several seeds, records the windowed tracking cost the detector would see,
and sets each threshold to a high percentile of it times a margin. It then
runs each goto_goal perturbation of the eval suite and reports how far the
faulted cost rises above that threshold, so the detector's sensitivity can be
compared with the residual detector's.

    python scripts/calibrate_cost_detector.py                 # report only
    python scripts/calibrate_cost_detector.py --write         # also update loka_ablations.yaml
    python scripts/calibrate_cost_detector.py --seeds 5 --percentile 99.5 --margin 1.5

CPU only; a few minutes on a laptop.
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
from loka.agent.runtime import LokaConfig, LokaRuntime  # noqa: E402
from loka.harness.config import load_suite_yaml  # noqa: E402
from loka.harness.perturbations import make_perturbation  # noqa: E402
from loka.harness.stochastic import apply_seeded_state_noise, trial_seed  # noqa: E402
from loka.sim import Simulation  # noqa: E402

#: Placeholder thresholds while recording: the detector must never trip.
_NEVER = 1e12


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


def _record(runtime: LokaRuntime, seconds: float, samples: dict, *, after: float = 0.0) -> bool:
    """Step for ``seconds``; append the windowed cost to samples[stand|walk]. False on a fall."""
    obs = runtime.effort_observer
    end = float(runtime.sim.data.time) + seconds
    while float(runtime.sim.data.time) < end:
        frame = runtime.step()
        if runtime.sim.fell:
            return False
        if obs.ready and float(frame["time"]) >= after and obs.trigger_value is not None:
            samples["walk" if obs.is_walking else "stand"].append(float(obs.trigger_value))
    return True


def _goal(runtime: LokaRuntime, goal, speed: float, radius: float) -> None:
    runtime.controller.set_task_targets(
        {
            "gait.mode": "walk",
            "gait.speed": float(speed),
            "gait.goal_x": float(goal[0]),
            "gait.goal_y": float(goal[1]),
            "gait.goal_tolerance": float(radius),
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default=str(ROOT / "loka" / "config" / "eval_suite.yaml"))
    parser.add_argument("--ablation-config", default=str(DEFAULT_ABLATION_PATH))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0, help="Base seed (trial seeds follow the harness).")
    parser.add_argument("--init-noise", type=float, default=0.005)
    parser.add_argument("--stand-s", type=float, default=6.0)
    parser.add_argument("--walk-s", type=float, default=14.0)
    parser.add_argument("--percentile", type=float, default=99.5)
    parser.add_argument("--margin", type=float, default=1.5)
    parser.add_argument("--after-fault-s", type=float, default=6.0, help="Seconds observed after each perturbation.")
    parser.add_argument("--no-perturbations", action="store_true")
    parser.add_argument("--write", action="store_true", help="Write the thresholds into the ablation config.")
    parser.add_argument("--out", default=str(ROOT / "results" / "cost_calibration"))
    args = parser.parse_args(argv)

    params = dict(load_ablation_file(args.ablation_config)["ablations"].get("cost_detector") or {})
    suite = load_suite_yaml(args.suite)
    goto = next((t for t in suite.tasks if t.type == "goto_goal"), None)
    goal = (goto.raw.get("goal") or {}) if goto else {}
    goal_xy = goal.get("xy", [3.0, 0.0])
    speed = float(goal.get("speed", 0.15))
    radius = float(goal.get("radius", 0.2))

    nominal = {"stand": [], "walk": []}
    for trial in range(args.seeds):
        seed = trial_seed(args.seed, trial)
        runtime = _runtime(params, seed, args.init_noise)
        ok = _record(runtime, args.stand_s, nominal)
        _goal(runtime, goal_xy, speed, radius)
        ok = ok and _record(runtime, args.walk_s, nominal)
        print(
            f"[calibrate] nominal seed {seed}: {len(nominal['stand'])} standing and "
            f"{len(nominal['walk'])} walking samples so far{'' if ok else ' (FELL)'}"
        )
    thresholds = {}
    report = {"percentile": args.percentile, "margin": args.margin, "nominal": {}, "perturbations": {}}
    for mode in ("stand", "walk"):
        values = np.asarray(nominal[mode], dtype=float)
        if values.size == 0:
            print(f"[calibrate] no {mode} samples; lengthen --{mode}-s", file=sys.stderr)
            return 1
        level = float(np.percentile(values, args.percentile))
        thresholds[mode] = float(f"{level * args.margin:.4g}")
        report["nominal"][mode] = {
            "samples": int(values.size),
            "median": float(np.median(values)),
            f"p{args.percentile:g}": level,
            "max": float(values.max()),
            "threshold": thresholds[mode],
        }
        print(
            f"[calibrate] {mode}: median {np.median(values):.4g}, p{args.percentile:g} {level:.4g}, "
            f"max {values.max():.4g} -> threshold {thresholds[mode]:.4g}"
        )

    if goto is not None and not args.no_perturbations:
        fault_time = float(goto.raw.get("perturbation_time_s", 5.0))
        for spec in goto.perturbations:
            if spec.kind == "none":
                continue
            runtime = _runtime(params, trial_seed(args.seed, 0), args.init_noise)
            _goal(runtime, goal_xy, speed, radius)
            before = {"stand": [], "walk": []}
            _record(runtime, fault_time, before)
            make_perturbation(spec).apply(_Agent(runtime))
            after = {"stand": [], "walk": []}
            survived = _record(runtime, args.after_fault_s, after)
            values = np.asarray(after["walk"] + after["stand"], dtype=float)
            walk = np.asarray(after["walk"], dtype=float)
            peak = float(values.max()) if values.size else float("nan")
            ratio = float(walk.max() / thresholds["walk"]) if walk.size else None
            # Longest run above the walking threshold, as the detector's streak would count it.
            streak = best = 0
            for value in walk:
                streak = streak + 1 if value > thresholds["walk"] else 0
                best = max(best, streak)
            report["perturbations"][spec.name] = {
                "peak_cost": peak,
                "peak_over_walk_threshold": ratio,
                "longest_streak_steps": int(best),
                "would_trip": bool(best >= int(params.get("consecutive_steps", 250))),
                "fell": not survived,
            }
            print(
                f"[calibrate] {spec.name}: peak {peak:.4g} "
                f"({'n/a' if ratio is None else f'{ratio:.2f}x'} walk threshold), "
                f"longest streak {best} steps, "
                f"{'trips' if best >= int(params.get('consecutive_steps', 250)) else 'does not trip'}"
                f"{', fell' if not survived else ''}"
            )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report["thresholds"] = thresholds
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[calibrate] report {path}")
    if args.write:
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
    raise SystemExit(main())
