"""Flat-floor bit-identity check for terrain-mode changes.

Terrain mode only runs when the belief holds a volume, so on a flat floor
the controller must reproduce a stored trace exactly. The scenario walks to
a goal off-axis (so the walk turns), stops, and stands, with no terrain.

    python scripts/flat_regress.py                    # compare with the reference
    python scripts/flat_regress.py --record           # rewrite the reference
    python scripts/flat_regress.py --record --root logs/head_src   # record from another tree

``--root`` puts another checkout's ``loka`` package first on the path, e.g.
one extracted with ``git archive HEAD loka | tar -x -C logs/head_src``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tests" / "data" / "flat_regress_reference.npz"
DURATION = 20.0
GOAL = (1.6, 0.5)
SPEED = 0.2
#: Control steps between samples.
EVERY = 10


def run(root: Path) -> np.ndarray:
    sys.path.insert(0, str(root))
    from loka.control.locomotion import LocomotionConfig
    from loka.sim import Simulation

    import loka

    assert Path(loka.__file__).resolve().is_relative_to(root.resolve()), loka.__file__
    sim = Simulation(LocomotionConfig.from_yaml(root / "loka" / "config" / "g1.yaml"))
    sim.reset()
    ctrl = sim.controller
    rows = []
    step = 0
    while sim.data.time < DURATION:
        if step == 250:
            ctrl.set_task_targets({
                "gait.mode": "walk",
                "gait.speed": SPEED,
                "gait.goal_x": GOAL[0],
                "gait.goal_y": GOAL[1],
                "gait.goal_tolerance": 0.15,
            })
        sim.step()
        step += 1
        if step % EVERY == 0:
            rows.append(np.concatenate([[sim.data.time], sim.data.qpos, sim.data.qvel, sim.torque]))
    return np.asarray(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    trace = run(args.root.resolve())
    if args.record:
        REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(REFERENCE, trace=trace)
        print(f"[flat] recorded {trace.shape} from {args.root} to {REFERENCE}")
        return
    ref = np.load(REFERENCE)["trace"]
    if ref.shape != trace.shape:
        print(f"[flat] FAIL shape {trace.shape} vs reference {ref.shape}")
        sys.exit(1)
    diff = float(np.abs(ref - trace).max())
    final = trace[-1, 1:4]
    print(f"[flat] max abs diff {diff:.3e}; final pelvis {np.round(final, 3)}")
    sys.exit(0 if diff == 0.0 else 1)


if __name__ == "__main__":
    main()
