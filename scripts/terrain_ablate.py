"""Turn terrain-mode components off one at a time and rerun a gate case.

Each ablation monkeypatches one piece back to its flat-floor behaviour, so
a component that breaks walking shows up as the one whose removal helps.

    python scripts/terrain_ablate.py --scene curb --riser 0.001
    python scripts/terrain_ablate.py --only toe_off dampers --riser 0.06
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _apply(name: str) -> None:
    from loka.control import gait, locomotion

    L = locomotion.LocomotionController
    if name == "none":
        return
    if "=" in name and name.split(".", 1)[0] in ("gait", "locomotion", "terrain", "stabilizer"):
        # ``module.CONSTANT=value``, e.g. locomotion.TERRAIN_WALK_HEIGHT=0.62
        from loka.control import stabilizer, terrain

        key, value = name.split("=", 1)
        module, const = key.split(".", 1)
        mod = {"gait": gait, "locomotion": locomotion, "terrain": terrain,
               "stabilizer": stabilizer}[module]
        old = getattr(mod, const)
        setattr(mod, const, type(old)(value) if not isinstance(old, bool) else value == "True")
        return
    if "=" in name:
        # ``section.field=value`` on the controller config, e.g. wbc.weight_force=0.5
        import stairs_gate

        key, value = name.split("=", 1)
        section, field = key.split(".", 1)
        base = stairs_gate.controller_config

        def patched():
            cfg = base()
            setattr(getattr(cfg, section), field, float(value))
            return cfg
        stairs_gate.controller_config = patched
        return
    if name == "plan_velocity":
        locomotion.STAIR_PLAN_VELOCITY = False
        return
    if name == "contact_mask":
        def flat_mask(self, heights):
            mask = heights < self._ground_height + locomotion.CONTACT_HEIGHT_TOLERANCE
            return mask
        L._terrain_contact_mask = flat_mask
    elif name == "toe_off":
        L._toe_off = lambda self, mask, *a, **k: mask
    elif name == "dampers":
        L._joint_limit_dampers = lambda self, *a, **k: None
    elif name == "reach_clamp":
        L._reach_clamp = lambda self, ref, vel, *a, **k: (ref, vel)
    elif name == "geometric_knees":
        L._geometric_knees = lambda self, err: self._knee_bias(self.nominal_height)
        locomotion.TERRAIN_KNEE_W = locomotion.STANCE_KNEE_W
    elif name == "crouch":
        locomotion.TERRAIN_WALK_HEIGHT = 10.0
    elif name == "dcm_3d":
        gait.DCM_3D = False
    elif name == "stance_width":
        gait.TERRAIN_STANCE_WIDTH = 10.0
    elif name == "early_contact":
        L._check_early_contact = lambda self, feet: None
    else:
        raise ValueError(name)


ABLATIONS = ("none", "contact_mask", "toe_off", "dampers", "reach_clamp", "geometric_knees",
             "crouch", "dcm_3d", "stance_width", "early_contact")


def _one(job):
    names, scene, riser, seed = job
    speed = 0.15
    for name in names:
        if name.startswith("speed="):
            speed = float(name.split("=", 1)[1])
            continue
        _apply(name)
    from stairs_gate import run
    from stairs_trace import FallAttribution

    row = run(scene, riser, seed, "oracle", speed, 30.0, 0.005, False, diagnose=FallAttribution)
    return "+".join(names), seed, row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="curb")
    parser.add_argument("--riser", type=float, default=0.001)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--combo", nargs="+", default=None,
                        help="Apply all of these together as one extra case.")
    args = parser.parse_args()
    cases = [[n] for n in (args.only or ABLATIONS)]
    if args.combo:
        cases.append(args.combo)
    jobs = [(c, args.scene, args.riser, s) for c in cases for s in args.seeds]
    with ProcessPoolExecutor(max_workers=min(len(jobs), 22)) as pool:
        for name, seed, row in pool.map(_one, jobs):
            d = row.get("diagnosis") or {}
            print(json.dumps({"ablate": name, "seed": seed, "outcome": row["outcome"],
                              "t": row["t_end"], "z": row["highest_both_feet_z"],
                              "label": d.get("fall_label")}), flush=True)


if __name__ == "__main__":
    main()
