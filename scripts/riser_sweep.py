"""Capability curve: oracle-belief success against riser height.

Runs the curb and the stairs at each riser on several seeds, each run with
:class:`stairs_trace.FallAttribution`, and prints one line per
(scene, riser): successes, mean highest both-feet height, and the most
common fall label. Every controller change is measured against this.

    python scripts/riser_sweep.py
    python scripts/riser_sweep.py --scenes curb --risers 0.06 0.10 --tag knee
    python scripts/riser_sweep.py --compare logs/stairs/sweep_base.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT_DIR = ROOT / "logs" / "stairs"
#: A riser counts as reliable at this many successes out of the seeds run.
RELIABLE = 0.8


def _one(job: tuple) -> dict:
    from loka.control import terrain
    from stairs_gate import run
    from stairs_trace import FallAttribution

    scene, riser, seed, speed, timeout, stack, goal = job
    terrain.TERRAIN_STACK = stack
    return run(scene, riser, seed, "oracle", speed, timeout, 0.005, False,
               diagnose=FallAttribution, goal_xy=goal)


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["scene"], row["riser"]), []).append(row)
    table = []
    for (scene, riser), runs in sorted(groups.items()):
        wins = sum(r["outcome"] == "success" for r in runs)
        labels = Counter(
            (r.get("diagnosis") or {}).get("fall_label", "?") for r in runs if r["outcome"] == "fell"
        )
        table.append({
            "scene": scene,
            "riser": riser,
            "success": wins,
            "runs": len(runs),
            "reliable": wins >= RELIABLE * len(runs),
            "mean_highest_m": round(sum(r["highest_both_feet_z"] for r in runs) / len(runs), 3),
            "fall_labels": dict(labels.most_common(3)),
        })
    return table


def highest_reliable(table: list[dict], scene: str) -> float:
    ok = [t["riser"] for t in table if t["scene"] == scene and t["reliable"]]
    return max(ok) if ok else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", default=["curb", "stairs_regular"])
    parser.add_argument("--risers", type=float, nargs="+", default=[0.04, 0.06, 0.08, 0.10])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--speed", type=float, default=0.15)
    parser.add_argument("--timeout", type=float, default=40.0)
    parser.add_argument("--workers", type=int, default=min(20, os.cpu_count() or 4))
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--compare", type=Path, default=None)
    parser.add_argument("--terrain-stack", choices=("heuristic", "classical"), default="classical",
                        help="Terrain-mode controller variant (loka.control.terrain.TERRAIN_STACK).")
    parser.add_argument("--goal", type=float, nargs=2, default=None,
                        help="Walk to this world xy instead of each flight's landing.")
    args = parser.parse_args()

    jobs = [(scene, riser, seed, args.speed, args.timeout, args.terrain_stack, args.goal)
            for scene in args.scenes for riser in args.risers for seed in args.seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_one, jobs))
    table = summarize(rows)
    before = {}
    if args.compare is not None and args.compare.is_file():
        for t in json.loads(args.compare.read_text())["table"]:
            before[(t["scene"], t["riser"])] = t
    for t in table:
        prev = before.get((t["scene"], t["riser"]))
        delta = "" if prev is None else (
            f"  (was {prev['success']}/{prev['runs']}, {prev['mean_highest_m']:.3f} m)"
        )
        print(f"{t['scene']:16s} riser={t['riser']:.2f}  {t['success']}/{t['runs']}  "
              f"mean_highest={t['mean_highest_m']:.3f} m  falls={t['fall_labels']}{delta}")
    for scene in args.scenes:
        print(f"[sweep] {scene}: highest reliable riser {highest_reliable(table, scene):.2f} m")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"sweep_{args.tag}.json"
    out.write_text(json.dumps({"table": table, "runs": rows}, indent=1))
    print(f"[sweep] wrote {out}")


if __name__ == "__main__":
    main()
