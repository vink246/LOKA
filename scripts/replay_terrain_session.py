"""Replay a logged LOKA terrain session without the LLM and record it.

The terrain counterpart of ``scripts/replay_session.py``: it rebuilds the
run's plant from its logged argv, including ``--scene`` terrain and
``--terrain-stack``, and applies every orchestrator edit (task targets and
terrain mutations) at the sim time it was applied. A session that was not
recorded can then be rendered from the overview camera (``recording`` in
loka.yaml). Props, faults and belief searches are not replayed.

    python scripts/replay_terrain_session.py                 # latest session
    python scripts/replay_terrain_session.py --log logs/loka/session.jsonl --out recordings/curb.mp4
    python scripts/replay_terrain_session.py --check         # compare with logged poses, no video
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_LOG = ROOT / "logs" / "loka" / "session.jsonl"


def load_session(path: Path) -> dict:
    """argv, timed edits and logged pelvis poses from a session.jsonl."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    argv: list[str] = []
    edits: list[dict] = []
    poses: list[tuple[float, np.ndarray]] = []
    mutations_by_time: dict[float, list] = {}
    for row in rows:
        title = row.get("title", "") or ""
        body = row.get("body") or ""
        body = body if isinstance(body, str) else json.dumps(body)
        t = row.get("sim_time")
        if title.startswith("NOTE") and body.startswith("argv:"):
            argv = shlex.split(body.split("argv:", 1)[1].split("startup props:")[0])
        elif title.startswith("LLM_RESPONSE"):
            raw = body.split("--- RAW YAML ---")[-1]
            try:
                reply = yaml.safe_load(raw) or {}
            except yaml.YAMLError:
                reply = {}
            mutations_by_time[round(float(t), 6)] = list(reply.get("Model_Mutations") or [])
        elif title.startswith("APPLY"):
            applied = json.loads(body)
            edits.append({
                "t": float(t),
                "task_targets": dict(applied.get("task_targets") or {}),
                "mutations": mutations_by_time.get(round(float(t), 6), []) if applied.get("mutations") else [],
            })
        elif title.startswith("DISPATCH"):
            # The dispatch repeats earlier turns; the last snapshot is this one.
            found = re.findall(r"pelvis xyz \[m\][^:]*: ([+-][\d.]+) ([+-][\d.]+) ([+-][\d.]+)", body)
            if found:
                poses.append((float(t), np.array([float(v) for v in found[-1]])))
    return {"argv": argv, "edits": edits, "poses": poses}


def build_runtime(argv: list[str]):
    """The run's plant and runtime as run_loka builds them, with the LLM off."""
    from loka.agent.loka_config import loka_config
    from loka.agent.runtime import LokaConfig, LokaRuntime
    from loka.control.locomotion import LocomotionConfig
    from loka.control.stacks import apply_stack_arg
    from loka.run_loka import _scene_terrain, parse_args
    from loka.sim import Simulation
    from loka.vision import EgoCameraSpec

    args = parse_args(argv)
    config = LocomotionConfig.from_yaml(args.config) if args.config else LocomotionConfig()
    apply_stack_arg(config, args)
    if args.terrain_stack:
        from loka.control import terrain

        terrain.TERRAIN_STACK = args.terrain_stack
    settings = loka_config()
    vision_cfg = settings.get("vision") or {}
    prop_cfg = settings.get("props") or {}
    need_camera = args.vision or bool(vision_cfg.get("enabled", False)) or args.ego_view
    plant = {
        "props": None if args.no_props else dict(prop_cfg.get("slots") or {}),
        "static_props": tuple(prop_cfg.get("static") or ()),
        "camera": EgoCameraSpec.from_config(vision_cfg).recipe() if need_camera else None,
        "terrain": _scene_terrain(args),
    }
    sim = Simulation(config, model_path=args.model, **plant)
    runtime = LokaRuntime(
        sim,
        LokaConfig(enable_llm=False, terrain_belief=args.terrain_belief,
                   enable_session_log=False, print_compressor=False),
    )
    return sim, runtime, settings


def apply_edit(runtime, edit: dict) -> None:
    from loka.agent.terrain_state import apply_terrain_mutation, sync_controller_terrain

    for mut in edit["mutations"]:
        if str(mut.get("object_type", "")) in ("volume", "stairs"):
            line = apply_terrain_mutation(runtime.loka_state, mut, sim_time=edit["t"])
            print(f"  t={edit['t']:6.2f} terrain: {line}")
    sync_controller_terrain(runtime.controller, runtime.loka_state)
    if edit["task_targets"]:
        runtime.controller.set_task_targets(edit["task_targets"])
        print(f"  t={edit['t']:6.2f} targets: {edit['task_targets']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--duration", type=float, default=None,
                        help="Sim seconds to replay; default 3 s past the last edit or logged pose.")
    parser.add_argument("--check", action="store_true", help="Replay and compare poses, no video.")
    parser.add_argument("--edit-delay", type=float, default=0.0,
                        help="Apply each edit this many sim seconds after its logged time.")
    args = parser.parse_args()

    session = load_session(args.log)
    argv = session["argv"]
    print(f"[replay] {args.log}: argv {' '.join(argv)}; {len(session['edits'])} edits")
    sim, runtime, settings = build_runtime(argv)
    last = max([e["t"] for e in session["edits"]] + [t for t, _ in session["poses"]] + [0.0])
    duration = args.duration or last + 3.0

    recorder = None
    if not args.check:
        from loka.recording import OverviewCamera, SessionRecorder

        record_cfg = settings.get("recording") or {}
        out = args.out or (ROOT / record_cfg.get("directory", "recordings")
                           / f"replay_{datetime.now():%Y%m%d_%H%M%S}.mp4")
        recorder = SessionRecorder(sim, out, OverviewCamera.from_config(record_cfg))
        recorder.maybe_capture()

    edits = sorted(session["edits"], key=lambda e: e["t"])
    poses = list(session["poses"])
    errors = []
    k = 0
    while sim.data.time < duration and not sim.fell:
        # The runtime applies an edit at the start of the step whose time it logs.
        while k < len(edits) and sim.data.time + 1e-9 >= edits[k]["t"] + args.edit_delay:
            apply_edit(runtime, edits[k])
            k += 1
        while poses and sim.data.time + 1e-9 >= poses[0][0]:
            t, logged = poses.pop(0)
            now = np.asarray(sim.data.qpos[:3], dtype=float)
            errors.append(float(np.linalg.norm(now - logged)))
            print(f"  t={t:6.2f} pelvis logged {np.round(logged, 3)} replay {np.round(now, 3)} "
                  f"(error {errors[-1] * 1000:.1f} mm)")
        runtime.step()
        if recorder is not None:
            recorder.maybe_capture()
    print(f"[replay] ended at t={sim.data.time:.2f}s fell={sim.fell} "
          f"final pelvis {np.round(sim.data.qpos[:3], 3)}; "
          f"max pose error {max(errors) * 1000 if errors else 0:.1f} mm")
    if recorder is not None:
        recorder.close()


if __name__ == "__main__":
    main()
