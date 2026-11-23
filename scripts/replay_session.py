#!/usr/bin/env python3
"""Re-run a logged run_loka session without the LLM and record it.

    python scripts/replay_session.py                     # newest log, setup read from it
    python scripts/replay_session.py --scene obstacle_course --vision
    python scripts/replay_session.py --log logs/loka/session.jsonl --props "box:1,0" --out replay.mp4

The plant is deterministic and the LLM only acts through the scratchpads it
returns, so applying each logged scratchpad at the sim time it was applied
originally reproduces the run, with no API calls. Logs written since the
argv note was added carry the command line and startup props, which are used
automatically; for older logs pass the same --scene / --props as the original
run (and --vision if it had the head camera, so the model matches).
Each replayed pelvis position is checked against the dispatch snapshots in
the log; a drift means the setup differs from the original.

Not replayed: faults, belief searches, runtime auto-stands, and stdin prop
commands typed during the run. The script stops with an error if the log
shows any of the first three.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loka.agent.apply import apply_stand_scratchpad  # noqa: E402
from loka.agent.loka_config import loka_config  # noqa: E402
from loka.agent.runtime import LokaConfig, LokaRuntime  # noqa: E402
from loka.control.locomotion import LocomotionConfig  # noqa: E402
from loka.recording import OverviewCamera, SessionRecorder  # noqa: E402
from loka.scene_props import (  # noqa: E402
    load_scene,
    load_scene_meta,
    load_scene_terrain,
    parse_props_spec,
)
from loka.sim import Simulation  # noqa: E402
from loka.vision import EgoCameraSpec  # noqa: E402

_PELVIS = re.compile(r"pelvis xyz \[m\][^:]*:\s*([-+\d.]+)\s+([-+\d.]+)\s+([-+\d.]+)")
_UNSUPPORTED = ("FAULT", "SEARCH", "effort bias while walking", "FELL")


def _read_log(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    applies, checks = [], []
    setup = {}
    for row in rows:
        kind = row.get("kind")
        t = float(row.get("sim_time") or 0.0)
        body = row.get("body", "")
        if kind == "event" and row.get("title") == "NOTE" and body.startswith("argv:"):
            argv_line, _, props_line = body.partition("\nstartup props: ")
            setup["argv"] = argv_line[len("argv:"):].strip()
            setup["props"] = "" if props_line.strip() == "(none)" else props_line.strip()
            continue
        if kind == "llm_response" and isinstance(row.get("scratchpad"), dict):
            applies.append((t, row["scratchpad"]))
        elif kind == "event" and row.get("title") == "DISPATCH":
            # The body repeats earlier turns; the newest user message is last.
            matches = _PELVIS.findall(row.get("body", ""))
            if matches:
                checks.append((t, np.array([float(v) for v in matches[-1]])))
        elif kind == "event":
            text = f"{row.get('title', '')} {row.get('body', '')}"
            if any(word in text for word in _UNSUPPORTED):
                raise SystemExit(f"log has an event this replay cannot reproduce at t={t:.2f}s: {text[:120]}")
    return applies, checks, setup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", default="logs/loka/session.jsonl")
    parser.add_argument("--scene", default=None, help="scene the original run used (--scene)")
    parser.add_argument("--props", default=None, help="props the original run used (--props)")
    parser.add_argument("--vision", action="store_true", help="original run had --vision / --ego-view")
    parser.add_argument("--until", type=float, default=None, help="sim time to stop [s]; default last apply + 6 s")
    parser.add_argument("--out", default=None, help="mp4 path; default recordings/replay_<stamp>.mp4")
    parser.add_argument("--check-only", action="store_true", help="compare against the log, no video")
    args = parser.parse_args()

    applies, checks, setup = _read_log(Path(args.log))
    if not applies:
        print(f"no applied scratchpads in {args.log}")
        return 1
    if setup:
        argv = shlex.split(setup.get("argv") or "")
        if args.scene is None and "--scene" in argv:
            args.scene = argv[argv.index("--scene") + 1]
        if args.props is None and setup.get("props"):
            # Scene already places these; only use the logged prop string when
            # there was no --scene (ad-hoc --props runs).
            if args.scene is None:
                args.props = setup["props"]
        args.vision = args.vision or "--vision" in argv or "--ego-view" in argv
        print(f"[replay] setup from the log: argv '{setup['argv']}', props '{setup['props']}'")
    until = args.until if args.until is not None else applies[-1][0] + 6.0

    settings = loka_config()
    vision_cfg = dict(settings.get("vision") or {})
    prop_cfg = settings.get("props") or {}
    scene_meta = load_scene_meta(args.scene) if args.scene else {}
    if isinstance(scene_meta.get("camera"), dict) and scene_meta["camera"]:
        vision_cfg.update(scene_meta["camera"])
    camera = EgoCameraSpec.from_config(vision_cfg).recipe() if args.vision else None
    plant = {
        "props": dict(prop_cfg.get("slots") or {}),
        "static_props": tuple(prop_cfg.get("static") or ()),
        "camera": camera,
        "terrain": load_scene_terrain(args.scene) if args.scene else None,
    }
    for key in ("room_visual", "props_variant", "ceiling_z", "model_path"):
        if scene_meta.get(key) not in (None, ""):
            plant[key] = scene_meta[key]
    model_path = plant.pop("model_path", None)
    sim = Simulation(LocomotionConfig(), model_path=model_path, **plant)
    runtime = LokaRuntime(sim, LokaConfig(enable_llm=False, enable_session_log=False, print_compressor=False))
    runtime.effort_observer.printout = False
    # Prefer the scene file's props; logged startup props duplicate them when
    # the original run used --scene.
    spawns = list(load_scene(args.scene) if args.scene else [])
    if args.props:
        spawns.extend(parse_props_spec(args.props))
    for kind, x, y, yaw in spawns:
        print(sim.props.spawn(kind, x, y, yaw))

    pending = list(applies)

    def drain_logged() -> None:
        while pending and sim.data.time >= pending[0][0] - 1e-9:
            t, scratchpad = pending.pop(0)
            runtime._sync_walk_block()
            apply_stand_scratchpad(
                runtime.controller,
                json.loads(json.dumps(scratchpad)),
                runtime.loka_state,
                plant_model=sim.model,
                sim_time=float(sim.data.time),
            )
            print(f"[replay] applied the t={t:.3f}s scratchpad at t={sim.data.time:.3f}s")

    runtime._drain_llm = drain_logged

    record_cfg = settings.get("recording") or {}
    out = Path(args.out) if args.out else (
        Path(record_cfg.get("directory", "recordings")) / f"replay_{datetime.now():%Y%m%d_%H%M%S}.mp4"
    )
    recorder = SessionRecorder(sim, out, OverviewCamera.from_config(record_cfg))
    checks = list(checks)
    worst = 0.0
    try:
        if args.check_only:
            recorder._closed = True
        recorder.maybe_capture()
        while sim.data.time < until and not sim.fell:
            while checks and sim.data.time >= checks[0][0] - 1e-9:
                t, logged = checks.pop(0)
                here = sim.data.qpos[:3]
                err = float(np.linalg.norm(here[:2] - logged[:2]))
                worst = max(worst, err)
                print(
                    f"[check] t={t:6.2f}s logged ({logged[0]:+.3f}, {logged[1]:+.3f}) "
                    f"replay ({here[0]:+.3f}, {here[1]:+.3f})  err {err * 1000:.1f} mm"
                )
            runtime.step()
            recorder.maybe_capture()
    finally:
        runtime.close()
        print(f"[replay] stopped at t={sim.data.time:.2f}s fell={sim.fell}; worst xy error {worst * 1000:.1f} mm")
        recorder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
