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
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loka.agent.apply import apply_stand_scratchpad  # noqa: E402
from loka.agent.loka_config import loka_config  # noqa: E402
from loka.agent.runtime import LokaConfig, LokaRuntime  # noqa: E402
from loka.control.locomotion import LocomotionConfig  # noqa: E402
from loka.recording import OverviewCamera, SessionRecorder  # noqa: E402
from loka.scene_props import load_scene, parse_props_spec  # noqa: E402
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
    if setup and args.scene is None and args.props is None:
        args.props = setup["props"] or None
        argv = setup["argv"].split()
        args.vision = args.vision or "--vision" in argv or "--ego-view" in argv
        print(f"[replay] setup from the log: argv '{setup['argv']}', props '{setup['props']}'")
    until = args.until if args.until is not None else applies[-1][0] + 6.0

    settings = loka_config()
    prop_cfg = settings.get("props") or {}
    camera = EgoCameraSpec.from_config(settings.get("vision") or {}).recipe() if args.vision else None
    sim = Simulation(
        LocomotionConfig(),
        props=dict(prop_cfg.get("slots") or {}),
        static_props=tuple(prop_cfg.get("static") or ()),
        camera=camera,
    )
    runtime = LokaRuntime(sim, LokaConfig(enable_llm=False, enable_session_log=False, print_compressor=False))
    runtime.effort_observer.printout = False
    spawns = (load_scene(args.scene) if args.scene else []) + (parse_props_spec(args.props) if args.props else [])
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
