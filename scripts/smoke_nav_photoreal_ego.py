#!/usr/bin/env python3
"""Dump one ego frame from navigation_photoreal (compile + render smoke).

    MUJOCO_GL=egl python scripts/smoke_nav_photoreal_ego.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

from loka.agent.context import format_compact_vision_block
from loka.agent.loka_config import loka_config
from loka.control.locomotion import LocomotionConfig
from loka.scene_props import load_scene, load_scene_meta, load_scene_terrain
from loka.sim import Simulation
from loka.vision import EgoCameraSpec, EgoRenderer, encode_png, floor_cell_size, nominal_camera_pose


def main() -> int:
    meta = load_scene_meta("navigation_photoreal")
    pool = loka_config().get("props") or {}
    cam = EgoCameraSpec(width=512, height=512, fovy_deg=90.0, pitch_down_deg=10.0)
    sim = Simulation(
        LocomotionConfig(),
        props=dict(pool.get("slots") or {}),
        static_props=tuple(pool.get("static") or ()),
        camera=cam.recipe(),
        terrain=load_scene_terrain("navigation_photoreal"),
        room_visual=meta["room_visual"],
        props_variant=meta["props_variant"],
        ceiling_z=meta.get("ceiling_z"),
    )
    for item in load_scene("navigation_photoreal"):
        print(sim.props.spawn(*item))
    cell = floor_cell_size(sim.model)
    print("floor_cell", cell)
    print(format_compact_vision_block(cam, nominal_camera_pose(sim.model), cell))
    print("--- without floor metric ---")
    print(format_compact_vision_block(cam, nominal_camera_pose(sim.model)))
    renderer = EgoRenderer(sim.recipe)
    try:
        rgb = renderer.render_rgb(sim.snapshot())
    finally:
        renderer.close()
    out = Path("results/navila_route_check/_analysis/photoreal_smoke")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "ego_t0.png"
    path.write_bytes(encode_png(rgb))
    print(f"wrote {path} shape={rgb.shape} mean_rgb={rgb.mean(axis=(0, 1))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
