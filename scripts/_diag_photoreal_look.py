"""One-off: in-process ego dumps for photoreal room look checks."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco

from loka.agent.loka_config import loka_config
from loka.control.locomotion import LocomotionConfig
from loka.scene_props import apply_snapshot, build_plant_model, load_scene, load_scene_meta, load_scene_terrain
from loka.sim import Simulation
from loka.vision import EGO_CAMERA, EgoCameraSpec, encode_png
from loka.viz import hide_foot_contact_markers


def main() -> int:
    meta = load_scene_meta("navigation_photoreal")
    pool = loka_config().get("props") or {}
    cam = EgoCameraSpec(width=512, height=512, fovy_deg=90.0, pitch_down_deg=10.0)
    sim = Simulation(
        LocomotionConfig(),
        model_path=meta["model_path"],
        props=dict(pool.get("slots") or {}),
        static_props=tuple(pool.get("static") or ()),
        camera=cam.recipe(),
        terrain=load_scene_terrain("navigation_photoreal"),
        room_visual=meta["room_visual"],
        props_variant=meta["props_variant"],
        ceiling_z=meta.get("ceiling_z"),
    )
    for item in load_scene("navigation_photoreal"):
        print("spawn", sim.props.spawn(*item))

    model, _ = build_plant_model(sim.recipe)
    hide_foot_contact_markers(model)
    data = mujoco.MjData(model)
    apply_snapshot(model, data, sim.snapshot())
    renderer = mujoco.Renderer(model, height=512, width=512)
    out = Path("results/navila_route_check/_analysis/photoreal_smoke")
    out.mkdir(parents=True, exist_ok=True)

    def shoot(name: str, xyz, quat) -> None:
        q = data.qpos.copy()
        q[0:3] = xyz
        q[3:7] = quat
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=EGO_CAMERA)
        rgb = renderer.render()
        (out / name).write_bytes(encode_png(rgb))
        fp = rgb[380:500, 150:360]
        print(name, "mean", rgb.mean(axis=(0, 1)).round(1), "floor_std", fp.std(axis=(0, 1)).round(1))

    shoot("diag_start.png", (0, 0, 0.78), (1, 0, 0, 0))
    shoot("diag_near_shelf.png", (3.6, 0, 0.78), (1, 0, 0, 0))
    shoot("diag_mid.png", (1.5, 0.3, 0.78), (1, 0, 0, 0))
    # Default run_loka vision (70 fov / 30 pitch) for comparison with the bad local trial.
    cam30 = EgoCameraSpec(width=512, height=512, fovy_deg=70.0, pitch_down_deg=30.0)
    recipe30 = dict(sim.recipe)
    recipe30["camera"] = cam30.recipe()
    model30, _ = build_plant_model(recipe30)
    hide_foot_contact_markers(model30)
    data30 = mujoco.MjData(model30)
    apply_snapshot(model30, data30, sim.snapshot())
    r30 = mujoco.Renderer(model30, height=512, width=512)
    r30.update_scene(data30, camera=EGO_CAMERA)
    rgb30 = r30.render()
    (out / "diag_start_70_30.png").write_bytes(encode_png(rgb30))
    fp = rgb30[380:500, 150:360]
    print(
        "diag_start_70_30.png",
        "mean",
        rgb30.mean(axis=(0, 1)).round(1),
        "floor_std",
        fp.std(axis=(0, 1)).round(1),
    )
    r30.close()
    renderer.close()
    from loka.vision import floor_cell_size

    print("floor_cell", floor_cell_size(sim.model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
