"""Offscreen isometric clips. Perturbations are drawn; controller overlays are not.

The clip is rendered after the episode from ``render.npz``, in a spawned
process. The GL backend is the walker suite's: OSMesa on a headless CPU node,
EGL when a display is present. MuJoCo locks that choice at import, so it is
set in this process before the child starts.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import tempfile
from pathlib import Path

import numpy as np

from loka.recording import prepare_offscreen_env, start_offscreen

ICE_RGBA = np.array([0.15, 0.75, 1.0, 1.0], dtype=np.float32)
WEAK_RGBA = np.array([1.0, 0.45, 0.05, 1.0], dtype=np.float32)
PUSH_RGBA = (0.10, 0.85, 1.00, 0.95)
MASS_RGBA = (1.00, 0.45, 0.05, 0.90)
# Smaller than the live viewer's 1 N cutoff so a modest shove is still visible.
MIN_DRAW_FORCE_N = 0.05


def render_episode_dir(directory, *, recording, model_path: str) -> Path:
    """Encode ``directory/episode.mp4`` from the stored ``render.npz``."""
    directory = Path(directory)
    blob_path = directory / "render.npz"
    if not blob_path.is_file():
        raise FileNotFoundError(f"No render.npz in {directory}")
    with np.load(blob_path) as blob:
        times = np.asarray(blob["times"])
        qpos = np.asarray(blob["qpos"])
        overlays = json.loads(str(blob["overlays_json"].item()))
    if times.size == 0 or qpos.size == 0:
        raise ValueError(f"No frames in {blob_path}")
    out_path = directory / "episode.mp4"
    _spawn_render(
        model_path=model_path,
        times=times,
        qpos=qpos,
        overlays=overlays,
        out_path=out_path,
        recording=recording,
    )
    return out_path


def _spawn_render(*, model_path, times, qpos, overlays, out_path, recording) -> None:
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        frames_path = tmp.name
    np.savez_compressed(
        frames_path,
        times=np.asarray(times),
        qpos=np.asarray(qpos),
        overlays_json=np.array(json.dumps(overlays)),
    )
    job = {
        "model_path": str(model_path),
        "frames_path": frames_path,
        "out_path": str(out_path),
        "fps": float(recording.fps),
        "width": int(recording.width),
        "height": int(recording.height),
        "azimuth": float(recording.azimuth),
        "elevation": float(recording.elevation),
        "distance": float(recording.distance),
        "min_push_visible_s": float(recording.min_push_visible_s),
    }
    # Spawn re-imports the main module, and that import pulls MuJoCo in,
    # before ``_render_worker`` runs. The GL backend is chosen on first
    # import, so the child has to inherit the variable. Setting it inside
    # the worker is too late: on a headless node the default is GLFW, and
    # GLFW exits because DISPLAY is unset.
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_render_worker, args=(job,))
    backend = start_offscreen(proc)
    proc.join()
    try:
        os.remove(frames_path)
    except OSError:
        pass
    if proc.exitcode != 0:
        raise RuntimeError(f"render subprocess failed (exit {proc.exitcode}, gl={backend})")


def _render_worker(job: dict) -> None:
    prepare_offscreen_env()
    import mediapy as media
    import mujoco

    from loka.viz import (
        FORCE_ARROW_SCALE,
        G,
        MAX_ARROW_LEN,
        WEIGHT_ARROW_SCALE,
        add_arrow,
        add_sphere,
        hide_foot_contact_markers,
    )

    model = mujoco.MjModel.from_xml_path(job["model_path"])
    hide_foot_contact_markers(model)
    data = mujoco.MjData(model)
    with np.load(job["frames_path"]) as blob:
        times = np.asarray(blob["times"])
        qpos_frames = np.asarray(blob["qpos"])
        overlays = json.loads(str(blob["overlays_json"].item()))
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    floor_rgba = None if floor < 0 else model.geom_rgba[floor].copy()
    actuator_rgba = _actuator_rgba(model, mujoco)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = int(pelvis)
    cam.azimuth = float(job["azimuth"])
    cam.elevation = float(job["elevation"])
    cam.distance = float(job["distance"])

    width = int(job["width"])
    height = int(job["height"])
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        with media.VideoWriter(
            job["out_path"], shape=(height, width), fps=float(job["fps"])
        ) as writer:
            for time_s, qpos in zip(times, qpos_frames):
                _restore_rgba(model, floor, floor_rgba, actuator_rgba)
                _apply_persistent_overlays(model, mujoco, overlays, float(time_s), floor)
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)
                cam.lookat[:] = data.xpos[pelvis]
                renderer.update_scene(data, camera=cam)
                _draw_transient(
                    renderer.scene,
                    model,
                    data,
                    mujoco,
                    overlays,
                    float(time_s),
                    float(job["min_push_visible_s"]),
                    add_arrow,
                    add_sphere,
                    G,
                    FORCE_ARROW_SCALE,
                    MAX_ARROW_LEN,
                    WEIGHT_ARROW_SCALE,
                )
                writer.add_image(np.asarray(renderer.render()))
    finally:
        renderer.close()
        try:
            os.remove(job["frames_path"])
        except OSError:
            pass


def _actuator_rgba(model, mujoco) -> dict[int, np.ndarray]:
    saved = {}
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            continue
        body_id = int(model.jnt_bodyid[joint_id])
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) == body_id:
                saved[geom_id] = model.geom_rgba[geom_id].copy()
    return saved


def _restore_rgba(model, floor: int, floor_rgba, actuator_rgba: dict) -> None:
    if floor >= 0 and floor_rgba is not None:
        model.geom_rgba[floor] = floor_rgba
    for geom_id, rgba in actuator_rgba.items():
        model.geom_rgba[geom_id] = rgba


def _apply_persistent_overlays(model, mujoco, overlays, time_s: float, floor: int) -> None:
    for overlay in overlays:
        if float(overlay.get("t", 0.0)) > time_s + 1e-9:
            continue
        kind = overlay.get("kind")
        if kind == "friction" and floor >= 0 and float(overlay.get("mu", 1.0)) < 0.6:
            model.geom_rgba[floor] = ICE_RGBA
        elif kind == "actuator_gear":
            _tint_actuator(model, mujoco, str(overlay.get("actuator", "")))


def _tint_actuator(model, mujoco, name: str) -> None:
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if actuator_id < 0:
        needle = name.lower()
        for index in range(model.nu):
            candidate = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) or ""
            if needle and needle in candidate.lower():
                actuator_id = index
                break
    if actuator_id < 0:
        return
    joint_id = int(model.actuator_trnid[actuator_id, 0])
    if joint_id < 0:
        return
    body_id = int(model.jnt_bodyid[joint_id])
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) == body_id:
            model.geom_rgba[geom_id] = WEAK_RGBA


def _draw_transient(
    scene, model, data, mujoco, overlays, time_s, min_push_s,
    add_arrow, add_sphere, gravity, force_scale, max_len, weight_scale,
) -> None:
    for overlay in overlays:
        start = float(overlay.get("t", 0.0))
        if start > time_s + 1e-9:
            continue
        kind = overlay.get("kind")
        if kind == "mass":
            _draw_mass(
                scene, model, data, mujoco, overlay,
                add_arrow, add_sphere, gravity, max_len, weight_scale,
            )
        elif kind == "push":
            duration = float(overlay.get("duration_s", 0.05))
            visible = max(duration, float(min_push_s))
            if time_s <= start + visible + 1e-9:
                _draw_push(
                    scene, model, data, mujoco, overlay,
                    add_arrow, force_scale, max_len,
                )


def _draw_mass(scene, model, data, mujoco, overlay, add_arrow, add_sphere, gravity, max_len, weight_scale) -> None:
    body = str(overlay.get("body", ""))
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    delta = float(overlay.get("delta_kg", 0.0))
    if body_id < 0 or delta <= 0.05:
        return
    center = data.xpos[body_id].copy()
    center[2] += 0.08
    add_sphere(scene, center, 0.045 + 0.008 * min(delta, 15.0), MASS_RGBA)
    weight = np.array([0.0, 0.0, -delta * gravity * weight_scale])
    norm = float(np.linalg.norm(weight))
    if norm > max_len:
        weight *= max_len / norm
    add_arrow(scene, center, weight, (1.0, 0.25, 0.05, 0.95), width=0.028)


def _draw_push(scene, model, data, mujoco, overlay, add_arrow, force_scale, max_len) -> None:
    force = np.asarray(overlay.get("force", [0.0, 0.0, 0.0]), dtype=float)
    if float(np.linalg.norm(force)) < MIN_DRAW_FORCE_N:
        return
    body_name = str(overlay.get("body", "pelvis"))
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return
    origin = data.xpos[body_id].copy()
    origin[2] += 0.05
    vector = force * force_scale
    norm = float(np.linalg.norm(vector))
    if norm > max_len:
        vector *= max_len / norm
    add_arrow(scene, origin, vector, PUSH_RGBA, width=0.025)
