"""Offscreen MuJoCo video recording with a tracking camera."""

from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
from pathlib import Path

import numpy as np


DEFAULT_RECORD_PATH = "loka_recording.mp4"
DEFAULT_RECORD_CAMERA = "side_follow"
DEFAULT_RECORD_FPS = 30.0
DEFAULT_RECORD_WIDTH = 640
DEFAULT_RECORD_HEIGHT = 480
SUITE_RECORD_WIDTH = 1920
SUITE_RECORD_HEIGHT = 1080


class VideoRecorder:
    """Capture sim frames from a named MuJoCo camera and stream them to a video file."""

    def __init__(
        self,
        model,
        path: str,
        camera: str = DEFAULT_RECORD_CAMERA,
        fps: float = DEFAULT_RECORD_FPS,
        width: int = DEFAULT_RECORD_WIDTH,
        height: int = DEFAULT_RECORD_HEIGHT,
    ):
        import mediapy as media
        import mujoco

        if fps <= 0:
            raise ValueError(f"Recording fps must be positive, got {fps}")

        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if cam_id < 0:
            available = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                for i in range(model.ncam)
            ]
            raise ValueError(
                f"Unknown camera {camera!r}. Available: {', '.join(available) or 'none'}"
            )

        self.path = path
        self.camera = camera
        self.fps = float(fps)
        self.frame_interval = 1.0 / self.fps
        self.next_frame_time = 0.0
        self.frame_count = 0
        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._writer = media.VideoWriter(
            path,
            shape=(height, width),
            fps=self.fps,
        )
        self._writer.__enter__()

    def maybe_capture(self, data) -> None:
        """Append a frame when enough sim time has elapsed for the target fps."""
        if data.time + 1e-9 < self.next_frame_time:
            return

        self._renderer.update_scene(data, camera=self.camera)
        pixels = np.asarray(self._renderer.render())
        self._writer.add_image(pixels)
        self.frame_count += 1
        while self.next_frame_time <= data.time + 1e-9:
            self.next_frame_time += self.frame_interval

    def close(self) -> None:
        if self._writer is not None:
            self._writer.__exit__(None, None, None)
            self._writer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        print(
            f"\n[LOKA] Saved recording ({self.frame_count} frames @ {self.fps:.1f} fps) "
            f"to {self.path}"
        )


class QposClipSampler:
    """Sample qpos at a fixed rate for post-hoc EGL rendering."""

    def __init__(self, fps: float, start_time: float = 0.0):
        if fps <= 0:
            raise ValueError(f"Recording fps must be positive, got {fps}")
        self.fps = float(fps)
        self.frame_interval = 1.0 / self.fps
        self.next_frame_time = float(start_time)
        self.times: list[float] = []
        self.qpos_frames: list[np.ndarray] = []

    @property
    def frame_count(self) -> int:
        return len(self.qpos_frames)

    def maybe_capture(self, data) -> None:
        if data.time + 1e-9 < self.next_frame_time:
            return
        self.times.append(float(data.time))
        self.qpos_frames.append(np.array(data.qpos, dtype=np.float64, copy=True))
        while self.next_frame_time <= data.time + 1e-9:
            self.next_frame_time += self.frame_interval

    def stacked(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.qpos_frames:
            return np.zeros((0,), dtype=np.float64), np.zeros((0, 0), dtype=np.float64)
        return (
            np.asarray(self.times, dtype=np.float64),
            np.stack(self.qpos_frames, axis=0),
        )


def _apply_obstacle_overlay(mujoco, model, visible: bool, pos, size) -> None:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle")
    if geom_id < 0:
        return
    if visible:
        model.geom_pos[geom_id] = np.asarray(pos, dtype=float)
        model.geom_size[geom_id] = np.asarray(size, dtype=float)
        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = 1
    else:
        model.geom_pos[geom_id] = np.array([0.0, 0.0, -5.0])
        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = 0


def _render_clip_worker(job: dict) -> None:
    """Child-process entry: load model, render buffered qpos, write mp4."""
    os.environ["MUJOCO_GL"] = "egl"
    import mediapy as media_worker
    import mujoco

    model = mujoco.MjModel.from_xml_path(job["model_path"])
    data = mujoco.MjData(model)
    payload = np.load(job["frames_path"])
    qpos_frames = payload["qpos"]
    times = payload["times"] if "times" in payload.files else None
    camera = job["camera"]
    fps = float(job["fps"])
    width = int(job["width"])
    height = int(job["height"])
    out_path = job["out_path"]
    obstacle = job.get("obstacle")

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cam_id < 0:
        available = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(model.ncam)
        ]
        raise ValueError(
            f"Unknown camera {camera!r}. Available: {', '.join(available) or 'none'}"
        )

    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        with media_worker.VideoWriter(out_path, shape=(height, width), fps=fps) as writer:
            for i, qpos in enumerate(qpos_frames):
                if obstacle is not None:
                    visible_from = float(obstacle.get("visible_from") or 0.0)
                    t = float(times[i]) if times is not None else visible_from
                    visible = t + 1e-9 >= visible_from
                    _apply_obstacle_overlay(
                        mujoco, model, visible, obstacle["pos"], obstacle["size"]
                    )
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                writer.add_image(np.asarray(renderer.render()))
    finally:
        renderer.close()
        try:
            os.remove(job["frames_path"])
        except OSError:
            pass


def render_qpos_clip(
    *,
    model_path: str | Path,
    qpos_frames: np.ndarray,
    out_path: str | Path,
    times: np.ndarray | None = None,
    obstacle: dict | None = None,
    camera: str = DEFAULT_RECORD_CAMERA,
    fps: float = DEFAULT_RECORD_FPS,
    width: int = SUITE_RECORD_WIDTH,
    height: int = SUITE_RECORD_HEIGHT,
) -> None:
    """Render ``qpos_frames`` to ``out_path`` in a spawned EGL process."""
    if qpos_frames.size == 0:
        raise ValueError("No frames to render")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        frames_path = tmp.name
    save_kwargs = {"qpos": np.asarray(qpos_frames, dtype=np.float64)}
    if times is not None:
        save_kwargs["times"] = np.asarray(times, dtype=np.float64)
    np.savez_compressed(frames_path, **save_kwargs)

    job = {
        "model_path": str(model_path),
        "frames_path": frames_path,
        "out_path": str(out_path),
        "camera": camera,
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "obstacle": obstacle,
    }
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_render_clip_worker, args=(job,))
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(
            f"Recording render subprocess failed (exit code {proc.exitcode})"
        )
