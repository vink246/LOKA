"""Sim-state video recording that coexists with the interactive MuJoCo viewer.

Live ``mujoco.Renderer`` shares the process with GLFW and fails under WSLg
(``GLX`` / ``EGL_BAD_ACCESS``). Instead we sample ``qpos`` at the target fps
while the viewer runs, then render the clip in a fresh spawned process that
owns EGL exclusively.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np


DEFAULT_RECORDINGS_DIR = Path("recordings")
DEFAULT_RECORD_PATH = "loka_recording.mp4"
DEFAULT_RECORD_CAMERA = "side_follow"
DEFAULT_RECORD_FPS = 30.0
#: 4:3, matching the prior 640x480 framing; fits the scene's 1920x1080 offscreen buffer.
DEFAULT_RECORD_WIDTH = 1440
DEFAULT_RECORD_HEIGHT = 1080


def timestamped_recording_path(
    directory: Path | str = DEFAULT_RECORDINGS_DIR,
) -> Path:
    """Return ``directory/loka_YYYYMMDD_HHMMSS.mp4``, creating ``directory`` if needed."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return directory / f"loka_{stamp}.mp4"


def _render_clip_worker(job: dict) -> None:
    """Child-process entry: load model, render buffered qpos, write mp4."""
    os.environ["MUJOCO_GL"] = "egl"
    # Import only after MUJOCO_GL is set so gl_context binds to EGL.
    import mediapy as media
    import mujoco

    model = mujoco.MjModel.from_xml_path(job["model_path"])
    data = mujoco.MjData(model)
    qpos_frames = np.load(job["frames_path"])["qpos"]
    camera = job["camera"]
    fps = float(job["fps"])
    width = int(job["width"])
    height = int(job["height"])
    out_path = job["out_path"]

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
        with media.VideoWriter(out_path, shape=(height, width), fps=fps) as writer:
            for qpos in qpos_frames:
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
    camera: str = DEFAULT_RECORD_CAMERA,
    fps: float = DEFAULT_RECORD_FPS,
    width: int = DEFAULT_RECORD_WIDTH,
    height: int = DEFAULT_RECORD_HEIGHT,
) -> None:
    """Render ``qpos_frames`` to ``out_path`` in a spawned EGL process."""
    if qpos_frames.size == 0:
        raise ValueError("No frames to render")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        frames_path = tmp.name
    np.savez_compressed(frames_path, qpos=np.asarray(qpos_frames, dtype=np.float64))

    job = {
        "model_path": str(model_path),
        "frames_path": frames_path,
        "out_path": str(out_path),
        "camera": camera,
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
    }
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_render_clip_worker, args=(job,))
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(
            f"Recording render subprocess failed (exit code {proc.exitcode})"
        )


class VideoRecorder:
    """Sample ``qpos`` at a fixed rate; encode the mp4 when :meth:`close` runs."""

    def __init__(
        self,
        model,  # mujoco.MjModel — typed loosely to avoid import-time GL binding
        path: str | Path,
        camera: str = DEFAULT_RECORD_CAMERA,
        fps: float = DEFAULT_RECORD_FPS,
        width: int = DEFAULT_RECORD_WIDTH,
        height: int = DEFAULT_RECORD_HEIGHT,
        start_time: float = 0.0,
        model_path: str | Path | None = None,
    ):
        if fps <= 0:
            raise ValueError(f"Recording fps must be positive, got {fps}")
        if model_path is None:
            raise ValueError("model_path is required to render after the viewer exits")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.path = str(path)
        self.model_path = str(model_path)
        self.camera = camera
        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)
        self.frame_interval = 1.0 / self.fps
        self.next_frame_time = float(start_time)
        self.frame_count = 0
        self._qpos_frames: list[np.ndarray] = []
        self._nq = int(model.nq)
        self._closed = False

    def maybe_capture(self, data) -> None:
        """Append a qpos sample when enough sim time has elapsed for the target fps."""
        if self._closed or data.time + 1e-9 < self.next_frame_time:
            return

        self._qpos_frames.append(np.array(data.qpos, dtype=np.float64, copy=True))
        self.frame_count += 1
        while self.next_frame_time <= data.time + 1e-9:
            self.next_frame_time += self.frame_interval

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._qpos_frames:
            print(f"\n[LOKA] Recording abandoned (0 frames) — nothing written to {self.path}")
            return

        print(
            f"\n[LOKA] Encoding {self.frame_count} frames @ {self.fps:.1f} fps "
            f"-> {self.path} ..."
        )
        render_qpos_clip(
            model_path=self.model_path,
            qpos_frames=np.stack(self._qpos_frames, axis=0),
            out_path=self.path,
            camera=self.camera,
            fps=self.fps,
            width=self.width,
            height=self.height,
        )
        self._qpos_frames.clear()
        print(
            f"[LOKA] Saved recording ({self.frame_count} frames @ {self.fps:.1f} fps) "
            f"to {self.path}"
        )


class RecordingToggle:
    """Keyboard toggle: first press starts a timestamped clip, second press saves it.

    ``key_callback`` only flips a flag. Capture / encode run on the sim thread via
    :meth:`maybe_capture`.
    """

    def __init__(
        self,
        model,
        model_path: str | Path,
        camera: str = DEFAULT_RECORD_CAMERA,
        fps: float = DEFAULT_RECORD_FPS,
        width: int = DEFAULT_RECORD_WIDTH,
        height: int = DEFAULT_RECORD_HEIGHT,
        directory: Path | str = DEFAULT_RECORDINGS_DIR,
    ):
        self.model = model
        self.model_path = str(model_path)
        self.camera = camera
        self.fps = fps
        self.width = width
        self.height = height
        self.directory = Path(directory)
        self.recorder: VideoRecorder | None = None
        self._pending_toggle = False

    @property
    def active(self) -> bool:
        return self.recorder is not None

    def request_toggle(self) -> None:
        """Signal a start/stop from the viewer thread (no OpenGL here)."""
        self._pending_toggle = True

    def toggle(self, sim_time: float = 0.0) -> None:
        if self.recorder is not None:
            self.recorder.close()
            self.recorder = None
            print("\n[KEYBOARD] Stopped recording.")
            return

        path = timestamped_recording_path(self.directory)
        print(f"\nstart recording -> {path}")
        self.recorder = VideoRecorder(
            self.model,
            path=path,
            camera=self.camera,
            fps=self.fps,
            width=self.width,
            height=self.height,
            start_time=sim_time,
            model_path=self.model_path,
        )

    def maybe_capture(self, data) -> None:
        if self._pending_toggle:
            self._pending_toggle = False
            self.toggle(data.time)
        if self.recorder is not None:
            self.recorder.maybe_capture(data)

    def close(self) -> None:
        self._pending_toggle = False
        if self.recorder is not None:
            self.recorder.close()
            self.recorder = None

    def key_callback(self, keycode: int, sim_time: float = 0.0) -> None:
        """MuJoCo viewer key hook. ``sim_time`` is ignored; capture uses live time."""
        del sim_time
        try:
            if chr(keycode).lower() != "r":
                return
        except ValueError:
            return
        self.request_toggle()
