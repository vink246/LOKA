"""Offscreen MuJoCo video recording with a tracking camera."""

from __future__ import annotations

import mediapy as media
import mujoco
import numpy as np


DEFAULT_RECORD_PATH = "loka_recording.mp4"
DEFAULT_RECORD_CAMERA = "side_follow"
DEFAULT_RECORD_FPS = 30.0
DEFAULT_RECORD_WIDTH = 640
DEFAULT_RECORD_HEIGHT = 480


class VideoRecorder:
    """Capture sim frames from a named MuJoCo camera and stream them to a video file."""

    def __init__(
        self,
        model: mujoco.MjModel,
        path: str,
        camera: str = DEFAULT_RECORD_CAMERA,
        fps: float = DEFAULT_RECORD_FPS,
        width: int = DEFAULT_RECORD_WIDTH,
        height: int = DEFAULT_RECORD_HEIGHT,
    ):
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

    def maybe_capture(self, data: mujoco.MjData) -> None:
        """Append a frame when enough sim time has elapsed for the target fps."""
        if data.time + 1e-9 < self.next_frame_time:
            return

        self._renderer.update_scene(data, camera=self.camera)
        pixels = np.asarray(self._renderer.render())
        self._writer.add_image(pixels)
        self.frame_count += 1
        # Advance by whole intervals so we do not drift if a step is skipped.
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
