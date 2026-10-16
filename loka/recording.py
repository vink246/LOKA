"""Sim-state video recording that coexists with the interactive MuJoCo viewer.

Live ``mujoco.Renderer`` shares the process with GLFW and fails under WSLg
(``GLX`` / ``EGL_BAD_ACCESS``). Instead we sample ``qpos`` at the target fps
while the viewer runs, then render the clip in a fresh spawned process. A
headless node uses OSMesa; a machine with a display uses EGL.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import pickle
import tempfile
from dataclasses import dataclass
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


_OFFSCREEN_KEYS = ("MUJOCO_GL", "PYOPENGL_PLATFORM", "LD_LIBRARY_PATH", "MPLCONFIGDIR")


def prepare_offscreen_env() -> str:
    """Pick a GL backend the way the walker suite does.

    A spawned render process imports MuJoCo while loading the main module, and
    MuJoCo locks its GL backend at that import. The choice has to be in the
    environment before ``Process.start``. CPU nodes have no ``DISPLAY``; EGL
    needs a GPU, so those nodes use OSMesa. Conda keeps ``libOSMesa.so`` in
    ``$CONDA_PREFIX/lib``, which the dynamic linker does not search.
    """
    backend = os.environ.get("MUJOCO_GL", "").strip()
    if not backend or backend.lower() in {"disable", "disabled", "off", "false", "0"}:
        headless = not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")
        backend = "osmesa" if headless else "egl"
        os.environ["MUJOCO_GL"] = backend
    if backend == "osmesa":
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        prefix = os.environ.get("CONDA_PREFIX", "").strip()
        lib = os.path.join(prefix, "lib") if prefix else ""
        if lib and os.path.isdir(lib):
            current = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(":") if part]
            if lib not in current:
                os.environ["LD_LIBRARY_PATH"] = ":".join([lib, *current])

    config_dir = os.environ.get("MPLCONFIGDIR", "").strip()
    if not config_dir:
        home = os.environ.get("HOME") or ""
        if home and home != "/":
            config_dir = os.path.join(home, ".cache", "matplotlib")
        else:
            config_dir = os.path.join(tempfile.gettempdir(), "matplotlib")
        os.makedirs(config_dir, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = config_dir
    return backend


def start_offscreen(proc) -> str:
    """Start ``proc`` with :func:`prepare_offscreen_env`, then restore this process."""
    saved = {key: os.environ.get(key) for key in _OFFSCREEN_KEYS}
    backend = prepare_offscreen_env()
    try:
        proc.start()
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return backend


def _render_clip_worker(job: dict) -> None:
    """Child-process entry: load model, render buffered qpos, write mp4."""
    prepare_offscreen_env()
    import mediapy as media
    import mujoco

    if job.get("recipe"):
        from loka.scene_props import build_plant_model

        model, _ = build_plant_model(job["recipe"])
    else:
        model = mujoco.MjModel.from_xml_path(job["model_path"])
    from loka.viz import hide_foot_contact_markers

    hide_foot_contact_markers(model)
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
    recipe: dict | None = None,
) -> None:
    """Render ``qpos_frames`` to ``out_path`` in a spawned EGL process.

    ``recipe`` (``Simulation.recipe``) rebuilds a plant with props or the head
    camera, whose ``nq`` differs from the plain MJCF at ``model_path``.
    """
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
        "recipe": recipe,
    }
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_render_clip_worker, args=(job,))
    start_offscreen(proc)
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
        recipe: dict | None = None,
    ):
        if fps <= 0:
            raise ValueError(f"Recording fps must be positive, got {fps}")
        if model_path is None:
            raise ValueError("model_path is required to render after the viewer exits")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.path = str(path)
        self.model_path = str(model_path)
        self.recipe = recipe
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
            recipe=self.recipe,
        )
        self._qpos_frames.clear()
        print(
            f"[LOKA] Saved recording ({self.frame_count} frames @ {self.fps:.1f} fps) "
            f"to {self.path}"
        )


@dataclass(frozen=True)
class OverviewCamera:
    """Fixed world camera for ``run_loka --record``.

    The camera sits at ``pos`` and looks along ``azimuth_deg`` (CCW from +x;
    90 is +y), tilted ``elevation_deg`` (negative is down). It never moves, so
    the clip shows the whole scene rather than following the robot.
    """

    pos: tuple[float, float, float] = (2.5, -4.5, 3.0)
    azimuth_deg: float = 90.0
    elevation_deg: float = -35.0
    fovy_deg: float = 45.0
    fps: float = DEFAULT_RECORD_FPS
    width: int = 1920
    height: int = 1080

    @classmethod
    def from_config(cls, cfg: dict | None) -> "OverviewCamera":
        cfg = dict(cfg or {})
        fields = {key: cfg[key] for key in cls.__dataclass_fields__ if key in cfg}
        if "pos" in fields:
            fields["pos"] = tuple(float(v) for v in fields["pos"])
        return cls(**fields)

    def free_camera(self) -> dict:
        """``mjvCamera`` free-camera fields: look at the floor point on the axis."""
        az = math.radians(self.azimuth_deg)
        el = math.radians(self.elevation_deg)
        forward = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        pos = np.asarray(self.pos, dtype=float)
        distance = pos[2] / -forward[2] if forward[2] < -1e-3 else 5.0
        return {
            "lookat": (pos + distance * forward).tolist(),
            "distance": float(distance),
            "azimuth": float(self.azimuth_deg),
            "elevation": float(self.elevation_deg),
            "fovy": float(self.fovy_deg),
        }


def _render_session_worker(job: dict) -> None:
    """Child-process entry: replay plant snapshots from the overview camera."""
    prepare_offscreen_env()
    import mediapy as media
    import mujoco

    from loka.scene_props import apply_snapshot, build_plant_model
    from loka.viz import hide_foot_contact_markers

    with open(job["frames_path"], "rb") as handle:
        snapshots = pickle.load(handle)
    model, _ = build_plant_model(job["recipe"])
    hide_foot_contact_markers(model)
    width, height = int(job["width"]), int(job["height"])
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    view = job["view"]
    model.vis.global_.fovy = float(view["fovy"])
    data = mujoco.MjData(model)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = view["lookat"]
    cam.distance = view["distance"]
    cam.azimuth = view["azimuth"]
    cam.elevation = view["elevation"]
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        with media.VideoWriter(job["out_path"], shape=(height, width), fps=float(job["fps"])) as writer:
            for snapshot in snapshots:
                apply_snapshot(model, data, snapshot)
                renderer.update_scene(data, camera=cam)
                writer.add_image(np.asarray(renderer.render()))
    finally:
        renderer.close()


class SessionRecorder:
    """Whole-session clip from :class:`OverviewCamera` for ``run_loka --record``.

    :meth:`maybe_capture` copies the plant state (robot, props, static prop
    poses) at ``camera.fps`` of sim time on the sim thread. :meth:`close`
    renders the mp4 in a spawned offscreen process after the run.
    """

    def __init__(self, sim, path: str | Path, camera: OverviewCamera) -> None:
        if camera.fps <= 0:
            raise ValueError(f"Recording fps must be positive, got {camera.fps}")
        self.sim = sim
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.camera = camera
        self.frame_interval = 1.0 / float(camera.fps)
        self.next_frame_time = float(sim.data.time)
        self._snapshots: list[dict] = []
        self._closed = False

    @property
    def frame_count(self) -> int:
        return len(self._snapshots)

    def maybe_capture(self) -> None:
        time_s = float(self.sim.data.time)
        if self._closed or time_s + 1e-9 < self.next_frame_time:
            return
        self._snapshots.append(self.sim.snapshot())
        while self.next_frame_time <= time_s + 1e-9:
            self.next_frame_time += self.frame_interval

    def close(self) -> Path | None:
        if self._closed:
            return None
        self._closed = True
        if not self._snapshots:
            print(f"[record] nothing captured; {self.path} not written")
            return None
        cam = self.camera
        print(
            f"[record] encoding {self.frame_count} frames @ {cam.fps:g} fps "
            f"({self.frame_count / cam.fps:.1f} s) -> {self.path} ..."
        )
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            pickle.dump(self._snapshots, tmp, protocol=pickle.HIGHEST_PROTOCOL)
            frames_path = tmp.name
        self._snapshots.clear()
        job = {
            "recipe": self.sim.recipe,
            "frames_path": frames_path,
            "out_path": str(self.path),
            "fps": float(cam.fps),
            "width": int(cam.width),
            "height": int(cam.height),
            "view": cam.free_camera(),
        }
        ctx = mp.get_context("spawn")
        proc = ctx.Process(target=_render_session_worker, args=(job,))
        backend = start_offscreen(proc)
        proc.join()
        try:
            os.remove(frames_path)
        except OSError:
            pass
        if proc.exitcode != 0:
            print(f"[record] render failed (exit {proc.exitcode}, gl={backend})")
            return None
        print(f"[record] saved {self.path}")
        return self.path


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
        recipe: dict | None = None,
    ):
        self.model = model
        self.model_path = str(model_path)
        self.recipe = recipe
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
            recipe=self.recipe,
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
