"""Egocentric head camera: mount, intrinsics, pose, and an offscreen renderer.

The camera ``ego_cam`` is added to the plant model only (see
:mod:`loka.scene_props`), midway between the G1 head's stereo cameras, on ``torso_link``. The
belief model the controller plans with never sees it.

An in-process ``mujoco.Renderer`` fails next to the GLFW viewer under WSLg
(see :mod:`loka.recording`), so :class:`EgoRenderer` keeps one spawned process
alive that rebuilds the identical plant from its recipe and renders a qpos
snapshot on request. Nothing here blocks the control loop: the snapshot is a
copy, and the render runs on the LLM worker thread.
"""

from __future__ import annotations

import io
import math
import multiprocessing as mp
import threading
import time
from dataclasses import asdict, dataclass

import numpy as np

EGO_CAMERA = "ego_cam"
EGO_MOUNT_BODY = "torso_link"
#: torso_link frame [m]. Centre of the head's stereo-camera window: the recess
#: in head_link.STL spans z 0.412-0.432 and y +/-0.028 under the brow, whose
#: front is at x 0.078. The optical centre sits just inside the brow line.
EGO_MOUNT_POS = (0.075, 0.0, 0.422)
RENDER_TIMEOUT_S = 20.0


@dataclass(frozen=True)
class EgoCameraSpec:
    """Pinhole RGB camera. Square pixels, principal point at the image centre."""

    width: int = 512
    height: int = 512
    fovy_deg: float = 70.0
    pitch_down_deg: float = 30.0
    detail: str = "high"
    save_frames: bool = True

    @classmethod
    def from_config(cls, cfg: dict | None) -> "EgoCameraSpec":
        cfg = dict(cfg or {})
        fields = {key: cfg[key] for key in cls.__dataclass_fields__ if key in cfg}
        return cls(**fields)

    @property
    def fy(self) -> float:
        return 0.5 * self.height / math.tan(math.radians(self.fovy_deg) / 2.0)

    @property
    def fx(self) -> float:
        return self.fy

    @property
    def cx(self) -> float:
        return 0.5 * self.width

    @property
    def cy(self) -> float:
        return 0.5 * self.height

    @property
    def fovx_deg(self) -> float:
        return math.degrees(2.0 * math.atan(0.5 * self.width / self.fx))

    @property
    def pitch_rad(self) -> float:
        return math.radians(self.pitch_down_deg)

    def mount_quat(self) -> np.ndarray:
        """Camera orientation in the mount body frame, wxyz.

        A MuJoCo camera looks along its -z axis with +x right and +y up in the
        image. Looking along body +x, pitched down: right is body -y, up is
        (sin p, 0, cos p), and -z is the optical axis (cos p, 0, -sin p).
        """
        import mujoco

        p = self.pitch_rad
        right = np.array([0.0, -1.0, 0.0])
        up = np.array([math.sin(p), 0.0, math.cos(p)])
        back = np.cross(right, up)
        mat = np.column_stack([right, up, back]).reshape(-1)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat)
        return quat

    def recipe(self) -> dict:
        return asdict(self)


def add_ego_camera(spec, camera: EgoCameraSpec) -> None:
    """Mount ``ego_cam`` on the head of the G1 in an ``MjSpec``."""
    body = spec.find_body(EGO_MOUNT_BODY)
    if body is None:
        raise ValueError(f"camera mount body {EGO_MOUNT_BODY!r} not in the model")
    cam = body.add_camera()
    cam.name = EGO_CAMERA
    cam.pos = list(EGO_MOUNT_POS)
    cam.quat = camera.mount_quat().tolist()
    cam.fovy = float(camera.fovy_deg)


def fit_offscreen_buffer(model, camera: EgoCameraSpec) -> None:
    """Grow the compiled model's offscreen buffer to hold one camera frame."""
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), camera.width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), camera.height)


@dataclass(frozen=True)
class CameraPose:
    """World pose of ``ego_cam``. Yaw is the optical axis heading, CCW from +x."""

    time: float
    xyz: tuple[float, float, float]
    yaw: float
    pitch_down: float
    roll: float

    def describe(self) -> str:
        x, y, z = self.xyz
        return (
            f"camera xyz [m] {x:+.3f} {y:+.3f} {z:+.3f}; "
            f"yaw {self.yaw:+.3f} rad ({math.degrees(self.yaw):+.1f} deg); "
            f"pitch down {math.degrees(self.pitch_down):+.1f} deg; "
            f"roll {math.degrees(self.roll):+.1f} deg"
        )


def camera_pose(model, data) -> CameraPose:
    """Pose at the current ``qpos``. Refreshes only the camera and light frames."""
    import mujoco

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, EGO_CAMERA)
    if cam_id < 0:
        raise ValueError(f"model has no camera {EGO_CAMERA!r}")
    mujoco.mj_camlight(model, data)
    rot = np.asarray(data.cam_xmat[cam_id], dtype=float).reshape(3, 3)
    forward = -rot[:, 2]
    right = rot[:, 0]
    yaw = math.atan2(forward[1], forward[0])
    pitch_down = math.asin(float(np.clip(-forward[2], -1.0, 1.0)))
    # Roll: right axis tilt out of the horizontal plane, positive when the
    # image rotates clockwise (right side down).
    roll = math.asin(float(np.clip(-right[2], -1.0, 1.0)))
    xyz = tuple(float(v) for v in data.cam_xpos[cam_id])
    return CameraPose(float(data.time), xyz, yaw, pitch_down, roll)


def nominal_camera_pose(model) -> CameraPose:
    """Camera pose in the ``stand`` keyframe, on a scratch ``MjData``."""
    import mujoco

    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return camera_pose(model, data)


def floor_cell_size(model, geom: str = "floor") -> float | None:
    """Side of one floor checker square [m], or None if the floor is untextured.

    A uniform-mapped checker on the infinite floor plane renders one square
    per ``1 / texrepeat`` m. Checked against a top-down render of
    ``models/g1/scene.xml`` (texrepeat 5: five squares per metre, with the
    edge marks drawn every second square).
    """
    import mujoco

    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
    if gid < 0:
        return None
    mat = int(model.geom_matid[gid])
    if mat < 0 or not bool(model.mat_texuniform[mat]):
        return None
    repeat = float(model.mat_texrepeat[mat, 0])
    return 1.0 / repeat if repeat > 0 else None


def encode_png(rgb: np.ndarray) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def _ego_render_worker(conn, recipe: dict) -> None:
    """Child process: build the plant once, then render each snapshot sent."""
    from loka.recording import prepare_offscreen_env

    prepare_offscreen_env()
    import mujoco

    from loka.scene_props import apply_snapshot, build_plant_model
    from loka.viz import hide_foot_contact_markers

    try:
        model, _ = build_plant_model(recipe)
        hide_foot_contact_markers(model)
        data = mujoco.MjData(model)
        camera = recipe["camera"]
        renderer = mujoco.Renderer(model, height=int(camera["height"]), width=int(camera["width"]))
    except Exception as exc:  # report and exit; the parent falls back to text
        conn.send((False, f"renderer init failed: {exc}"))
        return
    conn.send((True, "ready"))
    try:
        while True:
            message = conn.recv()
            if message is None:
                break
            # ``("rgb"|"png", snapshot)`` or a bare snapshot (PNG, legacy).
            if isinstance(message, tuple) and len(message) == 2:
                mode, snapshot = message
            else:
                mode, snapshot = "png", message
            try:
                apply_snapshot(model, data, snapshot)
                renderer.update_scene(data, camera=EGO_CAMERA)
                rgb = renderer.render()
                if mode == "rgb":
                    conn.send((True, np.asarray(rgb, dtype=np.uint8)))
                else:
                    conn.send((True, encode_png(rgb)))
            except Exception as exc:
                conn.send((False, f"render failed: {exc}"))
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        renderer.close()


class EgoRenderer:
    """One persistent offscreen render process for ``ego_cam``. Thread-safe."""

    def __init__(self, recipe: dict) -> None:
        from loka.recording import start_offscreen

        if not recipe.get("camera"):
            raise ValueError("plant recipe has no camera")
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_ego_render_worker, args=(child, recipe), daemon=True)
        self._lock = threading.Lock()
        self.backend = start_offscreen(self._proc)
        child.close()
        self.error: str | None = None
        self._ready = False

    def _wait_ready(self) -> None:
        if self._ready:
            return
        if not self._conn.poll(RENDER_TIMEOUT_S * 3):
            raise RuntimeError("ego renderer did not start")
        ok, message = self._conn.recv()
        if not ok:
            self.error = str(message)
            raise RuntimeError(self.error)
        self._ready = True

    def render(self, snapshot: dict) -> bytes:
        """PNG bytes of ``snapshot`` from ``ego_cam``."""
        return self._request("png", snapshot)

    def render_rgb(self, snapshot: dict) -> np.ndarray:
        """Raw HxWx3 uint8 RGB of ``snapshot`` from ``ego_cam``."""
        return self._request("rgb", snapshot)

    def _request(self, mode: str, snapshot: dict):
        with self._lock:
            if self.error is not None:
                raise RuntimeError(self.error)
            if not self._proc.is_alive() and self._ready:
                raise RuntimeError("ego renderer process exited")
            self._wait_ready()
            self._conn.send((mode, snapshot))
            if not self._conn.poll(RENDER_TIMEOUT_S):
                raise RuntimeError(f"ego render timed out after {RENDER_TIMEOUT_S:.0f}s")
            ok, payload = self._conn.recv()
            if not ok:
                raise RuntimeError(str(payload))
            return payload

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.send(None)
            except (BrokenPipeError, OSError):
                pass
            self._proc.join(timeout=5.0)
            if self._proc.is_alive():
                self._proc.terminate()


def _ego_view_worker(conn, recipe: dict) -> None:
    """Child process: render each snapshot sent and show it in a Tk window.

    Replies ``"ready"`` once, then ``"frame"`` after every displayed snapshot
    so the parent never queues more than one, and ``"closed"`` when the user
    closes the window.
    """
    from loka.recording import prepare_offscreen_env

    prepare_offscreen_env()
    import tkinter as tk

    import mujoco

    from loka.scene_props import apply_snapshot, build_plant_model
    from loka.viz import hide_foot_contact_markers

    try:
        model, _ = build_plant_model(recipe)
        hide_foot_contact_markers(model)
        data = mujoco.MjData(model)
        camera = recipe["camera"]
        renderer = mujoco.Renderer(model, height=int(camera["height"]), width=int(camera["width"]))
        root = tk.Tk()
    except Exception as exc:
        conn.send(f"error: {exc}")
        return
    root.title(f"LOKA {EGO_CAMERA} ({camera['width']}x{camera['height']})")
    image_label = tk.Label(root)
    image_label.pack()
    caption = tk.Label(root, font=("TkFixedFont", 9), anchor="w", justify="left")
    caption.pack(fill="x")
    state = {"open": True}

    def _close() -> None:
        state["open"] = False
        try:
            conn.send("closed")
        except (BrokenPipeError, OSError):
            pass
        root.destroy()

    def _pump() -> None:
        if not state["open"]:
            return
        try:
            if conn.poll(0):
                snapshot = conn.recv()
                if snapshot is None:
                    state["open"] = False
                    root.destroy()
                    return
                apply_snapshot(model, data, snapshot)
                renderer.update_scene(data, camera=EGO_CAMERA)
                rgb = renderer.render()
                header = f"P6 {rgb.shape[1]} {rgb.shape[0]} 255 ".encode()
                photo = tk.PhotoImage(data=header + rgb.tobytes(), format="ppm")
                image_label.configure(image=photo)
                image_label.image = photo
                caption.configure(text=camera_pose(model, data).describe())
                conn.send("frame")
        except (EOFError, BrokenPipeError, OSError):
            state["open"] = False
            root.destroy()
            return
        root.after(5, _pump)

    root.protocol("WM_DELETE_WINDOW", _close)
    conn.send("ready")
    root.after(0, _pump)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        renderer.close()


class EgoViewStream:
    """Live ``ego_cam`` window for run_loka, fed from the viewer loop.

    :meth:`offer` never blocks: it sends a snapshot only when ``hz`` allows
    and the window has finished the previous frame, so a slow render drops
    frames instead of slowing the simulation.
    """

    def __init__(self, recipe: dict, hz: float = 15.0) -> None:
        from loka.recording import start_offscreen

        if not recipe.get("camera"):
            raise ValueError("plant recipe has no camera")
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_ego_view_worker, args=(child, recipe), daemon=True)
        self.backend = start_offscreen(self._proc)
        child.close()
        self.period = 1.0 / max(float(hz), 0.1)
        self._busy = True  # until "ready"
        self._next = 0.0
        self.open = True
        self.frames = 0

    def _drain(self) -> None:
        try:
            while self._conn.poll(0):
                message = self._conn.recv()
                if message in ("ready", "frame"):
                    self._busy = False
                    self.frames += message == "frame"
                elif message == "closed":
                    self.open = False
                    print("[ego view] window closed")
                else:
                    self.open = False
                    print(f"[ego view] {message}")
        except (EOFError, OSError):
            self.open = False

    def offer(self, sim) -> None:
        """Send ``sim``'s current state if the window is ready for a frame."""
        if not self.open:
            return
        self._drain()
        now = time.perf_counter()
        if not self.open or self._busy or now < self._next:
            return
        try:
            self._conn.send(sim.snapshot())
        except (BrokenPipeError, OSError):
            self.open = False
            return
        self._busy = True
        self._next = now + self.period

    def close(self) -> None:
        try:
            self._conn.send(None)
        except (BrokenPipeError, OSError):
            pass
        self._proc.join(timeout=5.0)
        if self._proc.is_alive():
            self._proc.terminate()
