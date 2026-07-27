"""Real-time DearPyGui dashboard for CPG / MPC parameter tuning."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class SharedControlState:
    """Thread-safe control / tuning state shared between UI and main loop."""

    walking: bool = False
    estop: bool = False
    reset_requested: bool = False
    freeze_arms: bool = True
    freeze_waist: bool = False  # hips/waist unlocked by default
    enable_leg_torque: bool = False
    planner: str = "Convex QP"

    # CPG
    stride_period: float = 0.8
    clearance: float = 0.0
    sweep_amplitude: float = 0.0
    duty_factor: float = 0.6

    # Torso
    v_ref: float = 0.0
    y_offset: float = 0.0
    z_ref: float = 0.793
    theta_ref: float = 0.0

    # MPC weights
    w_z: float = 50.0
    w_p: float = 20.0
    w_u: float = 1.0
    w_theta: float = 10.0
    w_q: float = 5.0
    w_tau: float = 0.01

    # Status (controller → UI)
    status_text: str = "idle"
    last_cost: float = 0.0

    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def snapshot(self) -> "SharedControlState":
        """Return a shallow copy of tunable fields under lock."""
        with self._lock:
            return SharedControlState(
                walking=self.walking,
                estop=self.estop,
                reset_requested=self.reset_requested,
                freeze_arms=self.freeze_arms,
                freeze_waist=self.freeze_waist,
                enable_leg_torque=self.enable_leg_torque,
                planner=self.planner,
                stride_period=self.stride_period,
                clearance=self.clearance,
                sweep_amplitude=self.sweep_amplitude,
                duty_factor=self.duty_factor,
                v_ref=self.v_ref,
                y_offset=self.y_offset,
                z_ref=self.z_ref,
                theta_ref=self.theta_ref,
                w_z=self.w_z,
                w_p=self.w_p,
                w_u=self.w_u,
                w_theta=self.w_theta,
                w_q=self.w_q,
                w_tau=self.w_tau,
                status_text=self.status_text,
                last_cost=self.last_cost,
            )

    def apply_from(self, other: "SharedControlState") -> None:
        """Copy tunable fields from another snapshot (UI → controller)."""
        with self._lock:
            for name in (
                "walking",
                "estop",
                "reset_requested",
                "freeze_arms",
                "freeze_waist",
                "enable_leg_torque",
                "planner",
                "stride_period",
                "clearance",
                "sweep_amplitude",
                "duty_factor",
                "v_ref",
                "y_offset",
                "z_ref",
                "theta_ref",
                "w_z",
                "w_p",
                "w_u",
                "w_theta",
                "w_q",
                "w_tau",
            ):
                setattr(self, name, getattr(other, name))

    def set_status(self, text: str, cost: float = 0.0) -> None:
        with self._lock:
            self.status_text = text
            self.last_cost = cost

    def clear_reset(self) -> None:
        with self._lock:
            self.reset_requested = False

    def clear_estop(self) -> None:
        with self._lock:
            self.estop = False


class DashboardUI:
    """DearPyGui front-end running in a daemon thread."""

    PLANNER_OPTIONS = ("Convex QP", "Predictive Sampling", "iLQG")
    # Uniform UI scale (text, padding, control sizes, window).
    UI_SCALE = 2.0

    def __init__(
        self,
        shared: SharedControlState,
        *,
        title: str = "LOKA G1 CPG + MPC",
        width: int = 1040,
        height: int = 1560,
        on_start: Optional[Callable[[], None]] = None,
        on_stop: Optional[Callable[[], None]] = None,
        on_estop: Optional[Callable[[], None]] = None,
    ) -> None:
        self.shared = shared
        self.title = title
        self.width = width
        self.height = height
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_estop = on_estop
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def _s(self, value: float) -> int:
        """Scale a pixel size by ``UI_SCALE``."""
        return int(round(value * self.UI_SCALE))

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="loka-dashboard", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        try:
            import dearpygui.dearpygui as dpg

            if dpg.is_dearpygui_running():
                dpg.stop_dearpygui()
        except Exception:
            pass

    def _run(self) -> None:
        try:
            import dearpygui.dearpygui as dpg
        except ImportError:
            logger.error(
                "dearpygui is not installed. Install with: pip install dearpygui"
            )
            self._running = False
            return

        s = self.shared
        scale = self.UI_SCALE
        dpg.create_context()
        dpg.set_global_font_scale(scale)

        # Enlarge frames, spacing, and slider grabs to match 2x text.
        with dpg.theme() as large_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, self._s(4), self._s(3)
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_ItemSpacing, self._s(8), self._s(4)
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, self._s(8), self._s(8)
                )
                dpg.add_theme_style(dpg.mvStyleVar_GrabMinSize, self._s(10))
                dpg.add_theme_style(dpg.mvStyleVar_ScrollbarSize, self._s(14))
                dpg.add_theme_style(
                    dpg.mvStyleVar_FrameRounding, self._s(3)
                )
        dpg.bind_theme(large_theme)

        slider_width = self._s(300)
        button_width = self._s(150)
        estop_width = self._s(310)
        combo_width = self._s(220)

        def _sync_from_ui() -> None:
            with s._lock:
                s.freeze_arms = bool(dpg.get_value("chk_freeze_arms"))
                s.freeze_waist = bool(dpg.get_value("chk_freeze_waist"))
                s.enable_leg_torque = bool(dpg.get_value("chk_mixed"))
                s.planner = str(dpg.get_value("combo_planner"))
                s.stride_period = float(dpg.get_value("sl_T"))
                s.clearance = float(dpg.get_value("sl_H"))
                s.sweep_amplitude = float(dpg.get_value("sl_AS"))
                s.duty_factor = float(dpg.get_value("sl_d"))
                s.v_ref = float(dpg.get_value("sl_vref"))
                s.y_offset = float(dpg.get_value("sl_yoff"))
                s.z_ref = float(dpg.get_value("sl_zref"))
                s.theta_ref = float(dpg.get_value("sl_theta"))
                s.w_z = float(dpg.get_value("sl_wz"))
                s.w_p = float(dpg.get_value("sl_wp"))
                s.w_u = float(dpg.get_value("sl_wu"))
                s.w_theta = float(dpg.get_value("sl_wtheta"))
                s.w_q = float(dpg.get_value("sl_wq"))
                s.w_tau = float(dpg.get_value("sl_wtau"))

        def _on_start() -> None:
            with s._lock:
                s.walking = True
                s.estop = False
            _sync_from_ui()
            if self.on_start:
                self.on_start()

        def _on_stop() -> None:
            with s._lock:
                s.walking = False
            if self.on_stop:
                self.on_stop()

        def _on_estop() -> None:
            with s._lock:
                s.estop = True
                s.walking = False
                s.reset_requested = True
            if self.on_estop:
                self.on_estop()

        def _on_change(_sender=None, _app_data=None, _user_data=None) -> None:
            _sync_from_ui()

        with dpg.window(label=self.title, tag="primary", width=self.width, height=self.height):
            dpg.add_text("LOKA — Torso Position MPC (no CPG)")
            dpg.add_separator()

            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Start Stand / MPC", width=button_width, height=self._s(28), callback=_on_start
                )
                dpg.add_button(
                    label="Stop", width=button_width, height=self._s(28), callback=_on_stop
                )
            dpg.add_button(
                label="Emergency Stop / Reset",
                width=estop_width,
                height=self._s(28),
                callback=_on_estop,
            )
            dpg.add_separator()

            dpg.add_text("Baby-step: torso height/attitude tracking (no CPG)")
            dpg.add_checkbox(
                label="Freeze Arms",
                tag="chk_freeze_arms",
                default_value=s.freeze_arms,
                callback=_on_change,
            )
            dpg.add_checkbox(
                label="Freeze Waist / Hips",
                tag="chk_freeze_waist",
                default_value=s.freeze_waist,
                callback=_on_change,
            )
            dpg.add_checkbox(
                label="Enable Mixed Mode (unused)",
                tag="chk_mixed",
                default_value=False,
                callback=_on_change,
                show=False,
            )
            dpg.add_combo(
                items=list(self.PLANNER_OPTIONS),
                tag="combo_planner",
                label="MPC Planner",
                default_value=s.planner,
                width=combo_width,
                callback=_on_change,
            )
            dpg.add_separator()

            dpg.add_text("Torso target")
            dpg.add_slider_float(
                tag="sl_zref",
                label="Target Base Height z_ref [m]",
                default_value=s.z_ref,
                min_value=0.5,
                max_value=0.85,
                width=slider_width,
                callback=_on_change,
            )
            # Unused CPG / path sliders kept hidden but present for sync safety.
            for tag, val in (
                ("sl_T", s.stride_period),
                ("sl_H", s.clearance),
                ("sl_AS", s.sweep_amplitude),
                ("sl_d", s.duty_factor),
                ("sl_vref", s.v_ref),
                ("sl_yoff", s.y_offset),
                ("sl_theta", s.theta_ref),
            ):
                dpg.add_slider_float(
                    tag=tag,
                    default_value=val,
                    min_value=-1.0,
                    max_value=2.0,
                    show=False,
                    callback=_on_change,
                )
            dpg.add_separator()

            dpg.add_text("MPC Weights (torso tracking)")
            for tag, label, val, lo, hi in (
                ("sl_wz", "w_z (height)", s.w_z, 0.0, 200.0),
                ("sl_wp", "w_xy (horizontal)", s.w_p, 0.0, 100.0),
                ("sl_wtheta", "w_theta (pitch/roll)", s.w_theta, 0.0, 100.0),
                ("sl_wq", "w_q (stay near stand pose)", s.w_q, 0.0, 50.0),
                ("sl_wu", "w_u (unused)", s.w_u, 0.0, 50.0),
                ("sl_wtau", "w_tau (unused)", s.w_tau, 0.0, 1.0),
            ):
                dpg.add_slider_float(
                    tag=tag,
                    label=label,
                    default_value=val,
                    min_value=lo,
                    max_value=hi,
                    width=slider_width,
                    callback=_on_change,
                    show=tag not in ("sl_wu", "sl_wtau"),
                )

            dpg.add_separator()
            dpg.add_text("Status:", tag="txt_status_label")
            dpg.add_text("idle", tag="txt_status")

        def _update_status() -> None:
            with s._lock:
                text = f"{s.status_text} | cost={s.last_cost:.3f}"
            if dpg.does_item_exist("txt_status"):
                dpg.set_value("txt_status", text)

        with dpg.handler_registry():
            pass

        dpg.create_viewport(
            title=self.title,
            width=self.width + self._s(20),
            height=self.height + self._s(40),
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("primary", True)

        while dpg.is_dearpygui_running() and self._running:
            _update_status()
            dpg.render_dearpygui_frame()

        dpg.destroy_context()
        self._running = False
