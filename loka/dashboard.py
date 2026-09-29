#!/usr/bin/env python3
"""Manual tuning dashboard: stand and walk the G1 by hand, no LLM in the loop.

    python -m loka.dashboard              # sliders + MuJoCo viewer
    python -m loka.dashboard --no-robot   # sliders only, if the GPU is busy
    python -m loka.dashboard --walk 0.25  # start walking immediately

Every widget writes through one of the two surfaces the orchestrator uses:
weights and gains through :mod:`loka.control.tuning`, and task setpoints --
height, yaw, lean, and the whole ``gait.*`` policy -- through
``set_task_targets``. So a knob reachable by hand here is reachable by the
model later, with identical names, ranges and clamping, and a clamp that bites
is visible in the readout beside the slider rather than silently discarded.

Everything runs on one thread. The controller mutates ``MjData`` in place and
``viewer.sync()`` reads it, so stepping from a background thread is a genuine
data race (it segfaults); driving the physics from the GUI's own frame loop
sidesteps that entirely and costs nothing, because a frame is 8 control ticks
and both fit comfortably inside 16 ms.

Widget payloads travel through DearPyGui's ``user_data``. This is not a style
preference -- see :meth:`Dashboard._on_tunable` for why the obvious
alternative silently breaks.
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import dearpygui.dearpygui as dpg
import numpy as np

from loka import viz
from loka.control import tuning
from loka.control.gait import (
    GAIT_KNOBS,
    MODE_NAME_TO_CODE,
    gait_mode_name,
    heading_frame,
)
from loka.control.locomotion import DEFAULT_MODEL, LocomotionConfig
from loka.recording import RecordingToggle
from loka.sim import Push, Simulation

RENDER_HZ = 60.0
#: The robot window is redrawn slower than the plots -- a standing robot barely
#: moves between frames. Worth a little, but only a little: on a software GL
#: stack (WSLg) most of the cost is having a second window open at all, so
#: ``--no-robot`` is what actually buys back realtime.
ROBOT_HZ = 30.0
#: Seconds of telemetry kept on screen.
HISTORY_SECONDS = 10.0

MODE_NAMES = tuple(MODE_NAME_TO_CODE)


@dataclass(frozen=True)
class Trace:
    """One line on a plot: where its samples come from, and what to call it."""

    key: str
    label: str


@dataclass(frozen=True)
class Plot:
    tag: str
    label: str
    traces: tuple[Trace, ...]


#: Balance plots read the same whether or not the robot is walking; the walk
#: plots read zero while it stands, which is itself the useful answer.
PLOTS = (
    Plot("com_error", "CoM error [mm]", (Trace("com_error", "|error|"),)),
    Plot("tilt", "torso tilt [deg]", (Trace("tilt", "tilt"),)),
    Plot("torque", "peak |torque| [Nm]", (Trace("torque", "peak"),)),
    Plot("solve_ms", "controller [ms]", (Trace("solve_ms", "solve"),)),
    Plot(
        "speed",
        "forward speed [m/s]",
        (Trace("speed", "measured"), Trace("speed_ref", "plan")),
    ),
    # The open walking defect is lateral: the sagittal axis tracks, and the
    # robot topples sideways over eight to sixteen steps. Signed, in the
    # heading frame, so a one-sided drift is distinguishable from sway.
    Plot("lateral", "lateral CoM error [mm]", (Trace("lateral", "signed"),)),
    Plot(
        "swing",
        "swing foot [mm]",
        (Trace("swing_clearance", "clearance"), Trace("swing_error", "tracking")),
    ),
)

TRACE_KEYS = tuple(trace.key for plot in PLOTS for trace in plot.traces)


class Dashboard:
    def __init__(self, sim: Simulation, show_robot: bool = True) -> None:
        self.sim = sim
        self.show_robot = show_robot
        self.viewer = None
        self.running = True
        self.steps_per_frame = max(
            1, round(1.0 / (RENDER_HZ * sim.config.control_dt))
        )
        self.robot_sync_every = max(1, round(RENDER_HZ / ROBOT_HZ))
        self._frame = 0
        depth = int(HISTORY_SECONDS / (self.steps_per_frame * sim.config.control_dt))
        self.trace: dict[str, deque] = {
            key: deque(maxlen=depth) for key in ("t",) + TRACE_KEYS
        }
        self._defaults = sim.controller.tunables()
        self._task_defaults = sim.controller.task_snapshot()
        self._origin = np.array(sim.data.qpos[:2], dtype=float)
        self._last_wall = time.perf_counter()
        self._realtime = 0.0
        self.recording = RecordingToggle(sim.model, sim.config.model_path)

    # -- controller plumbing ----------------------------------------------
    #
    # Split from the widget callbacks below so the logic can be exercised
    # without a display.

    def _set_tunable(self, path: str, value: float) -> float:
        """Apply a real (already de-logged) value; return what the clamp kept."""
        return self.sim.controller.update_weights(**{path: value})[path]

    def _set_task(self, name: str, value: float | str) -> dict[str, float]:
        """Apply a task setpoint through the orchestrator's own apply path.

        Returns everything that moved, which is not always just ``name``:
        asking for walk with no speed set also picks a default crawl, and
        starting a walk aligns the torso with the heading.
        """
        return self.sim.controller.set_task_targets({name: value})

    def _push(self, direction, impulse: float = 6.0) -> None:
        direction = np.asarray(direction, dtype=float).reshape(3)
        self.sim.pushes.append(
            Push(impulse=impulse, direction=direction, time=self.sim.data.time)
        )

    def _reset(self) -> None:
        self.sim.pushes.clear()
        self.sim.reset()
        self._origin = np.array(self.sim.data.qpos[:2], dtype=float)
        for buffer in self.trace.values():
            buffer.clear()

    # -- widget callbacks ---------------------------------------------------
    #
    # DearPyGui decides how many arguments to hand a callback by reading
    # ``co_argcount``, which counts parameters that merely *have* defaults. So
    # the usual Python idiom for capturing a loop variable --
    # ``lambda s, v, path=entry.path: ...`` -- registers as a three-argument
    # callback, and DearPyGui dutifully overwrites the captured default with
    # ``user_data``. Payloads have to travel through ``user_data`` instead, and
    # every callback here takes the full ``(sender, app_data, user_data)``
    # triple so the convention is impossible to get subtly wrong.

    def _on_tunable(self, sender, value: float, path: str) -> None:
        entry = tuning.BY_PATH[path]
        applied = self._set_tunable(path, 10.0**value if entry.log else value)
        dpg.set_value(f"readout::{path}", f"{applied:.4g}")

    def _on_task(self, sender, value: float, name: str) -> None:
        self._show_task(self._set_task(name, value))

    def _on_mode(self, sender, value: str, user_data) -> None:
        self._show_task(self._set_task("gait.mode", value))

    def _on_push(self, sender, app_data, direction) -> None:
        self._push(direction, dpg.get_value("impulse"))

    def _on_reset(self, sender, app_data, user_data) -> None:
        self._reset()

    def _on_restore_defaults(self, sender, app_data, user_data) -> None:
        self.sim.controller.update_weights(**self._defaults)
        for path, value in self._defaults.items():
            entry = tuning.BY_PATH[path]
            dpg.set_value(path, np.log10(value) if entry.log else value)
            dpg.set_value(f"readout::{path}", f"{value:.4g}")
        # Mode last: dropping back to stand resets the gait clock, and doing it
        # first would let the remaining setpoints restart a walk.
        gait = {k: v for k, v in self._task_defaults.items() if k != "gait.mode"}
        self._show_task(self.sim.controller.set_task_targets(gait))
        self._show_task(self._set_task("gait.mode", self._task_defaults["gait.mode"]))

    def _on_toggle_running(self, sender, app_data, user_data) -> None:
        self.running = not self.running
        dpg.set_item_label("pause", "pause" if self.running else "resume")

    # -- layout -------------------------------------------------------------

    def _show_task(self, applied: Mapping[str, float]) -> None:
        """Write applied setpoints back onto their widgets.

        Clamps and side effects both land here, so the panel always shows what
        the controller actually holds rather than what was asked for.
        """
        for name, value in applied.items():
            if name == "gait.mode":
                dpg.set_value(name, gait_mode_name(value))
                continue
            if not dpg.does_item_exist(name):
                continue
            dpg.set_value(name, value)
            if dpg.does_item_exist(f"readout::{name}"):
                dpg.set_value(f"readout::{name}", f"{value:.4g}")

    def _add_slider(
        self, tag: str, label: str, value: float, low: float, high: float,
        *, log: bool, callback, tooltip: str,
    ) -> None:
        with dpg.group(horizontal=True):
            dpg.add_slider_float(
                tag=tag,
                label=label,
                width=240,
                default_value=np.log10(value) if log else value,
                min_value=np.log10(low) if log else low,
                max_value=np.log10(high) if log else high,
                # The readout beside the slider is the only number shown. A log
                # slider carries an exponent, which is unreadable; a linear one
                # would just print the same value twice.
                format="",
                user_data=tag,
                callback=callback,
            )
            dpg.add_text(f"{value:.4g}", tag=f"readout::{tag}")
        with dpg.tooltip(tag):
            dpg.add_text(tooltip, wrap=320)
            dpg.add_text(
                f"range [{low:g}, {high:g}]" + ("  (log scale)" if log else ""),
                color=(150, 150, 150),
            )

    def _build_tunables(self, group: str) -> None:
        for entry in tuning.TUNABLES:
            if entry.group != group:
                continue
            self._add_slider(
                entry.path,
                # The knob name, not the attribute: weight_orientation_rp and
                # weight_orientation_yaw both resolve to the same field and
                # would otherwise be two identical labels.
                entry.path.split(".", 1)[1],
                self._defaults[entry.path],
                entry.low,
                entry.high,
                log=entry.log,
                callback=self._on_tunable,
                tooltip=entry.summary,
            )

    def _build_gait(self) -> None:
        dpg.add_combo(
            MODE_NAMES,
            tag="gait.mode",
            label="mode",
            width=120,
            default_value=gait_mode_name(self._task_defaults["gait.mode"]),
            callback=self._on_mode,
        )
        with dpg.tooltip("gait.mode"):
            dpg.add_text(
                "Walking engages the gait clock and the footstep plan. Leaving "
                "walk finishes the step in progress before both feet settle.",
                wrap=320,
            )
        for knob in GAIT_KNOBS:
            if knob.name == "gait.mode":
                continue
            self._add_slider(
                knob.name,
                knob.field,
                self._task_defaults[knob.name],
                knob.low,
                knob.high,
                log=False,
                callback=self._on_task,
                tooltip=knob.summary,
            )

    def _build_command(self) -> None:
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="pause", tag="pause", callback=self._on_toggle_running
            )
            dpg.add_button(label="reset", callback=self._on_reset)
            dpg.add_button(
                label="restore defaults", callback=self._on_restore_defaults
            )
        dpg.add_text("push the pelvis")
        with dpg.group(horizontal=True):
            for label, vector in (
                ("fwd", (1, 0, 0)), ("back", (-1, 0, 0)),
                ("left", (0, 1, 0)), ("right", (0, -1, 0)),
            ):
                dpg.add_button(
                    label=label, user_data=vector, callback=self._on_push
                )
            dpg.add_slider_float(
                tag="impulse", label="N.s", width=90,
                default_value=6.0, min_value=1.0, max_value=25.0, format="%.1f",
            )

        dpg.add_separator()
        limits = self.sim.controller.lean_limits()
        self._add_slider(
            "height", "CoM height [m]", self._task_defaults["height"],
            limits["height_min"], limits["height_max"],
            log=False, callback=self._on_task,
            tooltip="CoM height above the foot plane.",
        )
        self._add_slider(
            "yaw", "yaw [rad]", self._task_defaults["yaw"], -0.6, 0.6,
            log=False, callback=self._on_task,
            tooltip="Torso facing. Starting a walk aligns this with the heading.",
        )

    def _build_ui(self) -> None:
        with dpg.window(tag="root"):
            with dpg.group(horizontal=True):
                # -- left: knobs ------------------------------------------
                with dpg.child_window(width=430, autosize_y=True):
                    dpg.add_text("controller", color=(120, 200, 255))
                    self._build_command()
                    with dpg.collapsing_header(label="gait", default_open=True):
                        self._build_gait()
                    for group, title in (
                        ("wbc", "whole-body QP"),
                        ("mpc", "centroidal MPC"),
                        ("stand", "saturation"),
                    ):
                        with dpg.collapsing_header(label=title):
                            self._build_tunables(group)

                # -- right: telemetry --------------------------------------
                with dpg.child_window(autosize_x=True, autosize_y=True):
                    dpg.add_text("", tag="status")
                    dpg.add_text("", tag="status2", color=(150, 150, 150))
                    dpg.add_text("", tag="status3", color=(150, 190, 150))
                    for plot in PLOTS:
                        with dpg.plot(label=plot.label, height=140, width=-1):
                            if len(plot.traces) > 1:
                                dpg.add_plot_legend()
                            dpg.add_plot_axis(dpg.mvXAxis, tag=f"x::{plot.tag}")
                            axis = dpg.add_plot_axis(
                                dpg.mvYAxis, tag=f"y::{plot.tag}"
                            )
                            for trace in plot.traces:
                                dpg.add_line_series(
                                    [], [], label=trace.label,
                                    parent=axis, tag=f"series::{trace.key}",
                                )
        dpg.set_primary_window("root", True)

    # -- frame --------------------------------------------------------------

    def _advance(self) -> None:
        for _ in range(self.steps_per_frame):
            self.sim.step()
        row = self.sim.history[-1]
        telemetry = self.sim.controller.telemetry
        forward, left = heading_frame(float(telemetry.yaw_ref))

        self.trace["t"].append(row["t"])
        self.trace["com_error"].append(row["com_error"] * 1e3)
        self.trace["tilt"].append(np.degrees(row["tilt"]))
        self.trace["torque"].append(row["torque"])
        self.trace["solve_ms"].append(row["solve_ms"])
        self.trace["speed"].append(float(telemetry.com_velocity[:2] @ forward))
        self.trace["speed_ref"].append(
            float(telemetry.com_velocity_reference[:2] @ forward)
        )
        self.trace["lateral"].append(float(telemetry.com_error[:2] @ left) * 1e3)
        self.trace["swing_clearance"].append(telemetry.swing_clearance * 1e3)
        self.trace["swing_error"].append(telemetry.swing_error * 1e3)

    def _refresh_plots(self) -> None:
        times = list(self.trace["t"])
        if not times:
            return
        for plot in PLOTS:
            for trace in plot.traces:
                dpg.set_value(
                    f"series::{trace.key}", [times, list(self.trace[trace.key])]
                )
            dpg.fit_axis_data(f"x::{plot.tag}")
            dpg.fit_axis_data(f"y::{plot.tag}")

        controller = self.sim.controller
        telemetry = controller.telemetry
        now = time.perf_counter()
        elapsed = now - self._last_wall
        self._last_wall = now
        if elapsed > 0:
            instant = self.steps_per_frame * self.sim.config.control_dt / elapsed
            self._realtime += 0.1 * (instant - self._realtime)

        if self.sim.fell:
            state = "FALLEN"
        elif telemetry.walking:
            state = "walking"
        else:
            state = "standing"
        dpg.set_value(
            "status",
            f"t {self.sim.data.time:7.2f} s   {self._realtime:4.2f}x realtime"
            f"   {state}"
            f"   contacts {int(telemetry.contact_mask.sum())}/8",
        )
        margin = controller.robot.support_margin()
        dpg.set_value(
            "status2",
            f"CoM margin  fwd {margin[1] * 1e3:5.1f} mm   back {margin[0] * 1e3:5.1f} mm"
            f"   |   QP fallbacks  wbc {controller.wbc._qp.failures}"
            f"  mpc {controller.mpc._qp.failures}",
        )
        forward, _ = heading_frame(float(telemetry.yaw_ref))
        travel = float((np.asarray(self.sim.data.qpos[:2]) - self._origin) @ forward)
        dpg.set_value(
            "status3",
            f"gait  phase {telemetry.gait_phase:4.2f}"
            f"   speed {controller.gait.cmd_speed:4.2f} m/s commanded"
            f"   steps {controller.gait.step_index:3d}"
            f"   travelled {travel:6.3f} m",
        )

    def run(self) -> None:
        dpg.create_context()
        dpg.create_viewport(title="LOKA - G1 locomotion", width=1280, height=900)
        self._build_ui()
        dpg.setup_dearpygui()
        dpg.show_viewport()

        if self.show_robot:
            import mujoco
            import mujoco.viewer

            print("[LOKA] Press R in the MuJoCo viewer to start/stop recording.")
            self.viewer = mujoco.viewer.launch_passive(
                self.sim.model,
                self.sim.data,
                show_left_ui=False,
                show_right_ui=False,
                key_callback=lambda keycode: self.recording.key_callback(
                    keycode, self.sim.data.time
                ),
            )
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

        try:
            while dpg.is_dearpygui_running():
                if self.viewer is not None and not self.viewer.is_running():
                    break
                if self.running and not self.sim.fell:
                    self._advance()
                    self.recording.maybe_capture(self.sim.data)
                    self._frame += 1
                    if (
                        self.viewer is not None
                        and self._frame % self.robot_sync_every == 0
                    ):
                        viz.begin_frame(self.viewer)
                        viz.render_gait_overlays(self.viewer, self.sim.controller)
                        self.viewer.sync()
                self._refresh_plots()
                dpg.render_dearpygui_frame()
        finally:
            self.recording.close()
            if self.viewer is not None:
                self.viewer.close()
            dpg.destroy_context()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--no-robot", action="store_true",
                        help="Skip the MuJoCo window and show only the controls")
    parser.add_argument("--walk", type=float, default=None, metavar="SPEED",
                        help="Start in walk mode at this speed [m/s]")
    parser.add_argument("--heading", type=float, default=0.0,
                        help="World-frame travel direction [rad]")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = LocomotionConfig.from_yaml(args.config) if args.config else LocomotionConfig()
    sim = Simulation(config, model_path=args.model)
    if args.walk is not None:
        sim.controller.set_task_targets(
            {
                "gait.mode": "walk",
                "gait.speed": args.walk,
                "gait.heading": args.heading,
            }
        )
    Dashboard(sim, show_robot=not args.no_robot).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
