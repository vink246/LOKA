#!/usr/bin/env python3
"""Live tuning dashboard for the standing controller.

    python -m loka.dashboard              # sliders + MuJoCo viewer
    python -m loka.dashboard --no-robot   # sliders only, if the GPU is busy

Every slider is one entry from :mod:`loka.control.tuning`, which is the same
typed surface the orchestrator LLM will mutate. Anything tunable by hand here
is tunable by the model later, with identical names, ranges and clamping --
the dashboard is a view onto that registry rather than a parallel list of
knobs that can drift away from it.

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
from pathlib import Path

import dearpygui.dearpygui as dpg
import numpy as np

from loka.control import tuning
from loka.control.stand import DEFAULT_MODEL, StandConfig
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

PLOTS = (
    ("com_error", "CoM error [mm]"),
    ("tilt", "torso tilt [deg]"),
    ("torque", "peak |torque| [Nm]"),
    ("solve_ms", "controller [ms]"),
)


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
            key: deque(maxlen=depth) for key, _ in (("t", ""),) + PLOTS
        }
        self._defaults = sim.controller.tunables()
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

    def _push(self, direction, impulse: float = 6.0) -> None:
        direction = np.asarray(direction, dtype=float).reshape(3)
        self.sim.pushes.append(
            Push(impulse=impulse, direction=direction, time=self.sim.data.time)
        )

    def _reset(self) -> None:
        self.sim.pushes.clear()
        self.sim.reset()
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

    def _on_push(self, sender, app_data, direction) -> None:
        self._push(direction, dpg.get_value("impulse"))

    def _on_height(self, sender, value: float, user_data) -> None:
        self.sim.controller.command.height = value

    def _on_yaw(self, sender, value: float, user_data) -> None:
        self.sim.controller.command.yaw = value

    def _on_reset(self, sender, app_data, user_data) -> None:
        self._reset()

    def _on_restore_defaults(self, sender, app_data, user_data) -> None:
        self.sim.controller.update_weights(**self._defaults)
        for path, value in self._defaults.items():
            entry = tuning.BY_PATH[path]
            dpg.set_value(path, np.log10(value) if entry.log else value)
            dpg.set_value(f"readout::{path}", f"{value:.4g}")

    def _on_toggle_running(self, sender, app_data, user_data) -> None:
        self.running = not self.running
        dpg.set_item_label("pause", "pause" if self.running else "resume")

    # -- layout -------------------------------------------------------------

    def _build_sliders(self, group: str) -> None:
        for entry in tuning.TUNABLES:
            if entry.group != group:
                continue
            value = self._defaults[entry.path]
            with dpg.group(horizontal=True):
                dpg.add_slider_float(
                    tag=entry.path,
                    # The knob name, not the attribute: weight_orientation_rp
                    # and weight_orientation_yaw both resolve to the same
                    # field and would otherwise be two identical labels.
                    label=entry.path.split(".", 1)[1],
                    width=240,
                    default_value=np.log10(value) if entry.log else value,
                    min_value=np.log10(entry.low) if entry.log else entry.low,
                    max_value=np.log10(entry.high) if entry.log else entry.high,
                    # A log slider carries the exponent, so the number beside
                    # it is the only readable form of the real value.
                    format="" if entry.log else "%.3g",
                    user_data=entry.path,
                    callback=self._on_tunable,
                )
                dpg.add_text(f"{value:.4g}", tag=f"readout::{entry.path}")
            with dpg.tooltip(entry.path):
                dpg.add_text(entry.summary, wrap=320)
                dpg.add_text(
                    f"range [{entry.low:g}, {entry.high:g}]"
                    + ("  (log scale)" if entry.log else ""),
                    color=(150, 150, 150),
                )

    def _build_ui(self) -> None:
        with dpg.window(tag="root"):
            with dpg.group(horizontal=True):
                # -- left: knobs ------------------------------------------
                with dpg.child_window(width=430, autosize_y=True):
                    dpg.add_text("controller", color=(120, 200, 255))
                    with dpg.group(horizontal=True):
                        dpg.add_button(
                            label="pause", tag="pause",
                            callback=self._on_toggle_running,
                        )
                        dpg.add_button(label="reset", callback=self._on_reset)
                        dpg.add_button(
                            label="restore defaults",
                            callback=self._on_restore_defaults,
                        )
                    dpg.add_text("push the pelvis")
                    with dpg.group(horizontal=True):
                        for label, vector in (
                            ("fwd", (1, 0, 0)), ("back", (-1, 0, 0)),
                            ("left", (0, 1, 0)), ("right", (0, -1, 0)),
                        ):
                            dpg.add_button(
                                label=label,
                                user_data=vector,
                                callback=self._on_push,
                            )
                        dpg.add_slider_float(
                            tag="impulse", label="N.s", width=90,
                            default_value=6.0, min_value=1.0, max_value=25.0,
                            format="%.1f",
                        )

                    dpg.add_separator()
                    dpg.add_slider_float(
                        tag="height", label="CoM height [m]", width=240,
                        default_value=self.sim.controller.nominal_height,
                        min_value=0.50,
                        max_value=self.sim.controller.nominal_height,
                        format="%.3f",
                        callback=self._on_height,
                    )
                    dpg.add_slider_float(
                        tag="yaw", label="yaw [rad]", width=240,
                        default_value=0.0, min_value=-0.6, max_value=0.6,
                        format="%.2f",
                        callback=self._on_yaw,
                    )

                    for group, title in (
                        ("wbc", "whole-body QP"),
                        ("mpc", "centroidal MPC"),
                        ("stand", "saturation"),
                    ):
                        with dpg.collapsing_header(
                            label=title, default_open=(group == "wbc")
                        ):
                            self._build_sliders(group)

                # -- right: telemetry --------------------------------------
                with dpg.child_window(autosize_x=True, autosize_y=True):
                    dpg.add_text("", tag="status")
                    dpg.add_text("", tag="status2", color=(150, 150, 150))
                    for key, label in PLOTS:
                        with dpg.plot(label=label, height=140, width=-1):
                            dpg.add_plot_axis(dpg.mvXAxis, tag=f"x::{key}")
                            axis = dpg.add_plot_axis(dpg.mvYAxis, tag=f"y::{key}")
                            dpg.add_line_series(
                                [], [], parent=axis, tag=f"series::{key}"
                            )
        dpg.set_primary_window("root", True)

    # -- frame --------------------------------------------------------------

    def _advance(self) -> None:
        for _ in range(self.steps_per_frame):
            self.sim.step()
        row = self.sim.history[-1]
        self.trace["t"].append(row["t"])
        self.trace["com_error"].append(row["com_error"] * 1e3)
        self.trace["tilt"].append(np.degrees(row["tilt"]))
        self.trace["torque"].append(row["torque"])
        self.trace["solve_ms"].append(row["solve_ms"])

    def _refresh_plots(self) -> None:
        times = list(self.trace["t"])
        if not times:
            return
        for key, _ in PLOTS:
            dpg.set_value(f"series::{key}", [times, list(self.trace[key])])
            dpg.fit_axis_data(f"x::{key}")
            dpg.fit_axis_data(f"y::{key}")

        telemetry = self.sim.controller.telemetry
        now = time.perf_counter()
        elapsed = now - self._last_wall
        self._last_wall = now
        if elapsed > 0:
            instant = self.steps_per_frame * self.sim.config.control_dt / elapsed
            self._realtime += 0.1 * (instant - self._realtime)
        dpg.set_value(
            "status",
            f"t {self.sim.data.time:7.2f} s   {self._realtime:4.2f}x realtime"
            f"   {'FALLEN' if self.sim.fell else 'standing'}"
            f"   contacts {int(telemetry.contact_mask.sum())}/8",
        )
        margin = self.sim.controller.robot.support_margin()
        dpg.set_value(
            "status2",
            f"CoM margin  fwd {margin[1] * 1e3:5.1f} mm   back {margin[0] * 1e3:5.1f} mm"
            f"   |   QP fallbacks  wbc {self.sim.controller.wbc._qp.failures}"
            f"  mpc {self.sim.controller.mpc._qp.failures}",
        )

    def run(self) -> None:
        dpg.create_context()
        dpg.create_viewport(title="LOKA - G1 stand", width=1280, height=860)
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = StandConfig.from_yaml(args.config) if args.config else StandConfig()
    sim = Simulation(config, model_path=args.model)
    Dashboard(sim, show_robot=not args.no_robot).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
