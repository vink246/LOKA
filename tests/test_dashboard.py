"""Tests for the tuning dashboard.

The interesting failure mode here is not an exception in our own code, it is
DearPyGui's argument-passing convention. DearPyGui reads ``co_argcount`` --
which counts parameters that merely *have* defaults -- and passes that many
positional arguments, filling the last with ``user_data``. A callback written
as ``lambda s, v, path=entry.path: ...`` therefore looks like a three-argument
callback and gets its captured default overwritten with ``None``.

That silently broke every slider (``KeyError: None``) and every push button
(which queued a NaN direction and then crashed the *next* frame, inside
``sim.step()``, far from the button that caused it). So these tests invoke
callbacks through DearPyGui's real dispatch rather than calling them directly,
because calling them directly is exactly what hid the bug.
"""

from __future__ import annotations

import inspect
import os

import numpy as np
import pytest

from loka.control import tuning
from loka.dashboard import PLOTS, Dashboard
from loka.sim import Simulation

dpg = pytest.importorskip("dearpygui.dearpygui")

needs_display = pytest.mark.skipif(
    not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"),
    reason="dashboard tests need a display",
)

#: sender, app_data, user_data -- names are the callback's business, but the
#: count is DearPyGui's.
CALLBACK_ARITY = 3


def fire(tag):
    """Invoke a widget's callback exactly as DearPyGui would."""
    config = dpg.get_item_configuration(tag)
    callback = config["callback"]
    count = callback.__code__.co_argcount - inspect.ismethod(callback)
    return callback(*(tag, dpg.get_value(tag), config.get("user_data"))[:count])


def item_labelled(label: str):
    for item in dpg.get_all_items():
        if dpg.get_item_configuration(item).get("label") == label:
            return item
    raise AssertionError(f"no widget labelled {label!r}")


# -- display-free contract ------------------------------------------------


def test_widget_callbacks_take_the_full_dearpygui_triple():
    """Guards the convention without needing a display.

    Any callback taking fewer than three parameters is one that could be
    tempted into capturing state through a default argument, which DearPyGui
    would then overwrite.
    """
    handlers = [
        (name, member)
        for name, member in inspect.getmembers(Dashboard, inspect.isfunction)
        if name.startswith("_on_")
    ]
    assert handlers, "no widget callbacks found"
    for name, handler in handlers:
        parameters = list(inspect.signature(handler).parameters.values())[1:]  # drop self
        assert len(parameters) == CALLBACK_ARITY, (
            f"{name} takes {len(parameters)} arguments; DearPyGui passes "
            f"{CALLBACK_ARITY} (sender, app_data, user_data)"
        )
        defaulted = [
            p.name for p in parameters if p.default is not inspect.Parameter.empty
        ]
        assert not defaulted, (
            f"{name} gives {defaulted} a default; DearPyGui counts defaulted "
            "parameters in co_argcount and overwrites them with user_data"
        )


# -- live widgets ----------------------------------------------------------


@pytest.fixture(scope="module")
def dashboard():
    sim = Simulation()
    dash = Dashboard(sim, show_robot=False)
    dpg.create_context()
    dpg.create_viewport(width=1280, height=860)
    dash._build_ui()
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dash._advance()
    try:
        yield dash
    finally:
        dpg.destroy_context()


@needs_display
def test_no_widget_callback_raises(dashboard):
    for item in dpg.get_all_items():
        if dpg.get_item_configuration(item).get("callback") is None:
            continue
        fire(item)


@needs_display
@pytest.mark.parametrize("entry", tuning.TUNABLES, ids=lambda e: e.path)
def test_slider_drives_its_own_parameter_and_no_other(dashboard, entry):
    controller = dashboard.sim.controller
    before = controller.tunables()
    target = float(np.clip(before[entry.path] * 2.0 + 0.5, entry.low, entry.high))

    dpg.set_value(entry.path, np.log10(target) if entry.log else target)
    fire(entry.path)

    after = controller.tunables()
    assert after[entry.path] == pytest.approx(target, rel=1e-6)
    strays = {
        path for path in after
        if path != entry.path and after[path] != before[path]
    }
    assert not strays, f"{entry.path} also moved {sorted(strays)}"
    assert dpg.get_value(f"readout::{entry.path}") == f"{after[entry.path]:.4g}"


@needs_display
def test_restore_defaults_resets_parameters_and_slider_positions(dashboard):
    for entry in tuning.TUNABLES:
        dpg.set_value(entry.path, np.log10(entry.high) if entry.log else entry.high)
        fire(entry.path)

    fire(item_labelled("restore defaults"))

    restored = dashboard.sim.controller.tunables()
    for path, value in dashboard._defaults.items():
        assert restored[path] == pytest.approx(value, rel=1e-9)
        expected = np.log10(value) if tuning.BY_PATH[path].log else value
        assert dpg.get_value(path) == pytest.approx(expected, rel=1e-6)


@needs_display
@pytest.mark.parametrize(
    "label,direction",
    [("fwd", (1, 0, 0)), ("back", (-1, 0, 0)),
     ("left", (0, 1, 0)), ("right", (0, -1, 0))],
)
def test_push_button_queues_a_usable_push(dashboard, label, direction):
    """A malformed direction used to survive the click and kill the next frame."""
    sim = dashboard.sim
    sim.pushes.clear()
    dpg.set_value("impulse", 7.0)

    fire(item_labelled(label))

    assert len(sim.pushes) == 1
    push = sim.pushes[0]
    assert push.direction.shape == (3,)
    np.testing.assert_allclose(push.direction, direction)
    assert push.impulse == pytest.approx(7.0)
    sim.step()  # the frame that used to crash
    sim.pushes.clear()


@needs_display
def test_command_sliders_reach_the_controller(dashboard):
    command = dashboard.sim.controller.command
    dpg.set_value("height", 0.62)
    fire("height")
    assert command.height == pytest.approx(0.62)

    dpg.set_value("yaw", 0.25)
    fire("yaw")
    assert command.yaw == pytest.approx(0.25)


@needs_display
def test_pause_button_toggles_and_relabels(dashboard):
    was_running = dashboard.running
    fire("pause")
    assert dashboard.running is not was_running
    assert dpg.get_item_label("pause") == ("pause" if dashboard.running else "resume")
    fire("pause")
    assert dashboard.running is was_running
    assert dpg.get_item_label("pause") == ("pause" if dashboard.running else "resume")


@needs_display
def test_every_declared_trace_gets_a_sample_per_frame(dashboard):
    """A plot wired to a key nothing writes stays empty forever, silently."""
    dashboard._reset()
    before = {key: len(buffer) for key, buffer in dashboard.trace.items()}
    dashboard._advance()

    for plot in PLOTS:
        for trace in plot.traces:
            assert len(dashboard.trace[trace.key]) == before[trace.key] + 1, (
                f"{plot.tag} plots {trace.key}, which _advance never appends to"
            )


# -- gait panel ------------------------------------------------------------
#
# These drive the controller through set_task_targets, the same entry point
# the orchestrator uses, so a clamp or a side effect that would surprise the
# LLM surprises the operator here first.


@pytest.fixture
def standing_again(dashboard):
    """Leave the gait clock where the other tests expect it."""
    yield
    dpg.set_value("gait.mode", "stand")
    fire("gait.mode")
    dashboard._reset()


@needs_display
def test_mode_combo_starts_a_walk_and_reports_the_speed_it_chose(
    dashboard, standing_again
):
    """Asking for walk with no speed set picks a crawl; the panel must show it."""
    controller = dashboard.sim.controller
    assert not controller.gait.wants_walk()

    dpg.set_value("gait.mode", "walk")
    fire("gait.mode")

    assert controller.gait.wants_walk()
    chosen = controller.gait.config.speed
    assert chosen > 0.0
    assert dpg.get_value("gait.speed") == pytest.approx(chosen)
    assert dpg.get_value("readout::gait.speed") == f"{chosen:.4g}"


@needs_display
def test_leaving_walk_stops_the_gait_clock(dashboard, standing_again):
    controller = dashboard.sim.controller
    dpg.set_value("gait.mode", "walk")
    fire("gait.mode")
    assert controller.gait.wants_walk()

    dpg.set_value("gait.mode", "stand")
    fire("gait.mode")

    assert not controller.gait.wants_walk()


@needs_display
@pytest.mark.parametrize(
    "path,field,value",
    [
        ("gait.speed", "speed", 0.30),
        ("gait.swing_height", "swing_height", 0.09),
        ("gait.step_period", "step_period", 0.80),
        ("gait.stance_width", "stance_width", 0.26),
        ("gait.capture_gain", "capture_gain", 1.0),
    ],
)
def test_gait_slider_reaches_the_scheduler(dashboard, standing_again, path, field, value):
    dpg.set_value(path, value)
    fire(path)

    scheduler = dashboard.sim.controller.gait
    assert getattr(scheduler.config, field) == pytest.approx(value)
    # StandConfig.gait and the live scheduler are the same object; if they ever
    # diverge, a --config run and a slider run stop agreeing.
    assert dashboard.sim.controller.config.gait is scheduler.config
    assert dpg.get_value(f"readout::{path}") == f"{value:.4g}"


@needs_display
def test_reset_button_rewinds_and_the_loop_keeps_rendering(dashboard):
    dashboard._advance()
    assert dashboard.sim.data.time > 0.0

    fire(item_labelled("reset"))

    assert dashboard.sim.data.time == 0.0
    assert not dashboard.trace["com_error"]
    assert not dashboard.sim.pushes
    for _ in range(3):
        dashboard._advance()
        dashboard._refresh_plots()
        dpg.render_dearpygui_frame()
