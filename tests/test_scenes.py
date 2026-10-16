"""Saved prop layouts: run_loka --scene and the scene save/load commands."""

from __future__ import annotations

import pytest

from loka.agent.interactive import parse_prop_command
from loka.scene_props import list_scenes, load_scene, save_scene
from loka.sim import Simulation


def test_shipped_scene_loads():
    assert "obstacle_course" in list_scenes()
    spawns = load_scene("obstacle_course")
    assert [kind for kind, *_ in spawns] == ["box", "table", "chair", "shelf"]
    assert spawns[2] == ("chair", 3.5, -0.6, 1.57)


def test_unknown_scene_names_the_saved_ones():
    with pytest.raises(ValueError, match="obstacle_course"):
        load_scene("no_such_scene")


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "mine.yaml"
    save_scene(path, [("box", 1.0, -0.5, 0.3), ("cube", 2.0, 0.0, 0.0)])
    assert load_scene(path) == [("box", 1.0, -0.5, 0.3), ("cube", 2.0, 0.0, 0.0)]


def test_scene_commands_parse():
    assert parse_prop_command("scene save Course_2") == ("save_scene", "Course_2")
    assert parse_prop_command("scene load obstacle_course") == ("load_scene", "obstacle_course")
    assert parse_prop_command("scenes") == ("scenes",)
    assert parse_prop_command("scene of the crime") is None


def test_manager_saves_and_reloads_live_poses(tmp_path):
    sim = Simulation(props={"box": 1, "chair": 1, "shelf": 1}, static_props=("shelf",))
    sim.props.spawn("box", 1.2, -0.4)
    sim.props.spawn("shelf", 4.5, 1.0, 3.14)
    path = tmp_path / "live.yaml"
    assert "saved 2 prop(s)" in sim.props.save_scene(str(path))
    sim.props.clear()
    lines = sim.props.load_scene(str(path))
    assert sum("spawned" in line for line in lines) == 2
    placed = {slot.kind for slot in sim.prop_layout.slots if slot.active}
    assert placed == {"box", "shelf"}
