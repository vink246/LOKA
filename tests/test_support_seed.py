"""Raised start pads seed support height without revealing terrain volumes."""

from __future__ import annotations

from loka.control.locomotion import LocomotionConfig
from loka.scene_props import load_scene_terrain
from loka.sim import Simulation


def test_platform_gap_stands_with_empty_belief():
    sim = Simulation(
        LocomotionConfig.from_yaml("loka/config/g1.yaml"),
        terrain=load_scene_terrain("platform_gap"),
    )
    assert not sim.controller.terrain.active
    assert sim.controller._ground_height > 0.05
    for _ in range(250):
        sim.step()
        assert not sim.fell, f"collapsed at t={sim.data.time:.3f} with empty belief"


def test_curb_and_stairs_still_start_on_floor_plane():
    cfg = LocomotionConfig.from_yaml("loka/config/g1.yaml")
    for scene in ("curb", "stairs_regular"):
        sim = Simulation(cfg, terrain=load_scene_terrain(scene))
        assert not sim.controller.terrain.active
        assert abs(sim.controller._ground_height) < 0.02
        for _ in range(100):
            sim.step()
            assert not sim.fell
