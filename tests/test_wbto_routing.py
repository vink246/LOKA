"""WBTO engages for raised terrain (curb, stairs, gap); flat stays classical."""

from __future__ import annotations

import numpy as np

import loka.control.terrain as terrain_mod
from loka.control.terrain import TerrainBelief, Volume, stairs_volumes, terrain_wants_wbto
from loka.control.wbto.session import crocoddyl_available, should_use_wbto, wbto_stack_requested


def _belief_from_specs(specs: list[dict]) -> TerrainBelief:
    volumes = []
    for spec in specs:
        volumes.extend(stairs_volumes(spec))
    belief = TerrainBelief()
    belief.set_volumes(volumes)
    return belief


def test_curb_wants_wbto():
    belief = _belief_from_specs(
        [
            {
                "kind": "stairs",
                "x": 1.0,
                "y": 0.0,
                "yaw": 0.0,
                "steps": 1,
                "riser": 0.10,
                "tread": 0.30,
                "width": 1.5,
                "landing": 1.5,
            }
        ]
    )
    assert terrain_wants_wbto(belief)


def test_four_step_stairs_want_wbto():
    belief = _belief_from_specs(
        [
            {
                "kind": "stairs",
                "x": 1.0,
                "y": 0.0,
                "yaw": 0.0,
                "steps": 4,
                "riser": 0.07,
                "tread": 0.30,
                "width": 1.2,
                "landing": 1.0,
            }
        ]
    )
    assert terrain_wants_wbto(belief)


def test_platform_gap_wants_wbto():
    belief = _belief_from_specs(
        [
            {
                "kind": "stairs",
                "x": -0.5,
                "y": 0.0,
                "yaw": 0.0,
                "steps": 1,
                "riser": 0.10,
                "tread": 0.30,
                "width": 1.5,
                "landing": 1.5,
            },
            {
                "kind": "stairs",
                "x": 1.10,
                "y": 0.0,
                "yaw": 0.0,
                "steps": 1,
                "riser": 0.10,
                "tread": 0.30,
                "width": 1.5,
                "landing": 2.0,
            },
        ]
    )
    assert terrain_wants_wbto(belief)


def test_almost_touching_pads_still_want_wbto():
    """Vision sometimes declares pads with only a centimetre seam — still raised."""
    belief = TerrainBelief()
    belief.set_volumes(
        [
            Volume(name="near", center=np.array([0.45, 0.0]), size=np.array([0.7, 1.46]), top=0.12),
            Volume(name="far", center=np.array([1.9, 0.0]), size=np.array([2.15, 1.46]), top=0.12),
        ]
    )
    assert terrain_wants_wbto(belief)


def test_flat_belief_skips_wbto():
    assert not terrain_wants_wbto(TerrainBelief())


def test_dust_height_volume_skips_wbto():
    belief = TerrainBelief()
    belief.set_volumes(
        [Volume(name="dust", center=np.array([0.0, 0.0]), size=np.array([1.0, 1.0]), top=0.01)]
    )
    assert not terrain_wants_wbto(belief)


def test_auto_stack_routing(monkeypatch):
    monkeypatch.setattr(terrain_mod, "TERRAIN_STACK", "auto")
    stairs = _belief_from_specs(
        [
            {
                "kind": "stairs",
                "x": 1.0,
                "y": 0.0,
                "yaw": 0.0,
                "steps": 4,
                "riser": 0.07,
                "tread": 0.30,
                "width": 1.2,
                "landing": 1.0,
            }
        ]
    )
    curb = _belief_from_specs(
        [
            {
                "kind": "stairs",
                "x": 1.0,
                "y": 0.0,
                "yaw": 0.0,
                "steps": 1,
                "riser": 0.10,
                "tread": 0.30,
                "width": 1.5,
                "landing": 1.5,
            }
        ]
    )
    assert wbto_stack_requested()
    assert should_use_wbto(stairs) == crocoddyl_available()
    assert should_use_wbto(curb) == crocoddyl_available()


def test_classical_stack_never_routes_wbto(monkeypatch):
    monkeypatch.setattr(terrain_mod, "TERRAIN_STACK", "classical")
    stairs = _belief_from_specs(
        [
            {
                "kind": "stairs",
                "x": 1.0,
                "y": 0.0,
                "yaw": 0.0,
                "steps": 4,
                "riser": 0.07,
                "tread": 0.30,
                "width": 1.2,
                "landing": 1.0,
            }
        ]
    )
    assert not should_use_wbto(stairs)
