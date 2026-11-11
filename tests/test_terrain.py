"""Terrain belief: geometry, snapping, orchestrator mutations, refinement."""

from __future__ import annotations

import numpy as np
import pytest

from loka.agent.compress import format_terrain_section
from loka.agent.terrain_refine import refine_from_touchdown
from loka.agent.terrain_state import (
    apply_terrain_mutation,
    ensure_terrain_state,
    format_terrain_belief,
    seed_oracle_terrain,
    volumes_of,
)
from loka.control.gait import STAIR_CLEARANCE, stair_swing_reference
from loka.control.terrain import (
    EDGE_MARGIN,
    MAX_VOLUMES,
    SOLE_HALF_LENGTH,
    TerrainBelief,
    Volume,
    stairs_volumes,
)

FLIGHT = {"name": "s", "x": 1.0, "y": 0.0, "yaw": 0.0, "steps": 3, "riser": 0.1, "tread": 0.3, "width": 1.0, "landing": 1.0}


def belief_of(spec=FLIGHT) -> TerrainBelief:
    belief = TerrainBelief()
    belief.set_volumes(stairs_volumes(spec))
    return belief


def test_stairs_heights_are_cumulative_and_landing_is_deep():
    belief = belief_of()
    assert belief.height_at([0.9, 0.0]) == 0.0
    assert belief.height_at([1.15, 0.0]) == pytest.approx(0.1)
    assert belief.height_at([1.45, 0.0]) == pytest.approx(0.2)
    assert belief.height_at([2.4, 0.0]) == pytest.approx(0.3)
    assert belief.height_at([2.65, 0.0]) == 0.0
    assert belief.height_at([1.15, 0.6]) == 0.0


def test_irregular_risers():
    belief = belief_of({**FLIGHT, "riser": [0.1, 0.14, 0.1]})
    assert belief.height_at([1.45, 0.0]) == pytest.approx(0.24)
    assert belief.height_at([2.0, 0.0]) == pytest.approx(0.34)


def test_footprint_on_the_edge_is_not_uniform():
    belief = belief_of()
    assert belief.footprint_uniform([1.15, 0.0], 0.0)
    assert not belief.footprint_uniform([1.0, 0.0], 0.0)
    assert belief.support_height([1.0, 0.0], 0.0) == pytest.approx(0.1)


def test_snap_backs_off_while_it_makes_progress_and_steps_up_otherwise():
    belief = belief_of()
    fwd = np.array([1.0, 0.0])
    # Toe would cross the riser; the other foot is well behind: back off.
    back = belief.snap_foothold([0.95, 0.1], 0.0, fwd, anchor=[0.6, -0.1], anchor_z=0.0)
    assert back[0] <= 1.0 - SOLE_HALF_LENGTH - EDGE_MARGIN + 1e-9
    assert belief.footprint_uniform(back, 0.0)
    # The other foot already stands at the back-off spot: step onto the tread.
    up = belief.snap_foothold([0.95, 0.1], 0.0, fwd, anchor=[0.88, -0.1], anchor_z=0.0)
    assert up[0] >= 1.0 + SOLE_HALF_LENGTH + EDGE_MARGIN - 1e-9
    assert belief.support_height(up, 0.0) == pytest.approx(0.1)


def test_snap_refuses_a_riser_above_the_step_limit():
    belief = belief_of({**FLIGHT, "riser": 0.25})
    fwd = np.array([1.0, 0.0])
    pos = belief.snap_foothold([0.95, 0.1], 0.0, fwd, anchor=[0.88, -0.1], anchor_z=0.0)
    assert belief.support_height(pos, 0.0) == 0.0


def test_stair_swing_lifts_above_the_higher_tread_before_moving():
    start = np.array([0.88, 0.1, 0.0])
    end = np.array([1.15, 0.1, 0.1])
    pos0, _, _ = stair_swing_reference(start, end, 0.0, clearance=STAIR_CLEARANCE)
    pos1, vel1, _ = stair_swing_reference(start, end, 1.0, clearance=STAIR_CLEARANCE)
    np.testing.assert_allclose(pos0, start, atol=1e-9)
    np.testing.assert_allclose(pos1, end, atol=1e-9)
    np.testing.assert_allclose(vel1, 0.0, atol=1e-9)
    samples = [stair_swing_reference(start, end, s, clearance=STAIR_CLEARANCE)[0] for s in np.linspace(0, 1, 201)]
    toe = np.array([p[0] + SOLE_HALF_LENGTH for p in samples])
    z = np.array([p[2] for p in samples])
    # Wherever the toe is past the riser, the sole is at or above the tread.
    assert np.all(z[toe > 1.0] >= 0.1 - 1e-9)
    assert z.max() == pytest.approx(0.1 + STAIR_CLEARANCE)


def test_stairs_mutation_expands_clamps_and_removes():
    state: dict = {}
    line = apply_terrain_mutation(state, {
        "object_type": "stairs", "name": "flight",
        "value": {"start": [1.0, 0.0], "heading": 0.0, "steps": 12, "riser": 0.5, "tread": 0.3},
    }, 1.0)
    assert "flight" in line
    vols = volumes_of(state)
    assert len(vols) == 8  # step count clamped
    assert vols[0].top == pytest.approx(0.2)  # riser clamped to 0.20
    version = state["terrain_version"]
    apply_terrain_mutation(state, {"object_type": "stairs", "name": "flight", "attribute": "remove"})
    assert volumes_of(state) == [] and state["terrain_version"] > version


def test_volume_mutation_merges_and_rejects_unknown_attributes():
    state: dict = {}
    apply_terrain_mutation(state, {
        "object_type": "volume", "name": "curb", "attribute": "box",
        "value": {"center": [1.5, 0.0], "size": [1.0, 1.0], "top": 0.12},
    })
    apply_terrain_mutation(state, {"object_type": "volume", "name": "curb", "attribute": "top", "value": 0.1})
    (vol,) = volumes_of(state)
    assert vol.top == pytest.approx(0.1) and vol.center[0] == pytest.approx(1.5)
    with pytest.raises(ValueError):
        apply_terrain_mutation(state, {"object_type": "volume", "name": "curb", "attribute": "mass", "value": 1})
    with pytest.raises(ValueError):
        apply_terrain_mutation(state, {"object_type": "volume", "name": "nope", "attribute": "remove"})


def test_volume_cap():
    state: dict = {}
    for i in range(MAX_VOLUMES):
        apply_terrain_mutation(state, {"object_type": "volume", "name": f"v{i}", "value": {"center": [i, 0], "top": 0.1}})
    with pytest.raises(ValueError):
        apply_terrain_mutation(state, {"object_type": "volume", "name": "extra", "value": {"center": [0, 3], "top": 0.1}})


def test_refine_moves_the_flight_above_a_mismeasured_step():
    state: dict = {}
    apply_terrain_mutation(state, {
        "object_type": "stairs", "name": "s",
        "value": {"start": [1.0, 0.0], "steps": 3, "riser": 0.10, "tread": 0.3},
    })
    result = refine_from_touchdown(state, {"xy": [1.45, 0.0], "measured_z": 0.24})
    assert result.kind == "refined"
    tops = [v.top for v in volumes_of(state)]
    assert tops == pytest.approx([0.10, 0.24, 0.34])
    assert refine_from_touchdown(state, {"xy": [1.15, 0.0], "measured_z": 0.104}).kind == "consistent"


def test_refine_reports_ground_the_belief_does_not_have():
    state: dict = {}
    ensure_terrain_state(state)
    result = refine_from_touchdown(state, {"xy": [1.2, 0.0], "measured_z": 0.1})
    assert result.kind == "unexplained"
    assert "no volume" in result.note


def test_oracle_seed_and_belief_text():
    state: dict = {}
    assert seed_oracle_terrain(state, [FLIGHT]) == 3
    text = format_terrain_belief(state)
    assert "s_0" in text and "oracle" in text


def test_terrain_section_lists_events_and_the_fault():
    frame = {"terrain": {
        "active": False, "volumes": 0,
        "feet": [[0.9, 0.1, 0.0], [0.9, -0.1, 0.0]],
        "leg_z": [0.0, 0.0], "believed_z": [0.0, 0.0],
        "load_n": [[0, 0, 170], [0, 0, 170]], "footholds": [],
        "events": [{"kind": "swing_obstructed", "sim_t": 6.0, "leg": 0, "s": 0.4,
                    "xy": [0.95, 0.1], "foot_z": 0.03, "target_z": 0.0, "load_n": [-80, 0, 20]}],
        "fault": {"note": "swing foot blocked"},
    }}
    text = format_terrain_section([frame])
    assert "TERRAIN" in text and "swing foot blocked" in text and "blocked at 40%" in text
    assert format_terrain_section([{"terrain": {**frame["terrain"], "events": [], "fault": None}}]) == ""


def _runtime(terrain=None, belief="empty"):
    from loka.agent.runtime import LokaConfig, LokaRuntime
    from loka.sim import Simulation

    sim = Simulation(terrain=terrain)
    return LokaRuntime(sim, LokaConfig(enable_llm=False, terrain_belief=belief))


def test_stairs_mutation_reaches_the_controller_belief():
    from loka.agent.apply import apply_stand_scratchpad

    runtime = _runtime()
    assert not runtime.controller.terrain.active
    summary = apply_stand_scratchpad(
        runtime.controller,
        {
            "Semantic_State": {"Hypothesis": "stairs ahead"},
            "Model_Mutations": [{
                "object_type": "stairs", "name": "flight",
                "value": {"start": [1.0, 0.0], "heading": 0.0, "steps": 4, "riser": 0.1, "tread": 0.3},
            }],
        },
        runtime.loka_state,
        sim_time=0.0,
    )
    assert summary["mutations"] == 1 and summary["terrain"]
    assert runtime.controller.terrain.active
    assert runtime.controller.terrain.height_at([1.45, 0.0]) == pytest.approx(0.2)
    assert "flight_0" in format_terrain_belief(runtime.loka_state)


def test_oracle_belief_matches_the_plant():
    runtime = _runtime(terrain=[FLIGHT], belief="oracle")
    plant = runtime.sim.plant_terrain
    for x in (0.5, 1.15, 1.45, 2.2):
        assert runtime.controller.terrain.height_at([x, 0.0]) == pytest.approx(plant.height_at([x, 0.0]))


def test_runtime_refines_touchdowns_and_faults_on_unexplained_ground():
    runtime = _runtime(terrain=[FLIGHT], belief="oracle")
    ctrl = runtime.controller
    ctrl.terrain_events.append({"kind": "touchdown", "t": 1.0, "leg": 0, "xy": [1.15, 0.1],
                                "yaw": 0.0, "believed_z": 0.1, "measured_z": 0.13, "step": 3})
    runtime._process_terrain_events(1.0)
    runtime.step()
    assert ctrl.terrain.height_at([1.15, 0.0]) == pytest.approx(0.13)
    assert ctrl.terrain.height_at([1.45, 0.0]) == pytest.approx(0.23)
    assert runtime._terrain_fault is None

    blank = _runtime(terrain=[FLIGHT], belief="empty")
    blank.controller.terrain_events.append({
        "kind": "swing_obstructed", "t": 2.0, "leg": 1, "s": 0.4, "xy": [0.95, -0.1],
        "foot_z": 0.03, "target_z": 0.0, "load_n": [-90.0, 0.0, 15.0],
    })
    blank._process_terrain_events(2.0)
    assert blank._terrain_fault is not None
    assert "blocked" in blank._terrain_fault["note"]


def test_foot_load_estimate_carries_body_weight_while_standing():
    from loka.sim import Simulation

    sim = Simulation()
    for _ in range(100):
        sim.step()
    load = sim.controller.foot_load
    weight = sim.controller.robot.total_mass * 9.81
    assert load[:, 2].sum() == pytest.approx(weight, rel=0.05)
    assert abs(load[0, 2] - load[1, 2]) < 0.1 * weight


def test_wrench_distribution_puts_cop_at_the_cmp():
    from loka.control.stabilizer import DCMStabilizer

    stab = DCMStabilizer(8, 35.0, mu=0.5, fz_min=1.0, fz_max=500.0)
    soles = []
    for y in (0.1, -0.1):
        for x, dy in ((-0.05, 0.025), (-0.05, -0.025), (0.12, 0.03), (0.12, -0.03)):
            soles.append([x, y + dy, 0.0])
    soles = np.array(soles)
    com = np.array([0.03, 0.02, 0.7])
    # A CoM at rest over the polygon, upright: the CMP is under the CoM.
    force, moment = stab.desired_wrench(
        com=com, com_vel=np.zeros(3), com_z_ref=0.7, com_vz_ref=0.0,
        dcm_ref=com[:2], dcm_rate_ref=np.zeros(2), omega=3.7,
        orientation_error=np.zeros(3), angular_momentum=np.zeros(3), inertia=np.eye(3),
    )
    forces = stab.distribute(force, moment, com, soles, np.ones(8, dtype=bool))
    total = forces.sum(axis=0)
    assert total == pytest.approx(force, abs=2.0)
    cop = (soles[:, :2] * forces[:, 2:3]).sum(axis=0) / forces[:, 2].sum()
    assert cop == pytest.approx(com[:2], abs=0.005)
    # A lifted foot carries nothing.
    lifted = stab.distribute(force, moment, com, soles, np.array([True] * 4 + [False] * 4))
    assert np.allclose(lifted[4:], 0.0)


def test_volume_round_trip():
    vol = Volume("a", [1, 2], [0.3, 1.0], 0.1, yaw=0.5, group="g", index=2)
    again = Volume.from_dict(vol.to_dict())
    assert again.group == "g" and again.index == 2 and again.yaw == pytest.approx(0.5)
