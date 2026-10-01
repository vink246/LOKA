"""Tests for the MuJoCo viewer overlays.

Overlays are pure side effects on a scene buffer: nothing downstream reads
them, so a broken one produces no error, just an empty picture. These drive
the real ``MjvScene`` -- the geometry calls are where the mistakes live -- and
check the two things that actually go wrong: drawing nothing, and running off
the end of a fixed-size buffer.
"""

from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from loka import viz
from loka.sim import Simulation

#: Deliberately smaller than the overlays would like, to prove they stop.
SMALL_SCENE = 12


@pytest.fixture(scope="module")
def walking():
    """A simulation stepped until the swing foot is airborne."""
    sim = Simulation()
    sim.controller.set_task_targets({"gait.mode": "walk", "gait.speed": 0.25})
    for _ in range(4000):
        sim.step()
        gait = sim.controller._last_gait
        if gait is not None and gait.swing.active and not sim.fell:
            return sim
    pytest.fail("the gait never entered a swing phase")


def _viewer(model, maxgeom: int = 1000):
    return SimpleNamespace(user_scn=mujoco.MjvScene(model, maxgeom=maxgeom))


def test_begin_frame_drops_the_previous_frames_geometry(walking):
    viewer = _viewer(walking.model)
    viz.render_gait_overlays(viewer, walking.controller)
    assert viewer.user_scn.ngeom > 0

    viz.begin_frame(viewer)

    assert viewer.user_scn.ngeom == 0


def test_a_walking_robot_draws_its_plan(walking):
    viewer = _viewer(walking.model)
    viz.begin_frame(viewer)

    viz.render_gait_overlays(viewer, walking.controller)

    # CoM and DCM references, the foothold, the commanded pose, and a sampled
    # arc -- an empty scene here means the operator is flying blind.
    assert viewer.user_scn.ngeom > 10


def test_a_standing_robot_draws_nothing():
    sim = Simulation()
    sim.step()
    viewer = _viewer(sim.model)
    viz.begin_frame(viewer)

    viz.render_gait_overlays(viewer, sim.controller)

    assert viewer.user_scn.ngeom == 0


def test_overlays_stop_at_the_end_of_the_scene_buffer(walking):
    """MuJoCo does not bounds-check ``scene.geoms``; overwriting it segfaults."""
    viewer = _viewer(walking.model, maxgeom=SMALL_SCENE)
    viz.begin_frame(viewer)

    viz.render_gait_overlays(viewer, walking.controller)

    assert viewer.user_scn.ngeom <= SMALL_SCENE


def test_fault_overlays_draw_a_push_and_an_added_mass(walking):
    viewer = _viewer(walking.model)
    viz.begin_frame(viewer)
    torso = mujoco.mj_name2id(
        walking.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
    )
    state = SimpleNamespace(
        external_force=np.array([40.0, 0.0, 0.0]),
        external_body_id=-1,  # falls back to the pelvis
        mass_loads=[(torso, 5.0)],
    )

    viz.render_fault_overlays(viewer, walking.model, walking.data, state)

    # A push arrow, plus a sphere and a weight arrow for the added mass.
    assert viewer.user_scn.ngeom == 3


def test_a_force_too_small_to_matter_draws_no_arrow(walking):
    viewer = _viewer(walking.model)
    viz.begin_frame(viewer)
    state = SimpleNamespace(
        external_force=np.zeros(3), external_body_id=-1, mass_loads=[]
    )

    viz.render_fault_overlays(viewer, walking.model, walking.data, state)

    assert viewer.user_scn.ngeom == 0
