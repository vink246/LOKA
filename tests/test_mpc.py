"""ConvexMPC: finite-foot CoP constraints on the Di Carlo SRBD QP."""

from __future__ import annotations

import numpy as np
import pytest

from loka.control.locomotion import LocomotionController
from loka.control.mpc import (
    SOLE_HALF_WIDTH,
    CentroidalNMPC,
    CentroidalReference,
    CentroidalState,
    ConvexMPC,
    MPCConfig,
)
from loka.control.robot import GRAVITY


@pytest.fixture(scope="module")
def controller() -> LocomotionController:
    return LocomotionController()


@pytest.fixture(scope="module")
def nominal(controller: LocomotionController):
    return controller.robot.nominal_qpos.copy(), np.zeros(controller.robot.nv)


def test_centroidal_nmpc_is_a_compat_alias():
    assert CentroidalNMPC is ConvexMPC


def test_standing_forces_keep_cop_inside_the_sole(controller, nominal):
    qpos, qvel = nominal
    controller.compute_torque(qpos, qvel)
    forces = np.asarray(controller.mpc.last_forces)
    robot = controller.robot
    robot.update(qpos, qvel)
    pos = robot.dynamics(qvel).contact_pos
    yaw = 0.0
    c, s = np.cos(yaw), np.sin(yaw)
    rotate = np.array([[c, s], [-s, c]])
    half_l = SOLE_HALF_WIDTH - float(controller.config.mpc.cop_margin)
    for foot in range(2):
        sl = slice(foot * 4, (foot + 1) * 4)
        fz = forces[sl, 2]
        total = float(fz.sum())
        assert total > 1.0
        pts = pos[sl, :2]
        center = pts.mean(axis=0)
        local = (pts - center) @ rotate.T
        cop = (fz[:, None] * local).sum(axis=0) / total
        assert abs(cop[1]) <= half_l + 1e-3, (
            f"foot {foot} CoP left {cop[1]*1e3:.1f} mm vs sole box "
            f"±{half_l*1e3:.1f} mm"
        )


def test_convex_mpc_solve_returns_finite_forces_on_a_dummy_horizon():
    mass = 35.0
    mpc = ConvexMPC(MPCConfig(), num_contacts=8, mass=mass)
    contacts = np.array(
        [
            [0.0, 0.09, 0.0],
            [0.0, 0.14, 0.0],
            [0.17, 0.09, 0.0],
            [0.17, 0.14, 0.0],
            [0.0, -0.14, 0.0],
            [0.0, -0.09, 0.0],
            [0.17, -0.14, 0.0],
            [0.17, -0.09, 0.0],
        ]
    )
    state = CentroidalState(
        rpy=np.zeros(3),
        com=np.array([0.08, 0.0, 0.70]),
        angular_velocity=np.zeros(3),
        com_velocity=np.zeros(3),
        inertia=np.diag([2.0, 2.0, 0.4]),
        contact_pos=contacts,
    )
    reference = CentroidalReference(com=state.com.copy())
    forces = mpc.solve(state, reference, contact_mask=np.ones(8, dtype=bool))
    assert forces.shape == (8, 3)
    assert np.all(np.isfinite(forces))
    assert forces[:, 2].sum() == pytest.approx(mass * GRAVITY, rel=0.15)


def test_yaw_reference_wraps_across_pi():
    """A goal just past ±π must be a small correction, not a full turn."""
    cfg = MPCConfig()
    mpc = ConvexMPC(cfg, num_contacts=8, mass=35.0)
    contacts = np.zeros((8, 3))
    contacts[:, 2] = 0.0
    contacts[:4, 1] = 0.1
    contacts[4:, 1] = -0.1
    state = CentroidalState(
        rpy=np.array([0.0, 0.0, -3.1]),
        com=np.array([0.0, 0.0, 0.70]),
        angular_velocity=np.zeros(3),
        com_velocity=np.zeros(3),
        inertia=np.diag([2.0, 2.0, 0.4]),
        contact_pos=contacts,
    )
    near = CentroidalReference(com=state.com.copy(), yaw=3.1)
    far = CentroidalReference(com=state.com.copy(), yaw=0.0)
    f_near = mpc.solve(state, near, contact_mask=np.ones(8, dtype=bool))
    f_far = mpc.solve(state, far, contact_mask=np.ones(8, dtype=bool))
    assert np.all(np.isfinite(f_near))
    # The short wrap (~0.08 rad) must cost less yaw moment than a 3 rad error.
    assert abs(f_near[:, 0]).sum() < abs(f_far[:, 0]).sum()
