"""Tests for the G1 standing controller.

Split into cheap invariants (physics identities the controller must satisfy at
a single state) and a handful of short closed-loop runs. The closed-loop tests
dominate the runtime; mark-deselect them with ``-m 'not slow'``.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from loka.control import tuning
from loka.control.qp import QP, Sparsity
from loka.control.robot import (
    GRAVITY,
    NUM_JOINTS,
    QVEL_JOINT0,
    G1Model,
    orientation_error,
    quat_to_rpy,
)
from loka.control.locomotion import DEFAULT_MODEL, LocomotionConfig, LocomotionController
from loka.sim import Push, Simulation


def settled(sim: Simulation) -> Simulation:
    """Drop the pelvis until the lowest sole point rests on the floor.

    Perturbing joint angles moves the feet, so without this a test would start
    with one foot buried in the ground and the other in mid-air -- a state the
    physics never produces and no controller owes an answer for.
    """
    mujoco.mj_forward(sim.model, sim.data)
    robot = sim.controller.robot
    robot.update(sim.data.qpos, sim.data.qvel)
    sole_z = robot.data.site_xpos[robot.contact_site_ids][:, 2]
    sim.data.qpos[2] -= sole_z.min()
    mujoco.mj_forward(sim.model, sim.data)
    return sim


@pytest.fixture(scope="module")
def controller() -> LocomotionController:
    return LocomotionController()


@pytest.fixture(scope="module")
def nominal(controller: LocomotionController):
    return controller.robot.nominal_qpos.copy(), np.zeros(controller.robot.nv)


# -- model ---------------------------------------------------------------


def test_model_matches_unitree_layout(controller: LocomotionController):
    robot = controller.robot
    assert robot.nu == NUM_JOINTS
    assert robot.nv == QVEL_JOINT0 + NUM_JOINTS
    assert robot.num_contacts == 8
    assert 30.0 < robot.total_mass < 45.0


def test_nominal_stance_has_flat_feet_on_the_floor(controller: LocomotionController):
    robot = controller.robot
    robot.update(robot.nominal_qpos, np.zeros(robot.nv))
    sole_z = robot.data.site_xpos[robot.contact_site_ids][:, 2]
    assert np.ptp(sole_z) < 1e-6, "soles are not coplanar"
    assert abs(sole_z.mean()) < 1e-3, "soles are not resting on z=0"


def test_nominal_com_sits_over_the_support_polygon(controller: LocomotionController):
    """Equal fore/aft margin is what makes push recovery symmetric."""
    robot = controller.robot
    robot.update(robot.nominal_qpos, np.zeros(robot.nv))
    backward, forward = robot.support_margin()[:2]
    assert forward == pytest.approx(backward, abs=2e-3)
    assert forward > 0.05, "no room to shift the centre of pressure"


def test_com_inertia_matches_a_direct_sum(controller: LocomotionController):
    robot = controller.robot
    robot.update(robot.nominal_qpos, np.zeros(robot.nv))
    model, data = robot.model, robot.data
    expected = np.zeros((3, 3))
    for body in range(1, model.nbody):
        rot = data.ximat[body].reshape(3, 3)
        expected += rot @ np.diag(model.body_inertia[body]) @ rot.T
        offset = data.xipos[body] - data.subtree_com[0]
        expected += model.body_mass[body] * (
            np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset)
        )
    np.testing.assert_allclose(robot.com_inertia(), expected, atol=1e-9)


def test_contact_bias_acceleration_matches_finite_differences():
    """``J̇q̇`` from the analytic path must agree with differentiating ``Jq̇``."""
    robot = G1Model(DEFAULT_MODEL)
    rng = np.random.default_rng(0)
    qpos = robot.nominal_qpos.copy()
    qvel = rng.normal(0.0, 0.3, robot.nv)
    robot.update(qpos, qvel)
    reported = robot.dynamics(qvel).contact_bias_acc

    step = 1e-5
    scratch = mujoco.MjData(robot.model)
    jacobians = []
    for sign in (+1, -1):
        scratch.qpos[:] = qpos
        mujoco.mj_integratePos(robot.model, scratch.qpos, qvel, sign * step)
        mujoco.mj_kinematics(robot.model, scratch)
        mujoco.mj_comPos(robot.model, scratch)
        jacobians.append(robot.contact_jacobian(scratch))
    central = (jacobians[0] - jacobians[1]) @ qvel / (2.0 * step)
    np.testing.assert_allclose(reported, central, atol=1e-3)


def test_orientation_error_is_zero_for_identical_orientations():
    quat = np.array([0.9689, 0.0, 0.2474, 0.0])
    quat /= np.linalg.norm(quat)
    np.testing.assert_allclose(orientation_error(quat, quat), np.zeros(3), atol=1e-12)


def test_orientation_error_points_the_short_way_round():
    upright = np.array([1.0, 0.0, 0.0, 0.0])
    pitched = np.array([np.cos(0.05), 0.0, np.sin(0.05), 0.0])
    # Rotating the pitched frame by the reported vector should reach upright.
    error = orientation_error(pitched, upright)
    assert error[1] < 0.0
    assert abs(np.linalg.norm(error) - 0.1) < 1e-6
    assert quat_to_rpy(pitched)[1] == pytest.approx(0.1)


# -- one-shot controller invariants --------------------------------------


def test_contact_forces_carry_exactly_the_body_weight(controller, nominal):
    controller.compute_torque(*nominal)
    telemetry = controller.telemetry
    weight = controller.robot.total_mass * GRAVITY
    assert telemetry.contact_forces[:, 2].sum() == pytest.approx(weight, rel=1e-3)
    # Nothing should be pulling sideways when the robot is simply standing.
    assert abs(telemetry.contact_forces[:, :2].sum()) < 1.0


def test_solved_forces_respect_the_friction_cone(controller, nominal):
    controller.compute_torque(*nominal)
    forces = controller.telemetry.contact_forces
    mu = controller.config.wbc.friction_mu
    tangential = np.linalg.norm(forces[:, :2], axis=1)
    assert np.all(tangential <= mu * forces[:, 2] * 1.01 + 1e-6)
    assert np.all(forces[:, 2] >= -1e-6)


def test_torques_stay_within_actuator_limits(controller, nominal):
    torque = controller.compute_torque(*nominal)
    assert np.all(np.abs(torque) <= controller.robot.torque_limit + 1e-6)


def test_standing_torques_are_small(controller, nominal):
    """A balanced double-support stance should cost almost nothing to hold."""
    torque = controller.compute_torque(*nominal)
    assert np.abs(torque).max() < 15.0


def test_wbc_torque_is_consistent_with_the_equation_of_motion(controller, nominal):
    """τ must be exactly the actuated rows of ``M q̈ + h − J_cᵀ f``."""
    qpos, qvel = nominal
    controller.compute_torque(qpos, qvel)
    robot = controller.robot
    dynamics = robot.dynamics(qvel)
    solution = controller.wbc._qp._solution
    qacc, forces = solution[: robot.nv], solution[robot.nv :]
    expected = (
        dynamics.mass_matrix @ qacc + dynamics.bias - dynamics.contact_jacobian.T @ forces
    )[QVEL_JOINT0:]
    np.testing.assert_allclose(controller.telemetry.torque, expected, atol=1e-6)


def test_unactuated_base_rows_are_balanced_by_contact_forces(controller, nominal):
    """The floating base has no motor, so its six rows must close on their own."""
    qpos, qvel = nominal
    controller.compute_torque(qpos, qvel)
    robot = controller.robot
    dynamics = robot.dynamics(qvel)
    solution = controller.wbc._qp._solution
    qacc, forces = solution[: robot.nv], solution[robot.nv :]
    residual = (
        dynamics.mass_matrix @ qacc + dynamics.bias - dynamics.contact_jacobian.T @ forces
    )[:QVEL_JOINT0]
    assert np.abs(residual).max() < 1e-2


def test_qp_falls_back_to_the_previous_answer_on_a_bad_solve():
    """A failed solve must hold the last command, not emit its stray iterate.

    OSQP reports infeasibility through a status code while still handing back
    a finite ``x``, so a controller that only screens for NaN will happily
    apply a nonsense torque -- which is exactly how this one used to fall.
    """
    # Two rows on one variable, so infeasibility comes from the pair of bounds
    # rather than from any single malformed row (which OSQP would reject
    # outright instead of reporting).
    qp = QP("test", constraint_pattern=Sparsity.dense(2, 1))
    hessian, gradient = np.eye(1), np.zeros(1)
    constraint = np.array([[1.0], [1.0]])
    infinite = np.array([np.inf, np.inf])
    good = qp.solve(hessian, gradient, constraint, -infinite, np.array([1.0, 1.0]))
    assert qp.failures == 0

    # x ≥ 1 and x ≤ −1 at the same time.
    bad = qp.solve(
        hessian,
        gradient,
        constraint,
        np.array([1.0, -np.inf]),
        np.array([np.inf, -1.0]),
    )
    assert qp.failures == 1
    np.testing.assert_allclose(bad, good)


def test_update_weights_rejects_unknown_fields(controller):
    with pytest.raises(KeyError):
        controller.update_weights(not_a_real_weight=1.0)


def test_update_weights_rejects_an_ambiguous_bare_field_name(controller):
    """``weight_force`` exists on both layers, 3000x apart. Refuse to guess."""
    with pytest.raises(KeyError):
        controller.update_weights(weight_force=2e-6)


def test_update_weights_takes_effect(controller):
    original = controller.tunables()
    try:
        controller.update_weights(
            **{"wbc.weight_base_position": 123.0, "mpc.weight_force": 2e-6}
        )
        assert controller.config.wbc.weight_base_position == 123.0
        assert controller.config.mpc.weight_force == 2e-6
        # The same-named WBC field must be untouched.
        assert controller.config.wbc.weight_force == original["wbc.weight_force"]
    finally:
        controller.update_weights(**original)


def test_update_weights_clamps_into_the_declared_range(controller):
    """A hallucinated exponent should degrade the stance, not the solver."""
    original = controller.tunables()
    try:
        applied = controller.update_weights(**{"wbc.kp_base_position": 1e9})
        limit = tuning.BY_PATH["wbc.kp_base_position"].high
        assert applied["wbc.kp_base_position"] == limit
        assert controller.config.wbc.kp_base_position == limit
    finally:
        controller.update_weights(**original)


def test_every_tunable_default_sits_inside_its_declared_range(controller):
    for path, value in controller.tunables().items():
        entry = tuning.BY_PATH[path]
        assert entry.low <= value <= entry.high, f"{path}={value} outside its range"


def test_tunable_paths_resolve_to_real_config_fields(controller):
    """Guards against a path drifting away from the dataclass it names."""
    for entry in tuning.TUNABLES:
        owner = (
            controller.config
            if entry.group == "stand"
            else getattr(controller.config, entry.group)
        )
        assert hasattr(owner, entry.field), f"{entry.path} -> missing {entry.field}"


def test_yaml_config_reproduces_the_dataclass_defaults(tmp_path):
    from pathlib import Path

    config = LocomotionConfig.from_yaml(Path("loka/config/g1.yaml"))
    defaults = LocomotionConfig()
    assert config.control_dt == defaults.control_dt
    assert config.mpc.horizon == defaults.mpc.horizon
    assert config.wbc.kp_posture_legs == defaults.wbc.kp_posture_legs
    assert config.mpc.weight_force == defaults.mpc.weight_force


def test_dashboard_drives_the_sim_without_a_display():
    """The non-GUI half of the dashboard: stepping, pushing, resetting.

    Building widgets needs a display, so only the parts that touch the
    controller are exercised here.
    """
    from loka.dashboard import Dashboard

    sim = Simulation()
    dashboard = Dashboard(sim, show_robot=False)
    assert dashboard.steps_per_frame > 1, "render rate is not decoupled from control"

    for _ in range(5):
        dashboard._advance()
    assert len(dashboard.trace["com_error"]) == 5
    assert sim.data.time > 0.0

    dashboard._push((1.0, 0.0, 0.0), impulse=4.0)
    assert len(sim.pushes) == 1

    dashboard._reset()
    assert sim.data.time == 0.0
    assert not sim.pushes
    assert not dashboard.trace["com_error"]


# -- closed loop ----------------------------------------------------------


@pytest.mark.slow
def test_holds_a_stand():
    stats = Simulation().run(4.0)
    assert not stats.fell
    assert stats.com_error_max < 5e-3
    assert stats.tilt_max < np.radians(1.0)


@pytest.mark.slow
def test_recovers_from_a_moderate_push():
    push = Push(impulse=5.0, direction=np.array([1.0, 0.0, 0.0]), time=1.0)
    sim = Simulation(pushes=[push])
    stats = sim.run(5.0)
    assert not stats.fell
    # Back on target well before the run ends.
    assert np.linalg.norm(sim.controller.telemetry.com_error) < 0.01


@pytest.mark.slow
@pytest.mark.parametrize("height", [0.55, 0.62, 0.70])
def test_tracks_commanded_height(height):
    sim = Simulation()
    sim.controller.command.height = height
    stats = sim.run(4.0)
    assert not stats.fell
    telemetry = sim.controller.telemetry
    assert abs(telemetry.com[2] - telemetry.com_reference[2]) < 5e-3


@pytest.mark.slow
def test_recovers_from_an_asymmetric_crouch():
    """Start off-nominal: the controller has to pull itself onto the target."""
    sim = Simulation()
    sim.data.qpos[7 + 3] += 0.15  # left knee
    sim.data.qpos[7 + 9] -= 0.10  # right knee
    stats = settled(sim).run(4.0)
    assert not stats.fell
    assert stats.com_error_max < 0.03


@pytest.mark.slow
@pytest.mark.parametrize("drop_height", [0.02, 0.05, 0.10])
def test_recovers_from_a_drop(drop_height):
    """Land and re-stabilise.

    Contact detection has to be anchored to the ground rather than to the
    lowest sole point, otherwise a robot in mid-air reads as fully planted and
    the QP braces against support that is not there.
    """
    sim = Simulation()
    sim.data.qpos[2] += drop_height
    mujoco.mj_forward(sim.model, sim.data)
    stats = sim.run(4.0)
    assert not stats.fell
    assert stats.com_error_max < 0.02


@pytest.mark.slow
@pytest.mark.parametrize("pitch_deg", [5.0, -5.0])
def test_recovers_from_a_pitched_start(pitch_deg):
    """A tilt small enough to keep the CoM over the (rotated) soles."""
    sim = Simulation()
    half = np.radians(pitch_deg) / 2.0
    sim.data.qpos[3:7] = [np.cos(half), 0.0, np.sin(half), 0.0]
    mujoco.mj_forward(sim.model, sim.data)
    stats = sim.run(4.0)
    assert not stats.fell
    assert stats.com_error_max < 0.02


@pytest.mark.slow
def test_absorbs_velocity_up_to_the_capture_point_limit():
    """Recover inside the capture region, and only inside it.

    A standing controller cannot beat the capture point: once ``c + ċ/ω``
    leaves the support polygon, no contact force can arrest the fall and the
    robot must step. Both halves matter -- the second guards against "fixing"
    a fall by quietly loosening what counts as falling.
    """
    reference = Simulation().controller.robot
    mujoco.mj_forward(reference.model, reference.data)
    limit = reference.capture_velocity_limit()[1]  # forward
    assert 0.25 < limit < 0.45, f"stance geometry moved: limit {limit:.3f} m/s"

    inside = Simulation()
    inside.data.qvel[0] = 0.8 * limit
    mujoco.mj_forward(inside.model, inside.data)
    stats = inside.run(4.0)
    assert not stats.fell, f"failed to capture {0.8 * limit:.3f} m/s"

    outside = Simulation()
    outside.data.qvel[0] = 1.5 * limit
    mujoco.mj_forward(outside.model, outside.data)
    assert outside.run(4.0).fell, "claimed to recover past the capture point"


@pytest.mark.slow
def test_control_loop_keeps_up_with_its_own_period():
    config = LocomotionConfig()
    stats = Simulation(config).run(3.0)
    # Mean cost must fit inside the control period, with headroom for the
    # decimated MPC ticks that land on top of it.
    assert stats.solve_ms_mean < config.control_dt * 1e3
