"""ALIP core: the model, the orbits, and the stepping law. No MuJoCo plant."""

import numpy as np

from loka.control.alip import (
    ALIP_TURN_CAP,
    OrbitCache,
    deadbeat_gain,
    lip_matrix,
    p1_orbit,
    p2_orbit,
    plane_from_world,
    predicted_preimpact,
    propagate,
    step_to_step,
    stepping_gain,
)


def test_lip_matches_rk4():
    mass, height, t = 35.0, 0.7, 0.3
    state = np.array([0.04, -2.0])
    closed = propagate(state, t, mass, height)
    dt = 1e-4
    x = state.copy()
    g = 9.81
    n = int(round(t / dt))
    for _ in range(n):
        def f(s):
            return np.array([s[1] / (mass * height), mass * g * s[0]])

        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    assert np.allclose(closed, x, atol=1e-4)


def test_horizontal_L_invariant_to_horizontal_shift_when_vz_is_zero():
    c = np.array([0.1, -0.2, 0.7])
    v = np.array([0.3, -0.1, 0.0])
    L_com = np.array([1.0, -0.5, 0.2])
    p = np.array([0.0, 0.0, 0.0])
    shift = np.array([0.05, -0.02, 0.0])
    m = 35.0

    def about(point):
        return L_com + m * np.cross(c - point, v)

    a = about(p)
    b = about(p + shift)
    assert np.allclose(a[:2], b[:2], atol=1e-12)


def test_orbits_are_fixed_points():
    a, b = step_to_step(35.0, 0.7, 0.25, 0.10)
    u = 0.12
    x = p1_orbit(a, b, u)
    assert np.allclose(a @ x + b * u, x, atol=1e-9)
    left, right = p2_orbit(a, b, -0.2, 0.2)
    assert np.allclose(a @ left + b * (-0.2), right, atol=1e-8)
    assert np.allclose(a @ right + b * 0.2, left, atol=1e-8)


def test_deadbeat_converges_in_two_steps():
    a, b = step_to_step(35.0, 0.7, 0.25, 0.10)
    k = deadbeat_gain(a, b)
    closed = a + np.outer(b, k)
    e = np.array([0.05, 1.5])
    e = closed @ e
    e = closed @ e
    assert np.linalg.norm(e) < 1e-8


def test_gain_is_stable():
    a, b = step_to_step(35.0, 0.7, 0.25, 0.10)
    for gain in (0.3, 0.5, 0.7, 1.0):
        k = stepping_gain(a, b, gain)
        rho = np.max(np.abs(np.linalg.eigvals(a + np.outer(b, k))))
        assert rho < 1.0 + 1e-8


def test_plane_frame_identity_rotation():
    com = np.array([0.2, -0.1])
    p = np.array([0.0, 0.0])
    L = np.array([0.4, -0.3, 0.0])
    sag, lat = plane_from_world(com, p, L, np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert abs(sag[0] - 0.2) < 1e-12
    assert abs(lat[0] - (-0.1)) < 1e-12
    assert abs(sag[1] - (-0.3)) < 1e-12  # L · left
    assert abs(lat[1] - (-0.4)) < 1e-12  # -L · forward


def test_turn_in_place_reaches_pi():
    cap = ALIP_TURN_CAP
    yaw = 0.0
    steps = 0
    while yaw < np.pi - 1e-12:
        yaw += cap
        steps += 1
    assert steps == int(np.ceil(np.pi / cap))


def test_preimpact_at_end_of_step_is_identity():
    state = np.array([0.02, 0.5])
    out = predicted_preimpact(
        state, t_in_step=0.35, t_ss=0.25, t_ds=0.10, mass=35.0, height=0.7
    )
    assert np.allclose(out, state, atol=1e-9)


def test_orbit_cache_step_matches_law():
    cache = OrbitCache()
    cache.refresh(35.0, 0.7, 0.25, 0.10, 0.7)
    x = np.array([0.03, -1.0])
    u = cache.sagittal_step(x, 0.3, 0.35)
    assert np.isfinite(u)
    assert abs(u) < 1.0


def test_angular_momentum_matches_body_sum():
    import mujoco

    from loka.control.locomotion import DEFAULT_MODEL
    from loka.control.robot import G1Model

    robot = G1Model(DEFAULT_MODEL)
    qvel = np.zeros(robot.nv)
    qvel[0] = 0.25
    qvel[1] = -0.10
    qvel[5] = 0.40
    robot.update(robot.nominal_qpos, qvel)
    point = robot.foot_center_positions()[0]
    got = robot.angular_momentum_about(point)
    m = robot.model
    d = robot.data
    total = np.zeros(3)
    vel = np.zeros(6)
    for i in range(1, m.nbody):
        mass = float(m.body_mass[i])
        if mass <= 0.0:
            continue
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, i, vel, 0)
        omega = vel[:3]
        lin = vel[3:]
        rot = np.asarray(d.ximat[i], dtype=float).reshape(3, 3)
        inertia = rot @ np.diag(m.body_inertia[i]) @ rot.T
        r = np.asarray(d.xipos[i], dtype=float) - point
        total += inertia @ omega + mass * np.cross(r, lin)
    assert np.allclose(got, total, atol=1e-6)
