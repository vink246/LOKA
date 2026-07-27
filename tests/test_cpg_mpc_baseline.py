"""Unit tests for the G1 CPG + MPC baseline (no DDS required)."""

from __future__ import annotations

import unittest

import numpy as np

from loka.nodes.actuator_manager import ActuatorConfig, ActuatorManager
from loka.nodes.cpg_generator import (
    CPGGenerator,
    CPGParams,
    LegKinematics,
    foot_cartesian,
    leg_ik,
)
from loka.nodes.mpc_planner import (
    ConvexQP_MPC,
    MPCConfig,
    MPCPlanner,
    PredictiveSampling_MPC,
    iLQG_MPC,
)
from loka.nodes.state_estimator import RobotState
from loka.nodes.trajectory_planner import (
    PathType,
    TorsoTrajectoryParams,
    TorsoTrajectoryPlanner,
)


class CPGTests(unittest.TestCase):
    def test_cycloid_swing_stance_continuity(self):
        p = CPGParams(sweep_amplitude=0.2, clearance=0.1, duty_factor=0.6)
        swing_end, in_swing = foot_cartesian(0.4 - 1e-9, p, y_stance=0.1)
        stance_start, in_stance = foot_cartesian(0.4, p, y_stance=0.1)
        self.assertFalse(in_swing)
        self.assertTrue(in_stance)
        self.assertAlmostEqual(swing_end[0], stance_start[0], places=5)
        self.assertAlmostEqual(swing_end[2] - p.stance_z, 0.0, places=5)

    def test_leg_ik_finite(self):
        q = leg_ik(np.array([0.0, 0.1, -0.72]), LegKinematics(), side="left")
        self.assertEqual(q.shape, (6,))
        self.assertTrue(np.all(np.isfinite(q)))

    def test_cpg_evaluate_shape(self):
        gen = CPGGenerator(params=CPGParams())
        gen.start(0.0)
        out = gen.evaluate(0.25)
        self.assertEqual(out.q_cpg.shape, (29,))
        self.assertEqual(out.foot_contacts.shape, (2,))


class TrajectoryMPCTests(unittest.TestCase):
    def test_torso_straight_path(self):
        tp = TorsoTrajectoryPlanner(
            TorsoTrajectoryParams(path_type=PathType.STRAIGHT, v_ref=0.5)
        )
        tp.reset(0.0, np.array([0.0, 0.0, 0.78]), yaw=0.0)
        ref = tp.evaluate(2.0)
        self.assertAlmostEqual(ref.position[0], 1.0)

    def test_mpc_backends(self):
        gen = CPGGenerator(params=CPGParams())
        gen.start(0.0)
        cpg = gen.evaluate(0.2)
        torso = TorsoTrajectoryPlanner(TorsoTrajectoryParams(v_ref=0.3))
        torso.reset(0.0, np.zeros(3))
        ref = torso.evaluate(0.5)
        state = RobotState(
            base_pos=np.array([0.0, 0.0, 0.75]), joint_pos=cpg.q_cpg.copy()
        )
        for cls in (ConvexQP_MPC, PredictiveSampling_MPC, iLQG_MPC):
            sol = cls(MPCConfig()).solve(state, ref, cpg)
            self.assertEqual(sol.tau.shape, (29,))
            self.assertTrue(np.all(np.isfinite(sol.tau)))
            self.assertLess(float(np.max(np.abs(sol.tau))), 100.0)

    def test_planner_facade_swap(self):
        mpc = MPCPlanner(MPCConfig(), "convex_qp")
        mpc.set_planner("Predictive Sampling")
        self.assertEqual(mpc.active_name, "predictive_sampling")
        mpc.set_planner("iLQG")
        self.assertEqual(mpc.active_name, "ilqg")


class ActuatorTests(unittest.TestCase):
    def test_mixed_modes(self):
        act = ActuatorManager(
            ActuatorConfig(default_q=np.zeros(29), enable_leg_torque=True)
        )
        cmd = act.build_command(tau_ff=np.ones(29), q_ref=np.zeros(29))
        self.assertEqual(cmd.kp[0], 0.0)
        self.assertEqual(cmd.kp[15], 40.0)
        self.assertEqual(cmd.tau[0], 1.0)
        self.assertEqual(cmd.tau[15], 0.0)
        act.set_enable_leg_torque(False)
        cmd2 = act.build_command(tau_ff=np.ones(29), q_ref=np.zeros(29))
        self.assertEqual(cmd2.kp[0], 60.0)
        self.assertEqual(cmd2.kp[3], 100.0)

    def test_torso_stand_controller_returns_29(self):
        from loka.nodes.stand_controller import TorsoStandController

        q0 = np.zeros(29)
        q0[3] = 0.30
        ctrl = TorsoStandController(q0, z_target=0.793)
        state = RobotState(
            base_pos=np.array([0.0, 0.0, 0.75]),
            base_quat=np.array([1.0, 0.0, 0.0, 0.0]),
            joint_pos=q0.copy(),
        )
        q_ref = ctrl.compute_q_ref(state)
        self.assertEqual(q_ref.shape, (29,))
        # Too low → extend knees (reduce flexion).
        self.assertLess(q_ref[3], q0[3])

    def test_stand_still_holds_default_q(self):
        default = np.zeros(29)
        default[:12] = [-0.2, 0, 0, 0.42, -0.23, 0] * 2
        gen = CPGGenerator(
            params=CPGParams(sweep_amplitude=0.0, clearance=0.0),
            default_q=default,
        )
        gen.start(0.0)
        out = gen.evaluate(0.5)
        np.testing.assert_allclose(out.q_cpg, default)
        np.testing.assert_array_equal(out.foot_contacts, [1.0, 1.0])

    def test_ik_yaw_not_singular_at_stance(self):
        """Foot nearly under hip must not slam hip_yaw to ±0.6."""
        kin = LegKinematics()
        q = leg_ik(np.array([0.0, 0.1185, -0.743]), kin, side="left")
        self.assertAlmostEqual(q[2], 0.0, places=6)
        self.assertLess(abs(q[3]), 1.5)

    def test_soft_start_interpolates(self):
        from loka.nodes.actuator_manager import SoftStarter

        ss = SoftStarter(duration=1.0)
        ss.begin(0.0, np.zeros(29), np.ones(29))
        mid = ss.evaluate(0.5)
        self.assertTrue(np.all(mid > 0.4) and np.all(mid < 0.6))
        end = ss.evaluate(1.0)
        np.testing.assert_allclose(end, np.ones(29))
        self.assertFalse(ss.active)


if __name__ == "__main__":
    unittest.main()
