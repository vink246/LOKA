"""Tests for the domain-randomized Walker PPO baseline (no MJPC required)."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import yaml

from loka.dr_rl.config import DEFAULT_XML_PATH, GYM_CONFIG_PATH, load_dr_config
from loka.dr_rl.observation import (
    ROOTX_QPOS,
    build_observation,
    gait_phase_features,
    observation_size as obs_size,
)
from loka.dr_rl.randomize import (
    apply_reset_randomization,
    capture_nominal_dynamics,
    restore_nominal_dynamics,
    sample_reset_randomization,
)
from loka.walker_suite.config import parse_suite_dict
from loka.walker_suite.outcomes import TORSO_Z0, pos_x, world_height

WALKER_XML = Path(__file__).resolve().parent.parent / "models" / "walker" / "task.xml"


def _load_walker():
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(WALKER_XML))
    data = mujoco.MjData(model)
    return mujoco, model, data


class ConfigTests(unittest.TestCase):
    def test_default_yaml_loads(self):
        cfg = load_dr_config()
        dr = cfg["domain_randomization"]
        self.assertGreater(dr["friction"][0], 0.2)
        self.assertGreater(dr["gear_scale"][0], 0.0)
        self.assertLess(dr["mass_scale"][1], 1.25)
        self.assertEqual(cfg["frame_skip"], 4)
        self.assertGreater(cfg["reward"]["forward_weight"], 0.0)
        self.assertLessEqual(cfg["reward"]["healthy_bonus"], cfg["reward"]["forward_weight"])
        self.assertGreater(cfg["reward"]["overspeed_weight"], 2.0 * cfg["reward"]["forward_weight"])
        self.assertGreater(cfg["reward"]["height_deadband"], 0.0)
        self.assertGreater(cfg["reward"]["pitch_deadband"], 0.0)
        self.assertGreater(cfg["reward"]["slip_weight"], 0.0)
        self.assertGreater(cfg["reward"]["stride_weight"], 0.0)
        self.assertGreater(cfg["reward"]["stride_cap"], cfg["reward"]["min_stride"])
        self.assertGreater(cfg["reward"]["clearance_weight"], 0.0)
        self.assertGreater(cfg["reward"]["clearance_target"], 0.0)
        self.assertGreater(cfg["reward"]["gait_weight"], 0.0)
        self.assertGreater(cfg["reward"]["gait_period"], 0.0)
        self.assertGreater(cfg["reward"]["gait_threshold"], 0.5)
        self.assertEqual(len(cfg["reward"]["gait_offsets"]), 2)
        self.assertGreaterEqual(cfg["reward"]["pose_weight"], 1.0)
        self.assertGreater(cfg["reward"]["sym_weight"], 0.0)
        self.assertGreater(cfg["reward"]["lead_weight"], 0.0)
        self.assertGreater(cfg["reward"]["hip_rom_weight"], 0.0)
        self.assertGreater(
            cfg["reward"]["hip_rom_goal"], cfg["reward"]["hip_amp"]
        )
        self.assertTrue(cfg["observation"]["include_gait_phase"])
        self.assertGreater(cfg["reward"]["flight_cost"], 0.0)
        self.assertGreater(cfg["reward"]["height_cost"], 0.0)
        self.assertGreaterEqual(cfg["reward"]["healthy_bonus"], 1.0)
        self.assertEqual(cfg["reward"]["healthy_min_speed"], 0.0)
        self.assertEqual(cfg["healthy"]["height_range"], [0.8, 2.0])
        self.assertGreaterEqual(cfg["healthy"]["pitch_abs"], 1.0)
        self.assertGreater(cfg["ppo"]["nominal_warmup_steps"], 0)

    def test_xml_exists(self):
        self.assertTrue(DEFAULT_XML_PATH.is_file())
        self.assertTrue(WALKER_XML.is_file())

    def test_gym_harness_is_isolated_and_has_no_gait_terms(self):
        gait = load_dr_config()
        gym = load_dr_config(GYM_CONFIG_PATH)
        self.assertNotEqual(gym["checkpoint"]["dir"], gait["checkpoint"]["dir"])
        self.assertNotEqual(gym["checkpoint"]["best_dir"], gait["checkpoint"]["best_dir"])
        self.assertNotEqual(gym["checkpoint"]["zip_name"], gait["checkpoint"]["zip_name"])
        self.assertNotEqual(
            gym["checkpoint"]["name_prefix"],
            gait["checkpoint"].get("name_prefix", "ppo_walker"),
        )
        self.assertFalse(gym["observation"]["include_gait_phase"])
        for key in (
            "gait_weight",
            "pose_weight",
            "sym_weight",
            "lead_weight",
            "hip_rom_weight",
            "slip_weight",
            "stride_weight",
            "clearance_weight",
        ):
            self.assertEqual(gym["reward"][key], 0.0)
        self.assertEqual(gym["reward"]["forward_weight"], gait["reward"]["forward_weight"])
        self.assertEqual(gym["reward"]["overspeed_weight"], gait["reward"]["overspeed_weight"])
        self.assertEqual(gym["reward"]["height_target"], gait["reward"]["height_target"])
        self.assertGreater(gym["reward"]["height_cost"], gait["reward"]["height_cost"])
        self.assertEqual(gym["reward"]["pitch_cost"], gait["reward"]["pitch_cost"])
        self.assertEqual(gym["healthy"]["height_range"], gait["healthy"]["height_range"])
        self.assertTrue(gym["healthy"]["terminate_when_unhealthy"])
        self.assertFalse(
            bool((gait.get("healthy") or {}).get("terminate_when_unhealthy", False))
        )

    def test_suite_accepts_dr_rl_baseline(self):
        payload = {
            "baselines": ["dr_rl"],
            "tests": [{"name": "ice", "perturbation": {"kind": "friction", "mu": 0.2}}],
        }
        config = parse_suite_dict(payload)
        self.assertEqual(config.baselines, ["dr_rl"])


class RandomizationTests(unittest.TestCase):
    def test_reset_dr_stays_in_range_and_restores(self):
        mujoco, model, data = _load_walker()
        nominal = capture_nominal_dynamics(model)
        cfg = load_dr_config()["domain_randomization"]
        rng = np.random.default_rng(0)
        sample = sample_reset_randomization(rng, model, cfg)
        self.assertTrue(np.all(sample.gear_scale > 0.0))
        self.assertGreaterEqual(sample.mu, cfg["friction"][0])
        self.assertLessEqual(sample.mu, cfg["friction"][1])
        apply_reset_randomization(model, data, nominal, sample)
        self.assertAlmostEqual(
            float(model.geom_friction[nominal.floor_id, 0]), sample.mu, places=6
        )
        np.testing.assert_allclose(
            model.actuator_gear[:, 0],
            nominal.plant.actuator_gear * sample.gear_scale,
        )
        restore_nominal_dynamics(model, data, nominal)
        np.testing.assert_allclose(
            model.actuator_gear[:, 0], nominal.plant.actuator_gear
        )
        np.testing.assert_allclose(model.body_mass, nominal.plant.body_mass)
        self.assertAlmostEqual(
            float(model.geom_friction[nominal.floor_id, 0]),
            float(nominal.plant.geom_friction[nominal.floor_id, 0]),
            places=6,
        )

    def test_rejects_zero_gear_range(self):
        mujoco, model, _data = _load_walker()
        rng = np.random.default_rng(1)
        bad = {
            "friction": [0.45, 1.2],
            "mass_scale": [0.85, 1.15],
            "gear_scale": [0.0, 1.0],
            "damping_scale": [0.75, 1.25],
        }
        with self.assertRaises(ValueError):
            sample_reset_randomization(rng, model, bad)


class ObservationTests(unittest.TestCase):
    def test_obs_excludes_rootx_and_matches_qpos_layout(self):
        mujoco, model, data = _load_walker()
        mujoco.mj_resetData(model, data)
        last = np.zeros(model.nu, dtype=np.float32)
        obs = build_observation(data, last)
        self.assertEqual(obs.shape, (obs_size(model),))
        self.assertEqual(obs_size(model), (model.nq - 1) + model.nv + model.nu)
        phase = gait_phase_features(0.0, 0.8)
        obs_p = build_observation(data, last, gait_phase=phase)
        self.assertEqual(
            obs_p.shape,
            (obs_size(model, include_gait_phase=True),),
        )
        np.testing.assert_allclose(obs_p[-2:], phase)
        # rootz then rooty (rootx dropped)
        self.assertAlmostEqual(float(obs[0]), float(data.qpos[0]), places=5)
        self.assertAlmostEqual(float(obs[1]), float(data.qpos[ROOTX_QPOS + 1]), places=5)
        self.assertEqual(int(ROOTX_QPOS), 1)
        self.assertGreater(world_height(data), 1.0)
        self.assertAlmostEqual(pos_x(data), float(data.qpos[1]), places=5)


class EnvTests(unittest.TestCase):
    def test_gym_env_reset_and_step(self):
        try:
            from loka.dr_rl.env import LokaWalkerDREnv
        except ImportError as exc:
            raise unittest.SkipTest(str(exc)) from exc

        env = LokaWalkerDREnv(domain_randomize=True, observation_noise=False)
        obs, info = env.reset(seed=0)
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertIn("randomization", info)
        gears = np.asarray(info["randomization"]["gear_scale"])
        self.assertTrue(np.all(gears > 0.0))
        next_obs, reward, terminated, truncated, step_info = env.step(
            env.action_space.sample()
        )
        self.assertEqual(next_obs.shape, obs.shape)
        self.assertIsInstance(reward, float)
        self.assertFalse(terminated and truncated)
        self.assertIn("vel_x", step_info)
        self.assertIn("reward_gait", step_info)
        self.assertIn("reward_pose", step_info)
        self.assertIn("reward_sym", step_info)
        self.assertIn("reward_lead", step_info)
        self.assertIn("reward_hip_rom", step_info)
        self.assertIn("reward_forward", step_info)
        self.assertIn("n_foot_contacts", step_info)
        env.close()

    def test_gym_style_env_is_23d_and_zeros_gait_costs(self):
        try:
            from loka.dr_rl.env import LokaWalkerDREnv
        except ImportError as exc:
            raise unittest.SkipTest(str(exc)) from exc

        cfg = load_dr_config(GYM_CONFIG_PATH)
        env = LokaWalkerDREnv(
            config=cfg, domain_randomize=False, observation_noise=False
        )
        obs, _info = env.reset(seed=0, options={"disable_dr": True})
        self.assertFalse(env.include_gait_phase)
        self.assertEqual(obs.shape, (obs_size(env.model, include_gait_phase=False),))
        _next, _reward, _term, _trunc, step_info = env.step(np.zeros(env.model.nu))
        self.assertAlmostEqual(step_info["reward_gait"], 0.0)
        self.assertAlmostEqual(step_info["reward_pose"], 0.0)
        self.assertAlmostEqual(step_info["reward_sym"], 0.0)
        self.assertAlmostEqual(step_info["reward_lead"], 0.0)
        self.assertAlmostEqual(step_info["reward_hip_rom"], 0.0)
        self.assertAlmostEqual(step_info["reward_slip"], 0.0)
        self.assertAlmostEqual(step_info["reward_stride"], 0.0)
        self.assertAlmostEqual(step_info["reward_clearance"], 0.0)
        self.assertTrue(env.terminate_when_unhealthy)
        import mujoco

        env.data.qpos[0] = 0.70 - TORSO_Z0
        mujoco.mj_forward(env.model, env.data)
        _n, _r, term, _tr, _i = env.step(np.zeros(env.model.nu))
        self.assertTrue(term)
        env.close()

    def test_gait_env_does_not_cut_at_gym_unhealthy_height(self):
        try:
            from loka.dr_rl.env import LokaWalkerDREnv
        except ImportError as exc:
            raise unittest.SkipTest(str(exc)) from exc

        import mujoco

        env = LokaWalkerDREnv(domain_randomize=False, observation_noise=False)
        env.reset(seed=0, options={"disable_dr": True})
        self.assertFalse(env.terminate_when_unhealthy)
        env.data.qpos[0] = 0.70 - TORSO_Z0
        mujoco.mj_forward(env.model, env.data)
        _n, _r, term, _tr, _i = env.step(np.zeros(env.model.nu))
        self.assertFalse(term)
        env.close()

    def test_no_dr_keeps_nominal_friction(self):
        try:
            from loka.dr_rl.env import LokaWalkerDREnv
        except ImportError as exc:
            raise unittest.SkipTest(str(exc)) from exc

        env = LokaWalkerDREnv(domain_randomize=False)
        env.reset(seed=1)
        floor = env.nominal.floor_id
        self.assertAlmostEqual(
            float(env.model.geom_friction[floor, 0]),
            float(env.nominal.plant.geom_friction[floor, 0]),
            places=6,
        )
        env.close()

    def test_set_domain_randomize_toggles(self):
        try:
            from loka.dr_rl.env import LokaWalkerDREnv
        except ImportError as exc:
            raise unittest.SkipTest(str(exc)) from exc

        env = LokaWalkerDREnv(domain_randomize=True, observation_noise=True)
        env.set_domain_randomize(False)
        self.assertFalse(env.domain_randomize)
        self.assertFalse(env.observation_noise)
        env.set_domain_randomize(True)
        self.assertTrue(env.domain_randomize)
        env.close()


class RewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from loka.dr_rl.env import locomotion_reward
        except ImportError as exc:
            raise unittest.SkipTest(str(exc)) from exc
        cls._reward_fn = staticmethod(locomotion_reward)

    def _terms(self, vx: float, healthy: bool = True):
        action = np.zeros(6, dtype=np.float32)
        return self._reward_fn(
            vx,
            action,
            speed_goal=1.0,
            forward_weight=1.0,
            overspeed_weight=3.0,
            healthy=healthy,
            healthy_bonus=1.0,
            healthy_min_speed=0.0,
            pitch_val=0.0,
            height=1.2,
            height_target=1.2,
            ctrl_cost_weight=0.0,
            pitch_cost_weight=0.0,
            height_cost_weight=0.0,
        )

    def test_speed_tracks_goal_and_penalizes_overspeed(self):
        r_back, _ = self._terms(-0.4)
        r_stand, info_stand = self._terms(0.0)
        r_shuffle, _ = self._terms(0.3)
        r_goal, info_goal = self._terms(1.0)
        r_fast, _ = self._terms(1.5)
        r_sprint, info_sprint = self._terms(2.0)
        self.assertLess(r_back, r_stand)
        self.assertLess(r_stand, r_shuffle)
        self.assertLess(r_shuffle, r_goal)
        self.assertLess(r_fast, r_goal)
        self.assertLess(r_sprint, r_goal)
        self.assertAlmostEqual(info_stand["reward_forward"], 0.0, places=6)
        self.assertAlmostEqual(info_stand["reward_healthy"], 1.0, places=6)
        self.assertAlmostEqual(info_goal["reward_forward"], 1.0, places=6)
        self.assertAlmostEqual(info_sprint["reward_forward"], -1.0, places=6)

    def test_short_walk_beats_long_stand(self):
        r_stand, _ = self._terms(0.0)
        r_walk, _ = self._terms(1.0)
        self.assertGreater(r_walk, r_stand)
        self.assertGreater(1000 * r_walk, 1000 * r_stand)

    def test_first_step_with_bounce_beats_standing(self):
        action = np.zeros(6, dtype=np.float32)
        shared = dict(
            speed_goal=1.0,
            forward_weight=3.0,
            overspeed_weight=9.0,
            healthy=True,
            healthy_bonus=0.05,
            healthy_min_speed=0.15,
            ctrl_cost_weight=0.0,
            pitch_cost_weight=1.0,
            height_cost_weight=4.0,
            height_target=1.2,
            height_deadband=0.10,
            pitch_deadband=0.25,
            vz_cost_weight=0.0,
            flight_cost_weight=0.4,
        )
        r_stand, _ = self._reward_fn(
            0.0, action, pitch_val=0.0, height=1.3, vz=0.0, n_foot_contacts=2, **shared
        )
        r_step, _ = self._reward_fn(
            0.3, action, pitch_val=0.15, height=1.25, vz=0.25, n_foot_contacts=1, **shared
        )
        self.assertGreater(r_step, r_stand)

    def test_deadband_ignores_walk_bob(self):
        action = np.zeros(6, dtype=np.float32)
        kwargs = dict(
            speed_goal=1.0,
            forward_weight=3.0,
            overspeed_weight=9.0,
            healthy=True,
            healthy_bonus=0.05,
            healthy_min_speed=0.15,
            ctrl_cost_weight=0.0,
            pitch_cost_weight=1.0,
            height_cost_weight=4.0,
            height_target=1.2,
            height_deadband=0.10,
            pitch_deadband=0.25,
        )
        r_center, info_c = self._reward_fn(
            1.0, action, pitch_val=0.0, height=1.2, **kwargs
        )
        r_bob, info_b = self._reward_fn(
            1.0, action, pitch_val=0.20, height=1.28, **kwargs
        )
        self.assertAlmostEqual(info_c["reward_height"], 0.0, places=6)
        self.assertAlmostEqual(info_b["reward_height"], 0.0, places=6)
        self.assertAlmostEqual(info_b["reward_pitch"], 0.0, places=6)
        self.assertAlmostEqual(r_center, r_bob, places=6)
        _, info_low = self._reward_fn(
            1.0, action, pitch_val=0.0, height=1.00, **kwargs
        )
        self.assertLess(info_low["reward_height"], 0.0)

    def test_slip_makes_shuffle_lose_to_a_step(self):
        action = np.zeros(6, dtype=np.float32)
        kwargs = dict(
            speed_goal=1.0,
            forward_weight=3.0,
            overspeed_weight=9.0,
            healthy=True,
            healthy_bonus=0.05,
            healthy_min_speed=0.15,
            ctrl_cost_weight=0.0,
            pitch_cost_weight=0.0,
            height_cost_weight=0.0,
            height_target=1.2,
            slip_weight=1.0,
            stride_weight=2.0,
            flight_cost_weight=2.0,
        )
        r_walk, info_w = self._reward_fn(
            1.0,
            action,
            pitch_val=0.0,
            height=1.2,
            n_foot_contacts=1,
            foot_slip=0.05,
            stride=0.30,
            **kwargs,
        )
        r_shuffle, info_s = self._reward_fn(
            1.12,
            action,
            pitch_val=0.07,
            height=1.09,
            n_foot_contacts=2,
            foot_slip=2.24,
            stride=0.0,
            **kwargs,
        )
        self.assertGreater(r_walk, r_shuffle)
        self.assertLess(info_s["reward_slip"], info_w["reward_slip"])
        self.assertGreater(info_w["reward_stride"], info_s["reward_stride"])

    def test_height_pitch_bounce_and_flight_cost_a_skip(self):
        action = np.zeros(6, dtype=np.float32)
        kwargs = dict(
            speed_goal=1.0,
            forward_weight=3.0,
            overspeed_weight=9.0,
            healthy=True,
            healthy_bonus=0.05,
            healthy_min_speed=0.15,
            ctrl_cost_weight=0.0,
            pitch_cost_weight=1.0,
            height_cost_weight=25.0,
            height_target=1.2,
            height_deadband=0.10,
            pitch_deadband=0.25,
            vz_cost_weight=0.0,
            flight_cost_weight=2.0,
        )
        r_walk, _ = self._reward_fn(
            1.0,
            action,
            pitch_val=0.0,
            height=1.2,
            n_foot_contacts=1,
            **kwargs,
        )
        r_skip, info = self._reward_fn(
            2.05,
            action,
            pitch_val=0.19,
            height=1.15,
            n_foot_contacts=0,
            **kwargs,
        )
        self.assertGreater(r_walk, r_skip)
        self.assertLess(info["reward_forward"], 0.0)
        self.assertLess(info["reward_flight"], 0.0)

    def test_one_leg_hold_loses_to_alternating_steps(self):
        action = np.zeros(6, dtype=np.float32)
        kwargs = dict(
            speed_goal=1.0,
            forward_weight=3.0,
            overspeed_weight=9.0,
            healthy=True,
            healthy_bonus=0.05,
            healthy_min_speed=0.15,
            ctrl_cost_weight=0.0,
            pitch_cost_weight=0.0,
            height_cost_weight=0.0,
            height_target=1.2,
            stride_weight=2.0,
            flight_cost_weight=2.0,
        )
        r_walk, info_w = self._reward_fn(
            1.0,
            action,
            pitch_val=0.0,
            height=1.2,
            n_foot_contacts=1,
            stride=0.30,
            **kwargs,
        )
        r_hop, info_h = self._reward_fn(
            1.16,
            action,
            pitch_val=0.39,
            height=1.08,
            n_foot_contacts=0,
            stride=0.0,
            **kwargs,
        )
        self.assertGreater(r_walk, r_hop)
        self.assertGreater(info_w["reward_stride"], info_h["reward_stride"])
        self.assertLess(info_h["reward_flight"], 0.0)

    def test_clearance_prefers_a_lifted_swing_over_a_low_slide(self):
        action = np.zeros(6, dtype=np.float32)
        kwargs = dict(
            speed_goal=1.0,
            forward_weight=1.0,
            overspeed_weight=3.0,
            healthy=True,
            healthy_bonus=1.0,
            ctrl_cost_weight=0.0,
            pitch_cost_weight=0.0,
            height_cost_weight=0.0,
            height_target=1.2,
            clearance_weight=0.5,
        )
        r_clear, info_c = self._reward_fn(
            1.0,
            action,
            pitch_val=0.0,
            height=1.2,
            n_foot_contacts=1,
            clearance=1.0,
            **kwargs,
        )
        r_low, info_l = self._reward_fn(
            1.0,
            action,
            pitch_val=0.0,
            height=1.2,
            n_foot_contacts=1,
            clearance=0.70,
            **kwargs,
        )
        self.assertGreater(r_clear, r_low)
        self.assertGreater(info_c["reward_clearance"], info_l["reward_clearance"])

    def test_gait_bonus_needs_speed_and_prefers_a_match(self):
        action = np.zeros(6, dtype=np.float32)
        kwargs = dict(
            speed_goal=1.0,
            forward_weight=1.0,
            overspeed_weight=3.0,
            healthy=True,
            healthy_bonus=1.0,
            ctrl_cost_weight=0.0,
            pitch_cost_weight=0.0,
            height_cost_weight=0.0,
            height_target=1.2,
            gait_weight=0.5,
        )
        _, info_stand = self._reward_fn(
            0.0,
            action,
            pitch_val=0.0,
            height=1.2,
            gait_match=2.0,
            **kwargs,
        )
        _, info_match = self._reward_fn(
            1.0,
            action,
            pitch_val=0.0,
            height=1.2,
            gait_match=2.0,
            **kwargs,
        )
        _, info_miss = self._reward_fn(
            1.0,
            action,
            pitch_val=0.0,
            height=1.2,
            gait_match=0.0,
            **kwargs,
        )
        self.assertAlmostEqual(info_stand["reward_gait"], 0.0)
        self.assertAlmostEqual(info_match["reward_gait"], 1.0)
        self.assertGreater(info_match["reward_gait"], info_miss["reward_gait"])

    def test_pose_and_sym_apply_at_stand_gait_does_not(self):
        action = np.zeros(6, dtype=np.float32)
        kwargs = dict(
            speed_goal=1.0,
            forward_weight=1.0,
            overspeed_weight=3.0,
            healthy=True,
            healthy_bonus=1.0,
            ctrl_cost_weight=0.0,
            pitch_cost_weight=0.0,
            height_cost_weight=0.0,
            height_target=1.2,
            pose_weight=2.0,
            sym_weight=0.5,
            gait_weight=0.5,
        )
        _, info_stand = self._reward_fn(
            0.0,
            action,
            pitch_val=0.0,
            height=1.2,
            pose_err=1.0,
            sym_err=1.0,
            gait_match=2.0,
            **kwargs,
        )
        _, info_walk = self._reward_fn(
            1.0,
            action,
            pitch_val=0.0,
            height=1.2,
            pose_err=1.0,
            sym_err=0.5,
            gait_match=2.0,
            **kwargs,
        )
        _, info_lopsided = self._reward_fn(
            1.0,
            action,
            pitch_val=0.0,
            height=1.2,
            pose_err=1.0,
            sym_err=2.0,
            **kwargs,
        )
        self.assertAlmostEqual(info_stand["reward_pose"], -2.0)
        self.assertAlmostEqual(info_stand["reward_sym"], -0.5)
        self.assertAlmostEqual(info_stand["reward_gait"], 0.0)
        self.assertAlmostEqual(info_walk["reward_pose"], -2.0)
        self.assertAlmostEqual(info_walk["reward_sym"], -0.25)
        self.assertAlmostEqual(info_walk["reward_gait"], 1.0)
        self.assertLess(info_lopsided["reward_sym"], info_walk["reward_sym"])

    def test_compass_loses_to_a_tracking_scissor(self):
        action = np.zeros(6, dtype=np.float32)
        kwargs = dict(
            speed_goal=1.0,
            forward_weight=1.0,
            overspeed_weight=3.0,
            healthy=True,
            healthy_bonus=1.0,
            ctrl_cost_weight=0.0,
            pitch_cost_weight=0.0,
            height_cost_weight=0.0,
            height_target=1.2,
            pose_weight=2.0,
            sym_weight=0.5,
            lead_weight=1.0,
            hip_rom_weight=1.0,
            gait_weight=0.5,
        )
        r_walk, info_w = self._reward_fn(
            1.0,
            action,
            pitch_val=0.0,
            height=1.2,
            gait_match=2.0,
            pose_err=0.20,
            sym_err=0.05,
            lead_err=0.02,
            hip_rom_err=0.0,
            **kwargs,
        )
        r_compass, info_c = self._reward_fn(
            0.85,
            action,
            pitch_val=0.19,
            height=1.21,
            gait_match=1.5,
            pose_err=1.92,
            sym_err=0.59,
            lead_err=1.04,
            hip_rom_err=0.23,
            **kwargs,
        )
        self.assertGreater(r_walk, r_compass)
        self.assertLess(info_c["reward_pose"], info_w["reward_pose"])
        self.assertLess(info_c["reward_lead"], info_w["reward_lead"])


class StrideClockTests(unittest.TestCase):
    def test_supported_touchdown_pays_and_hop_does_not(self):
        try:
            from loka.dr_rl.env import LokaWalkerDREnv
        except ImportError as exc:
            raise unittest.SkipTest(str(exc)) from exc

        env = LokaWalkerDREnv(domain_randomize=False, observation_noise=False)
        env.reset(seed=0, options={"disable_dr": True})
        env._prev_contacts[:] = True
        env._foot_air_time[:] = 0.0
        env._swing_supported[:] = False
        n_air = int(round(0.20 / env.dt))
        for _ in range(n_air):
            self.assertAlmostEqual(env._stride_on_touchdown(np.array([True, False])), 0.0)
        walk = env._stride_on_touchdown(np.array([True, True]))
        self.assertGreater(walk, 0.0)

        env._prev_contacts[:] = True
        env._foot_air_time[:] = 0.0
        env._swing_supported[:] = False
        for _ in range(n_air):
            env._stride_on_touchdown(np.array([False, False]))
        hop = env._stride_on_touchdown(np.array([True, False]))
        self.assertAlmostEqual(hop, 0.0)
        env.close()

    def test_still_feet_score_full_clearance(self):
        try:
            from loka.dr_rl.env import LokaWalkerDREnv
        except ImportError as exc:
            raise unittest.SkipTest(str(exc)) from exc

        env = LokaWalkerDREnv(domain_randomize=False, observation_noise=False)
        env.reset(seed=0, options={"disable_dr": True})
        still = env._foot_clearance()
        self.assertGreater(still, 0.95)
        env.close()

    def test_contact_clock_stand_matches_at_t0_not_flight(self):
        from loka.dr_rl.env import feet_gait_match

        self.assertAlmostEqual(feet_gait_match(np.array([True, True]), 0.0), 2.0)
        self.assertAlmostEqual(feet_gait_match(np.array([False, False]), 0.0), 0.0)
        # frac=0.3: right stance, left swing
        t = 0.3 * 0.8
        self.assertAlmostEqual(feet_gait_match(np.array([True, False]), t), 2.0)
        self.assertAlmostEqual(feet_gait_match(np.array([False, True]), t), 0.0)
        self.assertAlmostEqual(feet_gait_match(np.array([False, False]), t), 1.0)

    def test_sine_walk_is_phase_offset_not_same_time_mirror(self):
        from loka.dr_rl.env import pose_symmetry_errors, sine_walk_targets

        t0 = sine_walk_targets(0.0)
        self.assertEqual(t0.shape, (2, 3))
        # Heel-strike: right hip flexed, left (offset 0.5) extended — not equal.
        self.assertGreater(t0[0, 0], t0[1, 0])
        self.assertGreater(t0[0, 0], 0.5)
        self.assertLess(t0[1, 0], 0.2)
        # Mid-period: roles swap.
        t_half = sine_walk_targets(0.4)
        self.assertGreater(t_half[1, 0], t_half[0, 0])
        # frac=0.3: right stance, left mid-swing — swing knee more flexed.
        t_swing = sine_walk_targets(0.24)
        self.assertLess(t_swing[1, 1], t_swing[0, 1])

        pose_ok, sym_ok = pose_symmetry_errors(t0, t0)
        self.assertAlmostEqual(pose_ok, 0.0)
        self.assertAlmostEqual(sym_ok, 0.0)
        one_sided = t0.copy()
        one_sided[0] += 0.8
        pose_bad, sym_bad = pose_symmetry_errors(one_sided, t0)
        self.assertGreater(pose_bad, 0.0)
        self.assertGreater(sym_bad, 0.0)
        both = t0.copy() + 0.4
        pose_both, sym_both = pose_symmetry_errors(both, t0)
        self.assertGreater(pose_both, pose_ok)
        self.assertAlmostEqual(sym_both, 0.0)

    def test_cycle_lead_is_mean_offset_not_same_time_mirror(self):
        from loka.dr_rl.env import cycle_hip_rom_error, cycle_lead_error

        n = 80
        phase = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        # Scissor: opposite hips, same cycle mean.
        hip_r = 0.35 + 0.40 * np.cos(phase)
        hip_l = 0.35 + 0.40 * np.cos(phase + np.pi)
        self.assertAlmostEqual(cycle_lead_error(hip_r, hip_l), 0.0, places=6)
        self.assertAlmostEqual(
            cycle_hip_rom_error(hip_r, hip_l, rom_goal=0.80, ready=True),
            0.0,
            places=5,
        )
        # Compass: left stays flexed, right stays extended.
        compass_r = np.full(n, -0.21)
        compass_l = np.full(n, 0.81)
        self.assertGreater(cycle_lead_error(compass_r, compass_l), 0.9)
        self.assertGreater(
            cycle_hip_rom_error(compass_r, compass_l, rom_goal=0.80, ready=True),
            0.5,
        )
        self.assertAlmostEqual(
            cycle_hip_rom_error(compass_r, compass_l, rom_goal=0.80, ready=False),
            0.0,
        )

    def test_env_resolves_leg_joints_and_targets(self):
        try:
            from loka.dr_rl.env import LokaWalkerDREnv, sine_walk_targets
        except ImportError as exc:
            raise unittest.SkipTest(str(exc)) from exc

        env = LokaWalkerDREnv(domain_randomize=False, observation_noise=False)
        env.reset(seed=0, options={"disable_dr": True})
        for addrs in env._leg_qpos_adr:
            self.assertTrue(all(a >= 0 for a in addrs))
        pose, sym = env._pose_symmetry()
        self.assertGreaterEqual(pose, 0.0)
        self.assertGreaterEqual(sym, 0.0)
        lead, rom = env._cycle_style()
        self.assertGreaterEqual(lead, 0.0)
        self.assertAlmostEqual(rom, 0.0)  # window not full
        targets = sine_walk_targets(env._gait_time)
        self.assertEqual(targets.shape, (2, 3))
        env.close()


class YamlRoundTrip(unittest.TestCase):
    def test_config_file_is_mapping(self):
        path = Path(__file__).resolve().parent.parent / "loka" / "dr_rl" / "config.yaml"
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        self.assertIsInstance(raw, dict)
        self.assertIn("ppo", raw)

    def test_periodic_checkpoint_finds_vecnormalize(self):
        import tempfile

        from loka.dr_rl.config import _vecnormalize_beside

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            zip_path = folder / "ppo_walker_500000_steps.zip"
            vec_path = folder / "ppo_walker_vecnormalize_500000_steps.pkl"
            zip_path.write_bytes(b"x")
            vec_path.write_bytes(b"x")
            found = _vecnormalize_beside(zip_path, "vecnormalize.pkl")
            self.assertEqual(found, vec_path)


if __name__ == "__main__":
    unittest.main()
