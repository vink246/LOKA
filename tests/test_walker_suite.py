"""Unit tests for the Walker perturbation suite (no MuJoCo MPC required)."""

from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from loka.walker_suite.config import (
    apply_cli_overrides,
    load_suite_yaml,
    parse_suite_dict,
)
from loka.model_state import apply_loka_mutations
from loka.walker_suite.faults import (
    CommandDelay,
    PlantFaults,
    assert_belief_isolated,
    plant_friction_geom_ids,
    resolve_perturbation,
    sanitize_belief_worldview,
    walker_gravity,
    walker_total_mass,
)
from loka.walker_suite.forces import (
    FOOT_GEOM_NAMES,
    ground_reaction_channels,
    ground_reaction_force,
    sample_floor_contacts,
    walker_ground_geom_ids,
)
from loka.walker_suite.logging import EpisodeLogger
from loka.walker_suite.outcomes import (
    FALL_HEIGHT_M,
    FALL_PITCH_RAD,
    classify_outcome,
    has_fallen,
    pos_x,
    world_height,
)

WALKER_XML = Path(__file__).resolve().parent.parent / "models" / "walker" / "task.xml"


def _fake_data(qpos, qvel=None, time=0.0):
    qpos = np.asarray(qpos, dtype=float)
    if qvel is None:
        qvel = np.zeros_like(qpos)
    return SimpleNamespace(qpos=qpos, qvel=np.asarray(qvel, dtype=float), time=float(time))


def _minimal_suite_dict(**overrides):
    payload = {
        "output_dir": "results/walker",
        "record": True,
        "defaults": {
            "goal_distance_m": 10.0,
            "perturbation_time_s": 3.0,
            "timeout_s": 20.0,
            "speed_goal": 1.0,
        },
        "baselines": ["loka", "fixed_mpc"],
        "tests": [
            {
                "name": "ice",
                "perturbation": {"kind": "friction", "mu": 0.2},
            },
            {
                "name": "backpack",
                "perturbation": {"kind": "mass", "mass_frac": 0.25, "body": "torso"},
            },
        ],
    }
    payload.update(overrides)
    return payload


def _load_walker():
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(WALKER_XML))
    data = mujoco.MjData(model)
    return mujoco, model, data


def _obstacle_world_pos(model, geom_id):
    body_id = int(model.geom_bodyid[geom_id])
    if body_id > 0:
        return model.body_pos[body_id]
    return model.geom_pos[geom_id]


class ConfigTests(unittest.TestCase):
    def test_parse_and_cli_override(self):
        config = parse_suite_dict(_minimal_suite_dict())
        self.assertEqual([t.name for t in config.tests], ["ice", "backpack"])
        args = Namespace(
            tests="backpack",
            baselines="fixed_mpc",
            goal_distance=8.0,
            perturbation_time=2.0,
            timeout=12.0,
            speed_goal=None,
            output_dir="tmp/out",
            no_record=True,
            record=None,
        )
        merged = apply_cli_overrides(config, args)
        self.assertEqual([t.name for t in merged.tests], ["backpack"])
        self.assertEqual(merged.baselines, ["fixed_mpc"])
        self.assertEqual(merged.defaults.goal_distance_m, 8.0)
        self.assertEqual(merged.defaults.perturbation_time_s, 2.0)
        self.assertEqual(merged.defaults.timeout_s, 12.0)
        self.assertEqual(merged.output_dir, Path("tmp/out"))
        self.assertFalse(merged.record)
        self.assertEqual(merged.num_trials, 1)
        self.assertEqual(merged.seed, 0)
        self.assertAlmostEqual(merged.init_noise, 0.005)

    def test_num_trials_cli_and_alias(self):
        config = parse_suite_dict(_minimal_suite_dict(numtrials=4, seed=7, init_noise=0.01))
        self.assertEqual(config.num_trials, 4)
        self.assertEqual(config.seed, 7)
        args = Namespace(
            tests=None,
            baselines=None,
            goal_distance=None,
            perturbation_time=None,
            timeout=None,
            speed_goal=None,
            output_dir=None,
            no_record=False,
            record=None,
            num_trials=5,
            seed=3,
            init_noise=0.0,
        )
        merged = apply_cli_overrides(config, args)
        self.assertEqual(merged.num_trials, 5)
        self.assertEqual(merged.seed, 3)
        self.assertEqual(merged.init_noise, 0.0)
        with self.assertRaises(ValueError):
            parse_suite_dict(_minimal_suite_dict(num_trials=0))

    def test_dr_rl_paths_from_yaml_and_cli(self):
        config = parse_suite_dict(
            _minimal_suite_dict(
                dr_rl_checkpoint="results/dr_rl/best",
                dr_rl_config="loka/dr_rl/config_gym.yaml",
            )
        )
        self.assertEqual(config.dr_rl_checkpoint, "results/dr_rl/best")
        self.assertEqual(config.dr_rl_config, "loka/dr_rl/config_gym.yaml")
        blank = parse_suite_dict(
            _minimal_suite_dict(dr_rl_checkpoint=None, dr_rl_config="  ")
        )
        self.assertIsNone(blank.dr_rl_checkpoint)
        self.assertIsNone(blank.dr_rl_config)
        args = Namespace(
            tests=None,
            baselines=None,
            goal_distance=None,
            perturbation_time=None,
            timeout=None,
            speed_goal=None,
            output_dir=None,
            no_record=False,
            record=None,
            num_trials=None,
            seed=None,
            init_noise=None,
            dr_rl_checkpoint="results/dr_rl/ppo_walker.zip",
            dr_rl_config=None,
        )
        merged = apply_cli_overrides(config, args)
        self.assertEqual(merged.dr_rl_checkpoint, "results/dr_rl/ppo_walker.zip")
        self.assertEqual(merged.dr_rl_config, "loka/dr_rl/config_gym.yaml")

    def test_rejects_raw_mass_and_force(self):
        with self.assertRaises(ValueError):
            parse_suite_dict(
                _minimal_suite_dict(
                    tests=[
                        {
                            "name": "bad_mass",
                            "perturbation": {"kind": "mass", "delta_kg": 5.0},
                        }
                    ]
                )
            )
        with self.assertRaises(ValueError):
            parse_suite_dict(
                _minimal_suite_dict(
                    tests=[
                        {
                            "name": "bad_force",
                            "perturbation": {
                                "kind": "force",
                                "magnitude": 80.0,
                                "direction": "backward",
                            },
                        }
                    ]
                )
            )

    def test_load_default_yaml(self):
        config = load_suite_yaml(
            Path(__file__).resolve().parent.parent
            / "loka"
            / "config"
            / "walker_suite.yaml"
        )
        names = [t.name for t in config.tests]
        self.assertIn("nominal", names)
        self.assertIn("dead_right_hip", names)
        self.assertIn("box", names)
        backpack = next(t for t in config.tests if t.name == "backpack")
        self.assertEqual(backpack.perturbation["mass_frac"], 1.5)
        self.assertEqual(config.num_trials, 1)
        self.assertEqual(config.seed, 0)
        self.assertAlmostEqual(config.init_noise, 0.005)
        self.assertEqual(config.dr_rl_checkpoint, "results/dr_rl_gym/best_model.zip")
        self.assertEqual(config.dr_rl_config, "loka/dr_rl/config_gym.yaml")
        self.assertNotIn("delta_kg", backpack.perturbation)
        ice = next(t for t in config.tests if t.name == "ice")
        self.assertLessEqual(ice.perturbation["mu"], 0.01)
        box = next(t for t in config.tests if t.name == "box")
        self.assertAlmostEqual(box.perturbation["size"][2], 0.30)
        latency = next(t for t in config.tests if t.name == "command_latency")
        self.assertEqual(latency.perturbation["kind"], "command_latency")
        self.assertEqual(latency.perturbation["delay_steps"], 8)
        self.assertEqual(latency.perturbation_time_s, 2.0)

    def test_command_latency_config(self):
        config = parse_suite_dict(
            _minimal_suite_dict(
                tests=[
                    {
                        "name": "lag",
                        "perturbation": {"kind": "command_latency", "action_buf_len": 8},
                    }
                ]
            )
        )
        self.assertEqual(config.tests[0].perturbation["delay_steps"], 8)
        self.assertNotIn("action_buf_len", config.tests[0].perturbation)
        with self.assertRaises(ValueError):
            parse_suite_dict(
                _minimal_suite_dict(
                    tests=[
                        {
                            "name": "lag",
                            "perturbation": {"kind": "command_latency"},
                        }
                    ]
                )
            )
        with self.assertRaises(ValueError):
            parse_suite_dict(
                _minimal_suite_dict(
                    tests=[
                        {
                            "name": "lag",
                            "perturbation": {
                                "kind": "command_latency",
                                "delay_steps": 4,
                                "action_buf_len": 8,
                            },
                        }
                    ]
                )
            )


class OutcomeTests(unittest.TestCase):
    def test_success(self):
        data = _fake_data([0.0, 10.5, 0.0], time=5.0)
        self.assertEqual(
            classify_outcome(data, goal_distance_m=10.0, timeout_s=20.0),
            "success",
        )
        self.assertGreaterEqual(pos_x(data), 10.0)

    def test_fell_does_not_end_episode(self):
        fallen_height = _fake_data([-1.0, 1.0, 0.0], time=1.0)
        self.assertLess(world_height(fallen_height), FALL_HEIGHT_M)
        self.assertTrue(has_fallen(fallen_height))
        self.assertIsNone(
            classify_outcome(fallen_height, goal_distance_m=10.0, timeout_s=20.0)
        )
        fallen_pitch = _fake_data([0.0, 1.0, FALL_PITCH_RAD + 0.01], time=1.0)
        self.assertIsNone(
            classify_outcome(fallen_pitch, goal_distance_m=10.0, timeout_s=20.0)
        )

    def test_timeout_while_fallen_is_timeout(self):
        data = _fake_data([-1.0, 1.0, 0.0], time=20.0)
        self.assertTrue(has_fallen(data))
        self.assertEqual(
            classify_outcome(data, goal_distance_m=10.0, timeout_s=20.0),
            "timeout",
        )

    def test_success_after_fall_pose(self):
        data = _fake_data([-1.0, 10.5, 0.0], time=8.0)
        self.assertTrue(has_fallen(data))
        self.assertEqual(
            classify_outcome(data, goal_distance_m=10.0, timeout_s=20.0),
            "success",
        )

    def test_timeout(self):
        data = _fake_data([0.0, 1.0, 0.0], time=20.0)
        self.assertEqual(
            classify_outcome(data, goal_distance_m=10.0, timeout_s=20.0),
            "timeout",
        )

    def test_continue(self):
        data = _fake_data([0.0, 1.0, 0.0], time=5.0)
        self.assertIsNone(
            classify_outcome(data, goal_distance_m=10.0, timeout_s=20.0)
        )


class FaultTests(unittest.TestCase):
    def test_mass_and_force_are_bodyweight_fractions(self):
        mujoco, model, data = _load_walker()
        mass = walker_total_mass(model)
        gravity = walker_gravity(model)
        self.assertGreater(mass, 0.0)

        mass_fault = resolve_perturbation(
            {"kind": "mass", "mass_frac": 0.25, "body": "torso"}, model
        )
        self.assertAlmostEqual(mass_fault.delta_kg, 0.25 * mass)

        force_fault = resolve_perturbation(
            {
                "kind": "force",
                "force_frac": 0.5,
                "direction": "backward",
                "duration_s": 0.2,
            },
            model,
        )
        self.assertAlmostEqual(force_fault.force_n, 0.5 * mass * gravity)
        self.assertAlmostEqual(force_fault.force_vec[0], -force_fault.force_n)

        with self.assertRaises(ValueError):
            resolve_perturbation({"kind": "mass", "delta_kg": 5.0}, model)
        with self.assertRaises(ValueError):
            resolve_perturbation(
                {"kind": "force", "magnitude": 10.0, "direction": "forward"}, model
            )

    def test_actuator_friction_mass_obstacle_apply_and_restore(self):
        mujoco, model, data = _load_walker()
        plant = PlantFaults(model, data)
        snap = plant.snapshot

        none = resolve_perturbation({"kind": "none"}, model)
        plant.activate(none, 1.0)
        np.testing.assert_array_equal(model.actuator_gear[:, 0], snap.actuator_gear)
        np.testing.assert_array_equal(model.geom_friction, snap.geom_friction)
        plant.clear()

        hip = resolve_perturbation(
            {"kind": "actuator_dead", "actuator": "right_hip"}, model
        )
        plant.activate(hip, 1.0)
        self.assertEqual(model.actuator_gear[hip.actuator_id, 0], 0.0)
        plant.clear()
        self.assertAlmostEqual(
            model.actuator_gear[hip.actuator_id, 0],
            snap.actuator_gear[hip.actuator_id],
        )

        ice = resolve_perturbation({"kind": "friction", "mu": 0.2}, model)
        plant.activate(ice, 1.0)
        self.assertAlmostEqual(model.geom_friction[ice.floor_id, 0], 0.2)
        for geom_id in ice.friction_geom_ids:
            self.assertAlmostEqual(model.geom_friction[geom_id, 0], 0.2)
        plant.clear()
        self.assertAlmostEqual(
            model.geom_friction[ice.floor_id, 0], snap.geom_friction[ice.floor_id, 0]
        )

        backpack = resolve_perturbation(
            {"kind": "mass", "mass_frac": 0.25, "body": "torso"}, model
        )
        nominal = float(snap.body_mass[backpack.body_id])
        pack_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "backpack")
        self.assertGreaterEqual(pack_id, 0)
        self.assertAlmostEqual(float(model.geom_rgba[pack_id, 3]), 0.0)
        plant.activate(backpack, 1.0)
        self.assertAlmostEqual(
            float(model.body_mass[backpack.body_id]),
            nominal + backpack.delta_kg,
        )
        self.assertGreater(float(model.geom_rgba[pack_id, 3]), 0.9)
        plant.clear()
        self.assertAlmostEqual(float(model.body_mass[backpack.body_id]), nominal)
        np.testing.assert_allclose(model.body_inertia, snap.body_inertia)
        self.assertAlmostEqual(float(model.geom_rgba[pack_id, 3]), 0.0)

        box = resolve_perturbation(
            {"kind": "obstacle", "size": [0.25, 0.5, 0.5], "x": 4.0}, model
        )
        plant.activate(box, 3.0)
        self.assertAlmostEqual(float(_obstacle_world_pos(model, box.obstacle_id)[0]), 4.0)
        self.assertEqual(int(model.geom_contype[box.obstacle_id]), 1)
        self.assertEqual(int(model.geom_conaffinity[box.obstacle_id]), 1)
        plant.clear()
        self.assertAlmostEqual(float(_obstacle_world_pos(model, box.obstacle_id)[2]), -5.0)

    def test_command_latency_delays_ctrl_and_leaves_the_model(self):
        mujoco, model, data = _load_walker()
        plant = PlantFaults(model, data)
        snap_gear = model.actuator_gear[:, 0].copy()
        snap_mass = model.body_mass.copy()
        fault = resolve_perturbation(
            {"kind": "command_latency", "delay_steps": 2}, model
        )
        self.assertEqual(fault.delay_steps, 2)
        plant.activate(fault, 0.0)
        np.testing.assert_array_equal(model.actuator_gear[:, 0], snap_gear)
        np.testing.assert_array_equal(model.body_mass, snap_mass)

        applied = []
        for cmd in (0.2, 0.4, 0.6, 0.8):
            vector = np.full(model.nu, cmd)
            applied.append(float(plant.delay_command(vector, new_command=True)[0]))
            held = plant.delay_command(vector, new_command=False)
            self.assertAlmostEqual(float(held[0]), applied[-1])
        self.assertAlmostEqual(applied[0], 0.0)
        self.assertAlmostEqual(applied[1], 0.0)
        self.assertAlmostEqual(applied[2], 0.2)
        self.assertAlmostEqual(applied[3], 0.4)

        passthrough = CommandDelay(model.nu, 0)
        fresh = np.arange(model.nu, dtype=float)
        np.testing.assert_allclose(passthrough.push(fresh), fresh)

        plant.clear()
        self.assertIsNone(plant.command_delay)
        echoed = plant.delay_command(np.full(model.nu, 0.3), new_command=True)
        np.testing.assert_allclose(echoed, np.full(model.nu, 0.3))

    def test_ground_reaction_includes_knees_and_torso(self):
        mujoco, model, data = _load_walker()
        names = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            for geom_id in walker_ground_geom_ids(model)
        }
        self.assertIn("right_foot", names)
        self.assertIn("left_foot", names)
        self.assertIn("right_leg", names)
        self.assertIn("left_leg", names)
        self.assertIn("torso", names)
        self.assertNotIn("floor", names)
        self.assertNotIn("obstacle", names)

        for _ in range(800):
            data.ctrl[:] = 0.0
            mujoco.mj_step(model, data)
        touching = set()
        for i in range(int(data.ncon)):
            pair = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[i].geom1),
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[i].geom2),
            )
            if "floor" in pair:
                touching.add(pair[0] if pair[1] == "floor" else pair[1])
        self.assertTrue(
            touching.intersection({"torso", "right_leg", "left_leg"}),
            f"expected a non-foot ground contact, got {touching}",
        )
        force, nonfoot = sample_floor_contacts(model, data)
        self.assertTrue(set(nonfoot).intersection({"torso", "right_leg", "left_leg"}))
        self.assertFalse(set(nonfoot).intersection(FOOT_GEOM_NAMES))
        np.testing.assert_allclose(force, ground_reaction_force(model, data))
        channels = ground_reaction_channels(force)
        weight = walker_total_mass(model) * walker_gravity(model)
        self.assertAlmostEqual(channels["grf_vertical"], weight, delta=0.15 * weight)
        self.assertGreaterEqual(channels["grf_mag"], channels["grf_vertical"])
        self.assertGreaterEqual(channels["grf_mag"], channels["grf_horizontal"])

    def test_dead_actuator_zeros_torque_not_mpc_command(self):
        mujoco, model, data = _load_walker()
        plant = PlantFaults(model, data)
        hip = resolve_perturbation(
            {"kind": "actuator_dead", "actuator": "right_hip"}, model
        )
        plant.activate(hip, 0.0)
        data.ctrl[:] = 0.0
        data.ctrl[hip.actuator_id] = 0.8
        mujoco.mj_step(model, data)
        self.assertAlmostEqual(float(data.ctrl[hip.actuator_id]), 0.8)
        self.assertEqual(model.actuator_gear[hip.actuator_id, 0], 0.0)
        joint_id = int(model.actuator_trnid[hip.actuator_id, 0])
        dof = int(model.jnt_dofadr[joint_id])
        self.assertAlmostEqual(float(data.qfrc_actuator[dof]), 0.0, places=6)
        delivered = float(data.actuator_force[hip.actuator_id] * model.actuator_gear[hip.actuator_id, 0])
        self.assertAlmostEqual(delivered, 0.0, places=6)

    def test_plant_fault_does_not_change_belief_model(self):
        mujoco, plant_model, data = _load_walker()
        belief_model = mujoco.MjModel.from_xml_path(str(WALKER_XML))
        plant = PlantFaults(plant_model, data)
        hip = resolve_perturbation(
            {"kind": "actuator_dead", "actuator": "right_hip"}, plant_model
        )
        ice = resolve_perturbation({"kind": "friction", "mu": 0.2}, plant_model)
        backpack = resolve_perturbation(
            {"kind": "mass", "mass_frac": 0.25, "body": "torso"}, plant_model
        )

        belief_gear = belief_model.actuator_gear[:, 0].copy()
        belief_mu = belief_model.geom_friction.copy()
        belief_mass = belief_model.body_mass.copy()
        belief_inertia = belief_model.body_inertia.copy()

        plant.activate(hip, 1.0)
        self.assertEqual(plant_model.actuator_gear[hip.actuator_id, 0], 0.0)
        np.testing.assert_array_equal(belief_model.actuator_gear[:, 0], belief_gear)

        plant.clear()
        plant.activate(ice, 1.0)
        self.assertAlmostEqual(plant_model.geom_friction[ice.floor_id, 0], 0.2)
        np.testing.assert_array_equal(belief_model.geom_friction, belief_mu)
        for geom_id in ice.friction_geom_ids:
            self.assertAlmostEqual(plant_model.geom_friction[geom_id, 0], 0.2)
            self.assertAlmostEqual(
                float(belief_model.geom_friction[geom_id, 0]),
                float(belief_mu[geom_id, 0]),
            )
        loka_state = {
            "mutations": [],
            "nominal_gears": belief_gear,
            "nominal_friction": belief_mu,
            "nominal_mass": belief_mass,
            "nominal_inertia": belief_inertia,
        }
        assert_belief_isolated(belief_model, loka_state)
        self.assertGreater(len(plant_friction_geom_ids(plant_model, ice.floor_id)), 1)

        plant.clear()
        plant.activate(backpack, 1.0)
        np.testing.assert_array_equal(belief_model.body_mass, belief_mass)
        np.testing.assert_array_equal(belief_model.body_inertia, belief_inertia)
        pack_id = mujoco.mj_name2id(belief_model, mujoco.mjtObj.mjOBJ_GEOM, "backpack")
        self.assertAlmostEqual(float(belief_model.geom_rgba[pack_id, 3]), 0.0)
        self.assertGreater(float(plant_model.geom_rgba[pack_id, 3]), 0.9)

    def test_loka_mutations_change_belief_not_plant(self):
        mujoco, plant_model, _data = _load_walker()
        belief_model = mujoco.MjModel.from_xml_path(str(WALKER_XML))
        hip_id = mujoco.mj_name2id(plant_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_hip")
        floor_id = mujoco.mj_name2id(plant_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        plant_gear = plant_model.actuator_gear[:, 0].copy()
        plant_mu = plant_model.geom_friction.copy()

        loka_state = {
            "mutations": [
                {
                    "type": "actuator",
                    "id": hip_id,
                    "attr": "gear",
                    "val": 0.0,
                    "name": "right_hip",
                },
                {
                    "type": "geom",
                    "id": floor_id,
                    "attr": "friction",
                    "val": 0.2,
                    "name": "floor",
                },
            ],
            "nominal_gears": belief_model.actuator_gear[:, 0].copy(),
            "nominal_friction": belief_model.geom_friction.copy(),
            "nominal_mass": belief_model.body_mass.copy(),
            "nominal_inertia": belief_model.body_inertia.copy(),
        }
        apply_loka_mutations(belief_model, loka_state)

        self.assertEqual(belief_model.actuator_gear[hip_id, 0], 0.0)
        self.assertAlmostEqual(belief_model.geom_friction[floor_id, 0], 0.2)
        np.testing.assert_array_equal(plant_model.actuator_gear[:, 0], plant_gear)
        np.testing.assert_array_equal(plant_model.geom_friction, plant_mu)
        assert_belief_isolated(belief_model, loka_state)

    def test_loka_floor_friction_reaches_contact_mu(self):
        """A floor-only write must also drop the feet or contact μ stays 0.7."""
        mujoco, plant_model, data = _load_walker()
        belief_model = mujoco.MjModel.from_xml_path(str(WALKER_XML))
        floor_id = mujoco.mj_name2id(belief_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        foot_id = mujoco.mj_name2id(belief_model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot")
        contact_ids = plant_friction_geom_ids(belief_model, floor_id)
        loka_state = {
            "mutations": [
                {
                    "type": "geom",
                    "id": floor_id,
                    "attr": "friction",
                    "val": 0.0001,
                    "name": "floor",
                }
            ],
            "nominal_gears": belief_model.actuator_gear[:, 0].copy(),
            "nominal_friction": belief_model.geom_friction.copy(),
            "nominal_mass": belief_model.body_mass.copy(),
            "nominal_inertia": belief_model.body_inertia.copy(),
            "floor_geom_id": floor_id,
            "friction_contact_ids": contact_ids,
        }
        apply_loka_mutations(belief_model, loka_state)
        self.assertAlmostEqual(float(belief_model.geom_friction[floor_id, 0]), 0.0001)
        self.assertAlmostEqual(float(belief_model.geom_friction[foot_id, 0]), 0.0001)
        for gid in contact_ids:
            self.assertAlmostEqual(float(belief_model.geom_friction[gid, 0]), 0.0001)
        np.testing.assert_array_equal(
            plant_model.geom_friction, loka_state["nominal_friction"]
        )
        for _ in range(40):
            mujoco.mj_step(plant_model, data)
        belief_data = mujoco.MjData(belief_model)
        belief_data.qpos[:] = data.qpos
        belief_data.qvel[:] = 0
        mujoco.mj_forward(belief_model, belief_data)
        self.assertGreater(int(belief_data.ncon), 0)
        for i in range(belief_data.ncon):
            self.assertAlmostEqual(float(belief_data.contact[i].friction[0]), 0.0001)
        assert_belief_isolated(belief_model, loka_state)

    def test_loka_mass_mutation_is_belief_only(self):
        mujoco, plant_model, _data = _load_walker()
        belief_model = mujoco.MjModel.from_xml_path(str(WALKER_XML))
        torso_id = mujoco.mj_name2id(belief_model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        plant_mass = plant_model.body_mass.copy()
        nominal = float(belief_model.body_mass[torso_id])
        loka_state = {
            "mutations": [
                {
                    "type": "body",
                    "id": torso_id,
                    "attr": "mass",
                    "val": nominal + 4.0,
                    "name": "torso",
                }
            ],
            "nominal_gears": belief_model.actuator_gear[:, 0].copy(),
            "nominal_friction": belief_model.geom_friction.copy(),
            "nominal_mass": belief_model.body_mass.copy(),
            "nominal_inertia": belief_model.body_inertia.copy(),
        }
        apply_loka_mutations(belief_model, loka_state)
        self.assertAlmostEqual(float(belief_model.body_mass[torso_id]), nominal + 4.0)
        np.testing.assert_array_equal(plant_model.body_mass, plant_mass)
        assert_belief_isolated(belief_model, loka_state)

    def test_loka_com_mutation_is_belief_only(self):
        mujoco, plant_model, _data = _load_walker()
        belief_model = mujoco.MjModel.from_xml_path(str(WALKER_XML))
        torso_id = mujoco.mj_name2id(belief_model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        plant_ipos = plant_model.body_ipos.copy()
        shifted = [-0.08, 0.0, 0.02]
        loka_state = {
            "mutations": [
                {
                    "type": "body",
                    "id": torso_id,
                    "attr": "com",
                    "val": shifted,
                    "name": "torso",
                }
            ],
            "nominal_gears": belief_model.actuator_gear[:, 0].copy(),
            "nominal_friction": belief_model.geom_friction.copy(),
            "nominal_mass": belief_model.body_mass.copy(),
            "nominal_inertia": belief_model.body_inertia.copy(),
            "nominal_ipos": belief_model.body_ipos.copy(),
        }
        apply_loka_mutations(belief_model, loka_state)
        np.testing.assert_allclose(belief_model.body_ipos[torso_id], shifted, atol=1e-9)
        np.testing.assert_array_equal(plant_model.body_ipos, plant_ipos)
        assert_belief_isolated(belief_model, loka_state)

    def test_leaked_ice_on_belief_is_rejected(self):
        mujoco, belief_model, _data = _load_walker()
        floor_id = mujoco.mj_name2id(belief_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        loka_state = {
            "mutations": [],
            "nominal_gears": belief_model.actuator_gear[:, 0].copy(),
            "nominal_friction": belief_model.geom_friction.copy(),
            "nominal_mass": belief_model.body_mass.copy(),
            "nominal_inertia": belief_model.body_inertia.copy(),
        }
        assert_belief_isolated(belief_model, loka_state)
        belief_model.geom_friction[floor_id, 0] = 0.2
        with self.assertRaises(RuntimeError):
            assert_belief_isolated(belief_model, loka_state)

    def test_leaked_backpack_mass_on_belief_is_rejected(self):
        mujoco, belief_model, _data = _load_walker()
        torso_id = mujoco.mj_name2id(belief_model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        loka_state = {
            "mutations": [],
            "nominal_gears": belief_model.actuator_gear[:, 0].copy(),
            "nominal_friction": belief_model.geom_friction.copy(),
            "nominal_mass": belief_model.body_mass.copy(),
            "nominal_inertia": belief_model.body_inertia.copy(),
        }
        assert_belief_isolated(belief_model, loka_state)
        belief_model.body_mass[torso_id] += 5.0
        with self.assertRaises(RuntimeError):
            assert_belief_isolated(belief_model, loka_state)

    def test_leaked_com_on_belief_is_rejected(self):
        mujoco, belief_model, _data = _load_walker()
        torso_id = mujoco.mj_name2id(belief_model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        loka_state = {
            "mutations": [],
            "nominal_gears": belief_model.actuator_gear[:, 0].copy(),
            "nominal_friction": belief_model.geom_friction.copy(),
            "nominal_mass": belief_model.body_mass.copy(),
            "nominal_inertia": belief_model.body_inertia.copy(),
            "nominal_ipos": belief_model.body_ipos.copy(),
        }
        assert_belief_isolated(belief_model, loka_state)
        belief_model.body_ipos[torso_id, 0] -= 0.05
        with self.assertRaises(RuntimeError):
            assert_belief_isolated(belief_model, loka_state)

    def test_sanitize_belief_hides_backpack_and_disables_box(self):
        mujoco, model, _data = _load_walker()
        pack = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "backpack")
        box = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle")
        model.geom_rgba[pack, 3] = 1.0
        model.geom_contype[box] = 1
        model.geom_conaffinity[box] = 1
        sanitize_belief_worldview(model)
        self.assertAlmostEqual(float(model.geom_rgba[pack, 3]), 0.0)
        self.assertEqual(int(model.geom_contype[box]), 0)
        self.assertEqual(int(model.geom_conaffinity[box]), 0)

    def test_force_xfrc_and_duration(self):
        mujoco, model, data = _load_walker()
        plant = PlantFaults(model, data)
        shove = resolve_perturbation(
            {
                "kind": "force",
                "force_frac": 0.5,
                "direction": "forward",
                "duration_s": 0.2,
            },
            model,
        )
        plant.activate(shove, 1.0)
        plant.apply_physics(1.05)
        self.assertAlmostEqual(data.xfrc_applied[shove.body_id, 0], shove.force_n)
        plant.apply_physics(1.25)
        self.assertAlmostEqual(data.xfrc_applied[shove.body_id, 0], 0.0)

    def test_backpack_does_not_teleport_qpos(self):
        """mj_setConst on live MjData used to write qpos0 every backpack step."""
        mujoco, model, data = _load_walker()
        plant = PlantFaults(model, data)
        data.qpos[1] = 0.5
        data.qvel[1] = 1.2
        mujoco.mj_forward(model, data)
        x0 = float(data.qpos[1])
        backpack = resolve_perturbation(
            {"kind": "mass", "mass_frac": 0.25, "body": "torso"}, model
        )
        plant.activate(backpack, 0.0)
        self.assertAlmostEqual(float(data.qpos[1]), x0)
        self.assertAlmostEqual(float(data.qvel[1]), 1.2)
        bid = backpack.body_id
        old = float(plant.snapshot.body_mass[bid])
        new = old + backpack.delta_kg
        self.assertAlmostEqual(float(model.body_mass[bid]), new)
        np.testing.assert_allclose(
            model.body_inertia[bid],
            plant.snapshot.body_inertia[bid] * (new / old),
        )
        self.assertAlmostEqual(
            float(model.body_subtreemass[bid]),
            float(plant.snapshot.body_mass[1:].sum()) + backpack.delta_kg,
            places=5,
        )
        for _ in range(20):
            plant.apply_physics(float(data.time))
            mujoco.mj_step(model, data)
        self.assertGreater(float(data.qpos[1]), x0 + 0.01)
        self.assertTrue(np.isfinite(data.qpos).all())

    def test_obstacle_blocks_the_walker(self):
        """A 1 m wall on its own body must stop a walker shoved into it."""
        mujoco, model, data = _load_walker()
        plant = PlantFaults(model, data)
        box = resolve_perturbation(
            {"kind": "obstacle", "size": [0.25, 0.5, 0.5], "x": 4.0}, model
        )
        plant.activate(box, 0.0)
        data.qpos[:] = 0.0
        data.qpos[1] = 3.4
        data.qvel[1] = 3.0
        hits = 0
        for _ in range(250):
            plant.apply_physics(float(data.time))
            mujoco.mj_step(model, data)
            for i in range(int(data.ncon)):
                names = (
                    mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[i].geom1
                    ),
                    mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, data.contact[i].geom2
                    ),
                )
                if "obstacle" in names:
                    hits += 1
        self.assertGreater(hits, 0, "walker should be in contact with the box")
        self.assertLess(float(data.qpos[1]), 4.05, "walker walked through the wall")

        belief = mujoco.MjModel.from_xml_path(str(WALKER_XML))
        self.assertAlmostEqual(
            float(_obstacle_world_pos(belief, box.obstacle_id)[2]), -5.0
        )
        self.assertFalse(
            np.allclose(
                _obstacle_world_pos(belief, box.obstacle_id),
                _obstacle_world_pos(model, box.obstacle_id),
            )
        )

    def test_ice_lowers_contact_friction(self):
        mujoco, model, data = _load_walker()
        plant = PlantFaults(model, data)
        for _ in range(40):
            mujoco.mj_step(model, data)
        self.assertGreater(int(data.ncon), 0)
        before = [float(data.contact[i].friction[0]) for i in range(data.ncon)]
        self.assertTrue(all(abs(mu - 0.7) < 1e-6 for mu in before))
        for mu in (0.2, 0.0001):
            ice = resolve_perturbation({"kind": "friction", "mu": mu}, model)
            plant.activate(ice, float(data.time))
            mujoco.mj_forward(model, data)
            after = [float(data.contact[i].friction[0]) for i in range(data.ncon)]
            self.assertTrue(after, "ice should still have foot/floor contacts")
            self.assertTrue(
                all(abs(cmu - mu) < 1e-9 for cmu in after),
                f"contact μ {after} != {mu}",
            )
            foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot")
            self.assertAlmostEqual(float(model.geom_friction[ice.floor_id, 0]), mu)
            self.assertAlmostEqual(float(model.geom_friction[foot_id, 0]), mu)
            plant.clear()
            mujoco.mj_forward(model, data)


class ResultsWriterTests(unittest.TestCase):
    def test_writes_folder_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "ice__fixed_mpc"
            logger = EpisodeLogger(
                directory,
                log_hz=50.0,
                record=False,
                record_fps=30.0,
                record_width=64,
                record_height=48,
                record_camera="side_follow",
                model_path=str(WALKER_XML),
            )
            logger.rows.append(
                {
                    "t": 0.0,
                    "pitch": 0.0,
                    "height": 1.3,
                    "pos_x": 0.0,
                    "vel_x": 0.0,
                    "fault_active": 0,
                    "tracking_error": 0.0,
                    "in_failure": 0,
                }
            )
            logger.write(
                metadata={"test": "ice", "baseline": "fixed_mpc", "outcome": "timeout"},
                mpc_snapshots=[{"time": 0.0, "cost_weights": {"Height": 10.0}}],
                loka_turns=[],
            )
            self.assertTrue((directory / "metadata.json").is_file())
            self.assertTrue((directory / "timeseries.csv").is_file())
            self.assertTrue((directory / "timeseries.npz").is_file())
            self.assertTrue((directory / "mpc_params.jsonl").is_file())
            self.assertTrue((directory / "loka_turns.jsonl").is_file())
            self.assertFalse((directory / "episode.mp4").exists())


def _has_mjpc():
    try:
        from mujoco_mpc import agent as _mpc_agent  # noqa: F401

        return True
    except Exception:
        return False


@unittest.skipUnless(_has_mjpc(), "mujoco_mpc not installed")
class WalkerRuntimeBeliefIsolationTests(unittest.TestCase):
    def test_backpack_does_not_enter_planner(self):
        import mujoco

        from loka.walker_runtime import WalkerRuntime

        runtime = WalkerRuntime(enable_llm=False, speed_goal=1.0)
        try:
            runtime.reset()
            torso = mujoco.mj_name2id(runtime.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
            pack = mujoco.mj_name2id(
                runtime.belief_model, mujoco.mjtObj.mjOBJ_GEOM, "backpack"
            )
            belief_mass = runtime.belief_model.body_mass.copy()
            belief_inertia = runtime.belief_model.body_inertia.copy()
            agent_mass = runtime.agent.model.body_mass.copy()
            rgba = runtime.belief_model.geom_rgba[pack].copy()

            runtime.activate_fault(
                {"kind": "mass", "mass_frac": 0.25, "body": "torso"}, 0.0
            )
            for _ in range(8):
                runtime.step()

            np.testing.assert_array_equal(runtime.belief_model.body_mass, belief_mass)
            np.testing.assert_array_equal(
                runtime.belief_model.body_inertia, belief_inertia
            )
            np.testing.assert_array_equal(runtime.agent.model.body_mass, agent_mass)
            np.testing.assert_allclose(runtime.belief_model.geom_rgba[pack], rgba)
            self.assertAlmostEqual(float(runtime.belief_model.geom_rgba[pack, 3]), 0.0)
            self.assertGreater(
                float(runtime.model.body_mass[torso]), float(belief_mass[torso]) + 1.0
            )
            self.assertGreater(float(runtime.model.geom_rgba[pack, 3]), 0.9)
            runtime.assert_mpc_unaware_of_plant()
        finally:
            runtime.close()


class DefaultYamlRoundTrip(unittest.TestCase):
    def test_default_file_is_valid_yaml(self):
        path = (
            Path(__file__).resolve().parent.parent
            / "loka"
            / "config"
            / "walker_suite.yaml"
        )
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        parse_suite_dict(raw)


if __name__ == "__main__":
    unittest.main()
