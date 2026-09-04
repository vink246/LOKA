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
    PlantFaults,
    resolve_perturbation,
    walker_gravity,
    walker_total_mass,
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
        self.assertIn("dead_right_hip", names)
        self.assertIn("box", names)
        backpack = next(t for t in config.tests if t.name == "backpack")
        self.assertEqual(backpack.perturbation["mass_frac"], 0.25)
        self.assertNotIn("delta_kg", backpack.perturbation)


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
        plant.clear()
        self.assertAlmostEqual(
            model.geom_friction[ice.floor_id, 0], snap.geom_friction[ice.floor_id, 0]
        )

        backpack = resolve_perturbation(
            {"kind": "mass", "mass_frac": 0.25, "body": "torso"}, model
        )
        nominal = float(snap.body_mass[backpack.body_id])
        plant.activate(backpack, 1.0)
        self.assertAlmostEqual(
            float(model.body_mass[backpack.body_id]),
            nominal + backpack.delta_kg,
        )
        plant.clear()
        self.assertAlmostEqual(float(model.body_mass[backpack.body_id]), nominal)

        box = resolve_perturbation(
            {"kind": "obstacle", "size": [0.15, 0.4, 0.1], "x": 4.0}, model
        )
        plant.activate(box, 3.0)
        self.assertAlmostEqual(model.geom_pos[box.obstacle_id, 0], 4.0)
        self.assertEqual(int(model.geom_conaffinity[box.obstacle_id]), 1)
        plant.clear()
        self.assertEqual(int(model.geom_conaffinity[box.obstacle_id]), 0)

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

        plant.activate(hip, 1.0)
        self.assertEqual(plant_model.actuator_gear[hip.actuator_id, 0], 0.0)
        np.testing.assert_array_equal(belief_model.actuator_gear[:, 0], belief_gear)

        plant.clear()
        plant.activate(ice, 1.0)
        self.assertAlmostEqual(plant_model.geom_friction[ice.floor_id, 0], 0.2)
        np.testing.assert_array_equal(belief_model.geom_friction, belief_mu)

        plant.clear()
        plant.activate(backpack, 1.0)
        np.testing.assert_array_equal(belief_model.body_mass, belief_mass)

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
        }
        apply_loka_mutations(belief_model, loka_state)

        self.assertEqual(belief_model.actuator_gear[hip_id, 0], 0.0)
        self.assertAlmostEqual(belief_model.geom_friction[floor_id, 0], 0.2)
        np.testing.assert_array_equal(plant_model.actuator_gear[:, 0], plant_gear)
        np.testing.assert_array_equal(plant_model.geom_friction, plant_mu)

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
