"""Nominal MJCF snapshot lookups used in CURRENT MODEL BELIEF."""

from __future__ import annotations

import unittest
from pathlib import Path

from loka.robot_context import (
    capture_nominal_params,
    format_current_model_belief,
    format_live_model_parameters,
    _lookup_nominal,
)

WALKER_XML = Path(__file__).resolve().parent.parent / "models" / "walker" / "task.xml"


def _load_walker():
    import mujoco

    return mujoco, mujoco.MjModel.from_xml_path(str(WALKER_XML))


class NominalLookupTests(unittest.TestCase):
    def setUp(self):
        _mujoco, self.model = _load_walker()
        self.nominal = capture_nominal_params(self.model)

    def test_snapshot_has_actuator_geom_and_body_buckets(self):
        self.assertIn("actuators", self.nominal)
        self.assertIn("geoms", self.nominal)
        self.assertIn("bodies", self.nominal)
        self.assertNotIn("bodys", self.nominal)
        self.assertIn("right_hip", self.nominal["actuators"])
        self.assertIn("floor", self.nominal["geoms"])
        self.assertIn("torso", self.nominal["bodies"])
        self.assertGreater(self.nominal["bodies"]["torso"]["mass"], 1.0)
        self.assertEqual(len(self.nominal["bodies"]["torso"]["com"]), 3)

    def test_actuator_gear_lookup(self):
        value = _lookup_nominal(
            self.nominal,
            {"type": "actuator", "name": "right_hip", "attr": "gear"},
        )
        self.assertAlmostEqual(value, 200.0)

    def test_geom_friction_lookup(self):
        value = _lookup_nominal(
            self.nominal,
            {"type": "geom", "name": "floor", "attr": "friction"},
        )
        self.assertEqual(value, [0.7, 0.1, 0.1])

    def test_body_mass_lookup_is_not_unknown(self):
        value = _lookup_nominal(
            self.nominal,
            {"type": "body", "name": "torso", "attr": "mass"},
        )
        self.assertNotEqual(value, "unknown")
        self.assertAlmostEqual(value, float(self.nominal["bodies"]["torso"]["mass"]))
        self.assertGreater(value, 5.0)

    def test_body_com_lookup_is_not_unknown(self):
        value = _lookup_nominal(
            self.nominal,
            {"type": "body", "name": "torso", "attr": "com"},
        )
        self.assertNotEqual(value, "unknown")
        self.assertEqual(value, self.nominal["bodies"]["torso"]["com"])
        self.assertEqual(len(value), 3)

    def test_lookup_is_case_insensitive(self):
        value = _lookup_nominal(
            self.nominal,
            {"type": "Body", "name": "Torso", "attr": "Mass"},
        )
        self.assertAlmostEqual(value, float(self.nominal["bodies"]["torso"]["mass"]))

    def test_belief_text_prints_nominal_mass(self):
        torso_mass = float(self.nominal["bodies"]["torso"]["mass"])
        text = format_current_model_belief(
            {
                "nominal_params": self.nominal,
                "mutations": [
                    {
                        "type": "body",
                        "name": "torso",
                        "attr": "mass",
                        "val": 1.35,
                        "applied_at": 6.3,
                    }
                ],
            }
        )
        self.assertIn("body 'torso'.mass: 1.35", text)
        self.assertIn(f"nominal: {torso_mass}", text)
        self.assertNotIn("nominal: unknown", text)

    def test_live_parameters_mark_edited_mass(self):
        import mujoco

        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        nominal_mass = float(self.model.body_mass[torso_id])
        self.model.body_mass[torso_id] = 1.35
        text = format_live_model_parameters(
            self.model, {"nominal_params": self.nominal}
        )
        self.assertIn("Body mass (kg):", text)
        self.assertIn("torso: 1.35", text)
        self.assertIn(f"nominal {nominal_mass:.4g}", text)
        self.assertIn("[edited]", text)
        self.assertIn("right_hip:", text)
        self.assertIn("floor:", text)
        hip_line = next(
            line for line in text.splitlines() if line.startswith("  - right_hip:")
        )
        self.assertNotIn("[edited]", hip_line)

    def test_live_parameters_mark_edited_com(self):
        import mujoco

        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        nominal_com = self.model.body_ipos[torso_id].tolist()
        self.model.body_ipos[torso_id] = [-0.08, 0.0, 0.02]
        text = format_live_model_parameters(
            self.model, {"nominal_params": self.nominal}
        )
        self.assertIn("Body COM / inertial pos", text)
        com_line = next(
            line for line in text.splitlines() if line.startswith("  - torso:") and "[" in line
        )
        self.assertIn("[edited]", com_line)
        self.assertIn("-0.08", com_line)
        self.assertIn(
            f"nominal [{nominal_com[0]:.4g}, {nominal_com[1]:.4g}, {nominal_com[2]:.4g}]",
            com_line,
        )


if __name__ == "__main__":
    unittest.main()
