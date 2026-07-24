"""Unit tests for ErrorSpec evaluation (no MuJoCo MPC required)."""

import unittest
from types import SimpleNamespace

import numpy as np

from loka.error_spec import (
    default_walker_error_spec,
    get_tracking_error,
    parse_error_tracking,
)


def _fake_data(qpos, qvel):
    return SimpleNamespace(qpos=np.asarray(qpos, dtype=float), qvel=np.asarray(qvel, dtype=float))


class ErrorSpecTests(unittest.TestCase):
    def test_default_walker_nominal_is_zero(self):
        # height = 1.3 + 0 = 1.3 (> 1.05), pitch=0, speed=1.0 (>= 0.75)
        data = _fake_data([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        self.assertEqual(get_tracking_error(data, default_walker_error_spec()), 0.0)

    def test_default_walker_height_and_speed_errors(self):
        # height = 1.3 - 0.4 = 0.9 → excess below 1.05 is 0.15 → *2 = 0.3
        # speed = 0.5 → excess below 0.75 is 0.25
        data = _fake_data([-0.4, 0.0, 0.0], [0.0, 0.5, 0.0])
        err = get_tracking_error(data, default_walker_error_spec())
        self.assertAlmostEqual(err, 0.3 + 0.25)

    def test_parse_and_reject_bad_index(self):
        raw = {
            "trigger_threshold": 0.2,
            "terms": [
                {
                    "name": "angle",
                    "signal": "qpos",
                    "index": 0,
                    "mode": "abs_above",
                    "target": 0.0,
                    "tolerance": 0.1,
                    "weight": 1.0,
                }
            ],
        }
        spec = parse_error_tracking(raw, nq=2, nv=2)
        self.assertEqual(spec.trigger_threshold, 0.2)
        with self.assertRaises(ValueError):
            parse_error_tracking(
                {
                    "trigger_threshold": 0.2,
                    "terms": [
                        {
                            "name": "bad",
                            "signal": "qpos",
                            "index": 99,
                            "mode": "abs_above",
                            "target": 0.0,
                            "tolerance": 0.1,
                        }
                    ],
                },
                nq=2,
                nv=2,
            )


if __name__ == "__main__":
    unittest.main()
