"""Metrics, seeded reset noise, and suite charts. No MJPC or LLM."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from loka.recording import prepare_offscreen_env
from loka.walker_suite.plots import plot_run, rolling_rms
from loka.walker_suite.statistics import (
    HEIGHT_GOAL_M,
    aggregate_condition,
    metrics_from_log,
    summarize_run,
    time_not_fallen,
    time_to_recovery,
    touched_nonfoot_geoms,
    trajectory_rms,
    write_statistics,
)
from loka.walker_suite.stochastic import apply_seeded_state_noise, trial_seed

WALKER_XML = Path(__file__).resolve().parent.parent / "models" / "walker" / "task.xml"


class SeededNoiseTests(unittest.TestCase):
    def test_trial_seed_is_shared_offset(self):
        self.assertEqual(trial_seed(10, 0), 10)
        self.assertEqual(trial_seed(10, 4), 14)
        with self.assertRaises(ValueError):
            trial_seed(0, -1)

    def test_same_seed_matches_and_other_seed_diverges(self):
        import mujoco

        def rollout(seed: int, scale: float, n_steps: int = 30):
            model = mujoco.MjModel.from_xml_path(str(WALKER_XML))
            data = mujoco.MjData(model)
            mujoco.mj_resetData(model, data)
            apply_seeded_state_noise(model, data, seed, scale)
            qpos0 = data.qpos.copy()
            data.ctrl[:] = 0.2
            xs = []
            for _ in range(n_steps):
                mujoco.mj_step(model, data)
                xs.append(float(data.qpos[1]))
            return qpos0, np.asarray(xs)

        qpos_a, path_a = rollout(0, 0.005)
        qpos_b, path_b = rollout(0, 0.005)
        qpos_c, path_c = rollout(1, 0.005)
        np.testing.assert_allclose(qpos_a, qpos_b)
        np.testing.assert_allclose(path_a, path_b)
        self.assertFalse(np.allclose(qpos_a, qpos_c))
        self.assertFalse(np.allclose(path_a, path_c))

        model = mujoco.MjModel.from_xml_path(str(WALKER_XML))
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        nominal = data.qpos.copy()
        apply_seeded_state_noise(model, data, 0, 0.0)
        np.testing.assert_allclose(data.qpos, nominal)
        self.assertTrue(np.allclose(data.qvel, 0.0))

    def test_noise_scale_is_the_half_width(self):
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(WALKER_XML))
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        apply_seeded_state_noise(model, data, 2, 0.005)
        delta = np.abs(data.qpos - model.qpos0)
        self.assertLessEqual(float(delta.max()), 0.005 + 1e-12)
        self.assertGreater(float(delta.max()), 0.0)


class MetricTests(unittest.TestCase):
    def test_trajectory_rms_splits_height_speed_and_combined(self):
        height = np.full(4, HEIGHT_GOAL_M + 0.1)
        speed = np.full(4, 1.0)
        pitch = np.zeros(4)
        rms = trajectory_rms(height, speed, pitch, speed_goal=1.0)
        self.assertAlmostEqual(rms["rms_height_m"], 0.1)
        self.assertAlmostEqual(rms["rms_pitch_rad"], 0.0)
        self.assertAlmostEqual(rms["rms_speed_mps"], 0.0)
        self.assertAlmostEqual(rms["rms_tracking"], 0.1)

        speed = np.full(4, 1.3)
        pitch = np.full(4, 0.2)
        rms = trajectory_rms(height, speed, pitch, speed_goal=1.0)
        self.assertAlmostEqual(rms["rms_speed_mps"], 0.3)
        self.assertAlmostEqual(rms["rms_pitch_rad"], 0.2)
        self.assertAlmostEqual(rms["rms_tracking"], float(np.sqrt(0.1**2 + 0.2**2 + 0.3**2)))

    def test_time_to_recovery_uses_the_longest_fail_interval(self):
        times = [1.0, 2.0, 2.05, 4.0, 5.0, 5.5]
        failed = [True, False, True, True, False, False]
        self.assertAlmostEqual(time_to_recovery(times, failed), 4.0)
        self.assertEqual(time_to_recovery([0.0, 1.0, 2.0], [False, False, False]), 0.0)
        self.assertAlmostEqual(
            time_to_recovery([1.0, 4.0], [True, True]),
            3.0,
        )

    def test_startup_second_does_not_count_as_failure(self):
        self.assertEqual(
            time_to_recovery([0.0, 0.5, 0.99, 1.2], [True, True, True, False]),
            0.0,
        )
        self.assertEqual(
            time_to_recovery([0.0, 0.4, 2.0, 3.0], [True, True, False, False]),
            0.0,
        )
        self.assertAlmostEqual(
            time_to_recovery([0.0, 0.5, 1.0, 2.5, 3.0], [True, True, True, False, False]),
            1.5,
        )

    def test_metrics_from_log_and_condition_averages(self):
        rows = [
            {"t": 0.0, "height": HEIGHT_GOAL_M, "vel_x": 1.0, "pitch": 0.0, "in_failure": 0, "fallen": 0},
            {"t": 1.0, "height": HEIGHT_GOAL_M, "vel_x": 1.0, "pitch": 0.0, "in_failure": 1, "fallen": 0},
            {"t": 3.0, "height": HEIGHT_GOAL_M, "vel_x": 1.0, "pitch": 0.0, "in_failure": 0, "fallen": 0},
            {"t": 3.5, "height": HEIGHT_GOAL_M, "vel_x": 1.0, "pitch": 0.0, "in_failure": 0, "fallen": 0},
        ]
        metrics = metrics_from_log(rows, speed_goal=1.0)
        self.assertAlmostEqual(metrics["rms_height_m"], 0.0)
        self.assertAlmostEqual(metrics["time_to_recovery_s"], 2.0)
        self.assertAlmostEqual(metrics["time_not_fallen_s"], 3.5)

        trials = [
            {"test": "ice", "baseline": "loka", "outcome": "success", "t_end": 10.0, **metrics},
            {
                "test": "ice",
                "baseline": "loka",
                "outcome": "timeout",
                "t_end": 20.0,
                "rms_height_m": 0.2,
                "rms_speed_mps": 0.4,
                "rms_tracking": 0.5,
                "time_to_recovery_s": 4.0,
            },
        ]
        summary = aggregate_condition(trials)
        self.assertEqual(summary["n_success"], 1)
        self.assertAlmostEqual(summary["success_rate"], 0.5)
        self.assertAlmostEqual(summary["avg_completion_time_s"], 10.0)
        self.assertAlmostEqual(summary["avg_time_to_recovery_s"], 3.0)
        self.assertAlmostEqual(summary["avg_rms_height_m"], 0.1)

    def test_time_not_fallen_stops_at_the_fall_and_keeps_the_tail(self):
        times = [0.0, 1.0, 2.0, 3.0]
        fallen = [0.0, 0.0, 1.0, 1.0]
        self.assertAlmostEqual(time_not_fallen(times, fallen, t_end=4.0), 2.0)
        self.assertAlmostEqual(time_not_fallen([0.0, 1.0], [0.0, 0.0], t_end=1.5), 1.5)
        self.assertAlmostEqual(time_not_fallen([0.2, 1.0], [0.0, 0.0], t_end=1.0), 1.0)

    def test_rolling_rms_window(self):
        times = np.array([0.0, 0.1, 0.2, 0.3])
        values = np.array([0.0, 0.0, 3.0, 0.0])
        rms = rolling_rms(values, times, window_s=0.25)
        self.assertAlmostEqual(rms[2], np.sqrt(3.0))
        self.assertAlmostEqual(rms[3], np.sqrt(3.0))


class PlotTests(unittest.TestCase):
    def test_writes_bar_and_line_charts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            results = []
            for baseline, height_offset in (("loka", 0.0), ("fixed_mpc", 0.05)):
                for trial in range(2):
                    directory = run / f"ice__{baseline}" / f"trial_{trial:02d}"
                    directory.mkdir(parents=True)
                    times = np.linspace(0.0, 1.0, 40)
                    height = np.full_like(times, HEIGHT_GOAL_M + height_offset)
                    speed = np.full_like(times, 1.0 if baseline == "loka" else 0.7)
                    failed = np.zeros_like(times)
                    if baseline == "fixed_mpc":
                        failed[10:25] = 1.0
                    grf = np.full_like(times, 400.0 if baseline == "fixed_mpc" else 280.0)
                    grf_h = np.full_like(times, 80.0 if baseline == "fixed_mpc" else 20.0)
                    nonfoot = np.ones_like(times) if baseline == "fixed_mpc" else np.zeros_like(times)
                    knee = np.array(
                        [1.0 if baseline == "fixed_mpc" and i >= 10 else 0.0 for i in range(times.size)]
                    )
                    np.savez_compressed(
                        directory / "timeseries.npz",
                        t=times,
                        height=height,
                        vel_x=speed,
                        in_failure=failed,
                        fallen=np.zeros_like(times),
                        grf_mag=grf,
                        grf_horizontal=grf_h,
                        nonfoot_floor=nonfoot,
                        floor_right_leg=knee,
                    )
                    rows = [
                        {
                            "t": float(t),
                            "height": float(h),
                            "vel_x": float(v),
                            "in_failure": int(f),
                            "fallen": 0,
                            "grf_mag": float(g),
                            "grf_horizontal": float(gh),
                            "nonfoot_floor": float(nf),
                            "floor_right_leg": float(k),
                        }
                        for t, h, v, f, g, gh, nf, k in zip(
                            times, height, speed, failed, grf, grf_h, nonfoot, knee
                        )
                    ]
                    t_end = 8.0 if baseline == "loka" else 20.0
                    metrics = metrics_from_log(rows, speed_goal=1.0, t_end=t_end)
                    results.append(
                        {
                            "test": "ice",
                            "baseline": baseline,
                            "trial": trial,
                            "outcome": "success" if baseline == "loka" else "timeout",
                            "t_end": t_end,
                            "speed_goal": 1.0,
                            "episode_dir": str(directory),
                            **metrics,
                        }
                    )
            stats = summarize_run(results)
            write_statistics(run, stats)
            paths = plot_run(run, results, stats)
            self.assertTrue((run / "statistics.json").is_file())
            self.assertTrue((run / "statistics.csv").is_file())
            self.assertEqual(
                {path.name for path in paths},
                {
                    "ice_metrics.png",
                    "ice_rms_over_time.png",
                    "nonfoot_floor_contacts.png",
                },
            )
            for path in paths:
                self.assertGreater(path.stat().st_size, 1000)
            by_baseline = {
                row["baseline"]: row for row in stats["conditions"] if row["test"] == "ice"
            }
            self.assertAlmostEqual(by_baseline["loka"]["avg_time_not_fallen_s"], 8.0)
            self.assertAlmostEqual(by_baseline["fixed_mpc"]["avg_time_not_fallen_s"], 20.0)
            self.assertAlmostEqual(by_baseline["fixed_mpc"]["avg_grf_mag_n"], 400.0)
            self.assertAlmostEqual(by_baseline["loka"]["avg_grf_mag_n"], 280.0)
            self.assertAlmostEqual(by_baseline["fixed_mpc"]["avg_grf_horizontal_n"], 80.0)
            self.assertGreater(
                by_baseline["fixed_mpc"]["avg_grf_mag_n"],
                by_baseline["loka"]["avg_grf_mag_n"],
            )
            self.assertAlmostEqual(by_baseline["fixed_mpc"]["avg_nonfoot_contact_fraction"], 1.0)
            self.assertAlmostEqual(by_baseline["loka"]["avg_nonfoot_contact_fraction"], 0.0)
            self.assertEqual(
                touched_nonfoot_geoms(
                    [
                        {"floor_right_leg": 0, "floor_torso": 0},
                        {"floor_right_leg": 1, "floor_torso": 0},
                        {"floor_right_leg": 1, "floor_torso": 1},
                    ]
                ),
                ["right_leg", "torso"],
            )


class OffscreenEnvTests(unittest.TestCase):
    def test_headless_node_uses_osmesa(self):
        saved = {key: os.environ.get(key) for key in ("MUJOCO_GL", "DISPLAY", "WAYLAND_DISPLAY", "MPLCONFIGDIR", "HOME")}
        try:
            os.environ.pop("MUJOCO_GL", None)
            os.environ.pop("DISPLAY", None)
            os.environ.pop("WAYLAND_DISPLAY", None)
            os.environ.pop("MPLCONFIGDIR", None)
            os.environ["HOME"] = tempfile.gettempdir()
            backend = prepare_offscreen_env()
            self.assertEqual(backend, "osmesa")
            self.assertEqual(os.environ["MUJOCO_GL"], "osmesa")
            self.assertTrue(os.path.isdir(os.environ["MPLCONFIGDIR"]))
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
