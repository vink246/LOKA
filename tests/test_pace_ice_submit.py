"""PACE-ICE submission layout. No Slurm and no episode rollout."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from loka.submit_pace_ice import iter_jobs, main, render_worker_script
from loka.walker_suite.config import parse_suite_dict
from loka.walker_suite.pipeline import merge_partials, trial_indices


def _suite(**overrides):
    raw = {
        "output_dir": "results/walker",
        "num_trials": 2,
        "baselines": ["loka", "fixed_mpc", "dr_rl"],
        "defaults": {"goal_distance_m": 10, "perturbation_time_s": 0, "timeout_s": 20, "speed_goal": 1},
        "tests": [
            {"name": "nominal", "perturbation": {"kind": "none"}},
            {"name": "ice", "perturbation": {"kind": "friction", "mu": 0.001}},
        ],
    }
    raw.update(overrides)
    return parse_suite_dict(raw)


class PaceIceSubmitTests(unittest.TestCase):
    def test_one_job_per_baseline_trial(self):
        jobs = iter_jobs(_suite())
        self.assertEqual(len(jobs), 6)
        self.assertEqual(jobs[0], {"baseline": "loka", "trial": 0})
        self.assertEqual(jobs[1], {"baseline": "loka", "trial": 1})
        self.assertEqual(jobs[-1], {"baseline": "dr_rl", "trial": 1})

    def test_trial_index_selects_one(self):
        suite = _suite()
        suite.trial = 1
        self.assertEqual(trial_indices(suite), [1])
        self.assertEqual(iter_jobs(suite), [
            {"baseline": "loka", "trial": 1},
            {"baseline": "fixed_mpc", "trial": 1},
            {"baseline": "dr_rl", "trial": 1},
        ])
        suite.trial = 2
        with self.assertRaises(ValueError):
            trial_indices(suite)

    def test_worker_script_is_cpu_only_and_throttled(self):
        script = render_worker_script(
            run_dir=Path("/tmp/pace_run"),
            repo=Path("/tmp/LOKA"),
            conda_env="loka",
            n_jobs=6,
            cpus=32,
            mem="16G",
            time="02:00:00",
            max_in_flight=10,
        )
        headers = [line.lower() for line in script.splitlines() if line.startswith("#SBATCH")]
        blob = "\n".join(headers)
        self.assertIn("--array=0-5%10", script)
        self.assertIn("--cpus-per-task=32", blob)
        self.assertNotIn("gpu", blob)
        self.assertNotIn("gres", blob)
        self.assertIn("conda activate loka", script)
        self.assertIn('"$CONDA_PREFIX/bin/python"', script)
        self.assertNotIn("mjpc", script)

    def test_dry_run_writes_scripts_without_sbatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "suite.yaml"
            config.write_text(
                "output_dir: " + str(root / "out") + "\n"
                "num_trials: 2\n"
                "baselines: [loka, fixed_mpc]\n"
                "defaults: {goal_distance_m: 10, perturbation_time_s: 0, timeout_s: 20, speed_goal: 1}\n"
                "tests:\n"
                "  - {name: nominal, perturbation: {kind: none}}\n",
                encoding="utf-8",
            )
            run_dir = root / "out" / "pace_test"
            code = main([
                "--config", str(config),
                "--run-dir", str(run_dir),
                "--dry-run",
                "--cpus", "32",
                "--max-in-flight", "10",
            ])
            self.assertEqual(code, 0)
            script = (run_dir / "slurm" / "worker.sh").read_text(encoding="utf-8")
            self.assertIn("--array=0-3%10", script)
            self.assertIn("--time=00:05:00", script)
            jobs = json.loads((run_dir / "slurm" / "jobs.json").read_text(encoding="utf-8"))
            self.assertEqual(len(jobs["jobs"]), 4)
            self.assertTrue((run_dir / "slurm" / "merge.sh").is_file())

    def test_merge_partials_orders_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "partials").mkdir()
            (run / "slurm").mkdir()
            (run / "config.resolved.yaml").write_text(
                "num_trials: 2\n"
                "baselines: [loka, fixed_mpc]\n"
                "defaults: {goal_distance_m: 10, perturbation_time_s: 0, timeout_s: 20, speed_goal: 1}\n"
                "tests:\n"
                "  - {name: nominal, perturbation: {kind: none}}\n",
                encoding="utf-8",
            )
            (run / "slurm" / "jobs.json").write_text(
                json.dumps({
                    "jobs": [
                        {"baseline": "loka", "trial": 0},
                        {"baseline": "loka", "trial": 1},
                        {"baseline": "fixed_mpc", "trial": 0},
                    ]
                }),
                encoding="utf-8",
            )
            (run / "partials" / "fixed_mpc__trial_00.json").write_text(
                json.dumps([{
                    "test": "nominal",
                    "baseline": "fixed_mpc",
                    "trial": 0,
                    "outcome": "success",
                    "t_end": 10.0,
                }]),
                encoding="utf-8",
            )
            (run / "partials" / "loka__trial_00.json").write_text(
                json.dumps([{
                    "test": "nominal",
                    "baseline": "loka",
                    "trial": 0,
                    "outcome": "success",
                    "t_end": 11.0,
                }]),
                encoding="utf-8",
            )
            missing = merge_partials(run)
            summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [(row["baseline"], row["trial"]) for row in summary],
                [("loka", 0), ("fixed_mpc", 0)],
            )
            self.assertEqual(missing, ["loka trial 1"])
            self.assertTrue((run / "plots" / "nominal_metrics.png").is_file())


if __name__ == "__main__":
    unittest.main()
