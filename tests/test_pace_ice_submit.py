"""Slurm script rendering and job accounting for the PACE-ICE submitter."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from loka.submit_pace_ice import first_wave, main, missing_jobs, prepare_run, worker_job


def _suite(path: Path, *, trials: int = 2) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "output_dir": str(path.parent / "out"),
                "seed": 0,
                "num_trials": trials,
                "init_noise": 0.0,
                "baselines": ["low_level", "loka"],
                "tasks": [
                    {
                        "name": "goto_goal",
                        "type": "goto_goal",
                        "goal": {"xy": [1.0, 0.0], "speed": 0.2, "radius": 0.25},
                        "timeout_s": 5.0,
                        "perturbations": [
                            {"name": "nominal", "kind": "none"},
                            {"name": "ice", "kind": "friction", "mu": 0.3},
                        ],
                        "metrics": ["success"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_jobs_are_stable_and_worker_script_is_cpu_only(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("loka.submit_pace_ice._require_openai", lambda *_a, **_k: None)
    config = tmp_path / "suite.yaml"
    _suite(config)
    run_dir = tmp_path / "run"
    code = main(
        [
            "--config",
            str(config),
            "--run-dir",
            str(run_dir),
            "--dry-run",
            "--conda-prefix",
            str(tmp_path),
            "--max-in-flight",
            "10",
        ]
    )
    assert code == 0
    manifest = json.loads((run_dir / "slurm" / "jobs.json").read_text(encoding="utf-8"))
    ids = [job["id"] for job in manifest["jobs"]]
    # task, perturbation, baseline, trial.
    assert ids == [
        "goto_goal__nominal__low_level__trial_00",
        "goto_goal__nominal__low_level__trial_01",
        "goto_goal__nominal__loka__trial_00",
        "goto_goal__nominal__loka__trial_01",
        "goto_goal__ice__low_level__trial_00",
        "goto_goal__ice__low_level__trial_01",
        "goto_goal__ice__loka__trial_00",
        "goto_goal__ice__loka__trial_01",
    ]
    worker = (run_dir / "slurm" / "worker.sh").read_text(encoding="utf-8")
    assert "#SBATCH --array=0-7%10" in worker
    assert "export MUJOCO_GL=egl" in worker
    assert "--gres" not in worker
    assert "OPENAI" not in worker
    assert 'expected="$(cd' in worker
    assert str(tmp_path) in worker
    merge = (run_dir / "slurm" / "merge.sh").read_text(encoding="utf-8")
    assert "--gres" not in merge
    assert "submit_pace_ice --merge" in merge


def test_sbatch_merge_depends_on_the_array(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("loka.submit_pace_ice._require_openai", lambda *_a, **_k: None)
    config = tmp_path / "suite.yaml"
    _suite(config, trials=1)
    calls: list[tuple] = []

    def fake_sbatch(script, extra=None):
        calls.append((Path(script).name, extra))
        return "4242"

    monkeypatch.setattr("loka.submit_pace_ice._sbatch", fake_sbatch)
    code = main(["--config", str(config), "--run-dir", str(tmp_path / "run")])
    assert code == 0
    assert calls[0][0] == "worker.sh"
    assert calls[1][0] == "merge.sh"
    assert calls[1][1] == ["--dependency", "afterany:4242"]


def test_worker_selects_the_array_index(tmp_path: Path, monkeypatch):
    config = tmp_path / "suite.yaml"
    _suite(config, trials=1)
    from loka.harness.config import load_suite_yaml

    suite = load_suite_yaml(config)
    run_dir = tmp_path / "run"
    _worker, _merge, jobs = prepare_run(
        suite,
        run_dir,
        repo=tmp_path,
        conda_env="loka",
        conda_prefix=None,
        cpus=2,
        mem="8G",
        time_limit="02:00:00",
        max_in_flight=10,
        partition=None,
        account=None,
        qos=None,
        constraint=None,
        mujoco_gl=None,
    )
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    monkeypatch.setenv("LOKA_JOBS_FILE", str(run_dir / "slurm" / "jobs.json"))
    assert worker_job(run_dir)["id"] == jobs[1]["id"]

    captured: dict = {}

    def fake_run(suite_obj):
        captured["unit"] = suite_obj.unit
        return run_dir

    monkeypatch.setattr("loka.harness.pipeline.run_suite", fake_run)
    from loka.submit_pace_ice import run_worker

    assert run_worker(run_dir) == 0
    assert captured["unit"] == jobs[1]["id"]


def test_merge_and_resubmit_report_missing_units(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("loka.submit_pace_ice._require_openai", lambda *_a, **_k: None)
    config = tmp_path / "suite.yaml"
    _suite(config, trials=1)
    run_dir = tmp_path / "run"
    code = main(["--config", str(config), "--run-dir", str(run_dir), "--dry-run"])
    assert code == 0
    jobs = json.loads((run_dir / "slurm" / "jobs.json").read_text(encoding="utf-8"))["jobs"]
    finished = run_dir / jobs[0]["id"]
    finished.mkdir(parents=True)
    (finished / "done.json").write_text("{}", encoding="utf-8")
    missing = missing_jobs(run_dir)
    assert jobs[0]["id"] not in {job["id"] for job in missing}
    assert len(missing) == len(jobs) - 1

    code = main(["--merge", str(run_dir)])
    assert code == 1

    monkeypatch.setattr("loka.submit_pace_ice._sbatch", lambda script, extra=None: "9")
    code = main(
        ["--resubmit-missing", str(run_dir), "--dry-run", "--conda-prefix", str(tmp_path)]
    )
    assert code == 0
    resubmit = json.loads((run_dir / "slurm" / "resubmit_jobs.json").read_text(encoding="utf-8"))
    assert len(resubmit["jobs"]) == len(jobs) - 1
    script = (run_dir / "slurm" / "resubmit.sh").read_text(encoding="utf-8")
    assert f"#SBATCH --array=0-{len(jobs) - 2}%10" in script
    assert "resubmit_jobs.json" in script
    assert "--gres" not in script


def test_loka_baseline_requires_an_api_key(tmp_path: Path, monkeypatch):
    config = tmp_path / "suite.yaml"
    _suite(config, trials=1)
    monkeypatch.setattr("loka.submit_pace_ice.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = main(["--config", str(config), "--run-dir", str(tmp_path / "run"), "--dry-run"])
    assert code == 2
    os.environ["OPENAI_API_KEY"] = "test-key"
    try:
        code = main(
            ["--config", str(config), "--run-dir", str(tmp_path / "run2"), "--dry-run"]
        )
    finally:
        os.environ.pop("OPENAI_API_KEY", None)
    assert code == 0
    script = (tmp_path / "run2" / "slurm" / "worker.sh").read_text(encoding="utf-8")
    assert "test-key" not in script


def test_first_wave_stays_within_the_pace_cap():
    # 70 units, cap 50: 49 array tasks plus the chaining job.
    assert first_wave(70, 50, self_counts=False) == (49, "chain")
    # The chain job is itself still queued, so the next wave is smaller.
    assert first_wave(21, 50, self_counts=True) == (21, "merge")
    assert first_wave(80, 50, self_counts=True) == (48, "chain")
    # A suite that fits is one array plus the merge job.
    assert first_wave(8, 50, self_counts=False) == (8, "merge")


def test_over_cap_submits_a_wave_then_the_rest(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("loka.submit_pace_ice._require_openai", lambda *_a, **_k: None)
    config = tmp_path / "suite.yaml"
    _suite(config, trials=2)
    run_dir = tmp_path / "run"
    code = main(
        [
            "--config",
            str(config),
            "--run-dir",
            str(run_dir),
            "--dry-run",
            "--max-submitted",
            "5",
        ]
    )
    assert code == 0
    worker = (run_dir / "slurm" / "worker.sh").read_text(encoding="utf-8")
    assert "#SBATCH --array=0-3%10" in worker
    assert "LOKA_JOB_OFFSET=0" in worker
    chain = (run_dir / "slurm" / "chain.sh").read_text(encoding="utf-8")
    assert "--advance" in chain
    assert "--gres" not in chain
    manifest = json.loads((run_dir / "slurm" / "jobs.json").read_text(encoding="utf-8"))
    assert manifest["next_offset"] == 0
    assert len(manifest["jobs"]) == 8

    calls: list[tuple] = []

    def fake_sbatch(script, extra=None):
        calls.append((Path(script).name, extra))
        return str(100 + len(calls))

    monkeypatch.setattr("loka.submit_pace_ice._sbatch", fake_sbatch)
    code = main(["--config", str(config), "--run-dir", str(run_dir), "--max-submitted", "5"])
    assert code == 0
    assert calls[0][0] == "worker.sh"
    assert calls[1] == ("chain.sh", ["--dependency", "afterany:101"])
    manifest = json.loads((run_dir / "slurm" / "jobs.json").read_text(encoding="utf-8"))
    assert manifest["next_offset"] == 4
    assert manifest["pending_wave"] is None

    code = main(["--advance", "--run-dir", str(run_dir)])
    assert code == 0
    assert calls[2][0] == "wave_0004.sh"
    assert calls[3][1] == ["--dependency", "afterany:103"]
    wave = (run_dir / "slurm" / "wave_0004.sh").read_text(encoding="utf-8")
    assert "#SBATCH --array=0-2%10" in wave
    assert "LOKA_JOB_OFFSET=4" in wave
    manifest = json.loads((run_dir / "slurm" / "jobs.json").read_text(encoding="utf-8"))
    assert manifest["next_offset"] == 7

    code = main(["--advance", "--run-dir", str(run_dir)])
    assert code == 0
    assert calls[4][0] == "wave_0007.sh"
    assert calls[5] == ("merge.sh", ["--dependency", "afterany:105"])
    manifest = json.loads((run_dir / "slurm" / "jobs.json").read_text(encoding="utf-8"))
    assert manifest["next_offset"] == 8
