#!/usr/bin/env python3
"""Queue the Walker suite on PACE-ICE as one CPU job per baseline trial.

Run this on a PACE-ICE login node. It submits a Slurm array with
``--array=0-N%10`` so at most 10 jobs are in flight. Each array task is one
trial of one baseline and runs every perturbation in the suite config. A
follow-up job merges the partial logs and writes the plots.

Jobs request CPUs only. They do not set ``--gres`` and do not ask for a GPU.

    bash scripts/pace_ice_walker_suite.sh --num-trials 5
    bash scripts/pace_ice_walker_suite.sh --num-trials 5 --cpus 32 \\
        --conda-prefix /home/hice1/vkulkarni46/scratch/envs/loka
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from loka.run_walker_suite import build_parser as build_suite_parser
from loka.walker_suite.config import SuiteConfig, load_suite_config, load_suite_yaml
from loka.walker_suite.pipeline import merge_partials, trial_indices


def iter_jobs(suite: SuiteConfig) -> list[dict[str, int | str]]:
    """One job per baseline and trial. Each job still runs every configured test."""
    jobs: list[dict[str, int | str]] = []
    for baseline in suite.baselines:
        for trial in trial_indices(suite):
            jobs.append({"baseline": baseline, "trial": int(trial)})
    return jobs


def _sbatch_headers(
    *,
    name: str,
    cpus: int,
    mem: str,
    time: str,
    output: str,
    error: str,
    array: str | None = None,
    partition: str | None = None,
    account: str | None = None,
    qos: str | None = None,
    constraint: str | None = None,
) -> list[str]:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={name}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={int(cpus)}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time}",
        f"#SBATCH --output={output}",
        f"#SBATCH --error={error}",
    ]
    if array is not None:
        lines.append(f"#SBATCH --array={array}")
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    if account:
        lines.append(f"#SBATCH --account={account}")
    if qos:
        lines.append(f"#SBATCH --qos={qos}")
    if constraint:
        lines.append(f"#SBATCH --constraint={constraint}")
    return lines


def _preamble(repo: Path, conda_env: str, conda_prefix: str | None = None) -> list[str]:
    """Activate the PACE conda env and refuse any other interpreter.

    ``conda_prefix`` is a directory (``conda activate /path/to/env``). A named
    env is used only when no prefix is given. Prefix installs set
    ``CONDA_DEFAULT_ENV`` to the folder name, so the check compares
    ``CONDA_PREFIX`` to that directory.
    """
    repo_q = shlex.quote(str(repo))
    target = conda_prefix if conda_prefix else conda_env
    target_q = shlex.quote(target)
    lines = [
        "set -euo pipefail",
        "if command -v module >/dev/null 2>&1; then",
        "  module load anaconda3 >/dev/null 2>&1 || module load anaconda >/dev/null 2>&1 || true",
        "fi",
        "if ! command -v conda >/dev/null 2>&1; then",
        f'  echo "conda is not on PATH; cannot activate {target_q}" >&2',
        "  exit 1",
        "fi",
        'source "$(conda info --base)/etc/profile.d/conda.sh"',
        f"conda activate {target_q}",
    ]
    if conda_prefix:
        lines += [
            f'expected="$(cd {target_q} && pwd)"',
            'actual="$(cd "$CONDA_PREFIX" && pwd)"',
            'if [ "$actual" != "$expected" ]; then',
            '  echo "expected conda prefix $expected, got ${CONDA_PREFIX:-none}" >&2',
            "  exit 1",
            "fi",
        ]
    else:
        lines += [
            f'if [ "${{CONDA_DEFAULT_ENV:-}}" != {target_q} ]; then',
            f'  echo "expected conda env {target_q}, got ${{CONDA_DEFAULT_ENV:-none}}" >&2',
            "  exit 1",
            "fi",
        ]
    lines += [
        f"cd {repo_q}",
        "export OMP_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
    ]
    return lines


def render_worker_script(
    *,
    run_dir: Path,
    repo: Path,
    conda_env: str,
    conda_prefix: str | None = None,
    n_jobs: int,
    cpus: int,
    mem: str,
    time: str,
    max_in_flight: int,
    partition: str | None = None,
    account: str | None = None,
    qos: str | None = None,
    constraint: str | None = None,
) -> str:
    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1")
    if max_in_flight < 1:
        raise ValueError("max_in_flight must be >= 1")
    if cpus < 1:
        raise ValueError("cpus must be >= 1")
    run_q = shlex.quote(str(run_dir))
    lines = _sbatch_headers(
        name="loka-walker",
        cpus=cpus,
        mem=mem,
        time=time,
        output=f"{run_dir}/slurm/walker_%A_%a.out",
        error=f"{run_dir}/slurm/walker_%A_%a.err",
        array=f"0-{n_jobs - 1}%{int(max_in_flight)}",
        partition=partition,
        account=account,
        qos=qos,
        constraint=constraint,
    )
    lines += [
        "# CPU only. This script does not request a GPU.",
        *_preamble(repo, conda_env, conda_prefix),
        f'"$CONDA_PREFIX/bin/python" -m loka.submit_pace_ice --worker --run-dir {run_q}',
        "",
    ]
    return "\n".join(lines)


def render_merge_script(
    *,
    run_dir: Path,
    repo: Path,
    conda_env: str,
    conda_prefix: str | None = None,
    partition: str | None = None,
    account: str | None = None,
    qos: str | None = None,
    constraint: str | None = None,
) -> str:
    run_q = shlex.quote(str(run_dir))
    lines = _sbatch_headers(
        name="loka-walker-merge",
        cpus=1,
        mem="4G",
        time="00:30:00",
        output=f"{run_dir}/slurm/merge_%j.out",
        error=f"{run_dir}/slurm/merge_%j.err",
        partition=partition,
        account=account,
        qos=qos,
        constraint=constraint,
    )
    lines += [
        "# CPU only. This script does not request a GPU.",
        *_preamble(repo, conda_env, conda_prefix),
        f'"$CONDA_PREFIX/bin/python" -m loka.submit_pace_ice --merge {run_q}',
        "",
    ]
    return "\n".join(lines)


def _sbatch(script: Path, extra: list[str] | None = None) -> str:
    command = ["sbatch", *(extra or []), str(script)]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr or completed.stdout)
        raise RuntimeError(f"sbatch failed ({completed.returncode}): {' '.join(command)}")
    text = (completed.stdout or "").strip()
    print(text)
    marker = "Submitted batch job "
    if marker not in text:
        raise RuntimeError(f"Could not read a job id from sbatch output: {text}")
    return text.split(marker, 1)[1].split()[0].strip()


def prepare_run(suite: SuiteConfig, run_dir: Path, *, repo: Path, conda_env: str, conda_prefix: str | None, cpus: int, mem: str, time: str, max_in_flight: int, partition: str | None, account: str | None, qos: str | None, constraint: str | None) -> tuple[Path, Path, list[dict]]:
    from loka.walker_suite.pipeline import _write_resolved_config

    jobs = iter_jobs(suite)
    if not jobs:
        raise ValueError("No jobs to submit")
    run_dir = Path(run_dir)
    slurm_dir = run_dir / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(run_dir, replace(suite, run_dir=run_dir))
    (slurm_dir / "jobs.json").write_text(
        json.dumps({"jobs": jobs, "num_trials": suite.num_trials}, indent=2),
        encoding="utf-8",
    )
    worker = slurm_dir / "worker.sh"
    worker.write_text(
        render_worker_script(
            run_dir=run_dir,
            repo=repo,
            conda_env=conda_env,
            conda_prefix=conda_prefix,
            n_jobs=len(jobs),
            cpus=cpus,
            mem=mem,
            time=time,
            max_in_flight=max_in_flight,
            partition=partition,
            account=account,
            qos=qos,
            constraint=constraint,
        ),
        encoding="utf-8",
    )
    merge = slurm_dir / "merge.sh"
    merge.write_text(
        render_merge_script(
            run_dir=run_dir,
            repo=repo,
            conda_env=conda_env,
            conda_prefix=conda_prefix,
            partition=partition,
            account=account,
            qos=qos,
            constraint=constraint,
        ),
        encoding="utf-8",
    )
    return worker, merge, jobs


def run_worker(run_dir: Path) -> int:
    from loka.walker_suite.pipeline import run_pipeline

    run_dir = Path(run_dir)
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task is None:
        print("SLURM_ARRAY_TASK_ID is not set", file=sys.stderr)
        return 2
    manifest = json.loads((run_dir / "slurm" / "jobs.json").read_text(encoding="utf-8"))
    job = manifest["jobs"][int(task)]
    suite = load_suite_yaml(run_dir / "config.resolved.yaml")
    suite.baselines = [str(job["baseline"])]
    suite.trial = int(job["trial"])
    suite.run_dir = run_dir
    if suite.num_trials <= suite.trial:
        suite.num_trials = suite.trial + 1
    run_pipeline(suite)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = build_suite_parser()
    parser.description = (
        "Submit the Walker suite to PACE-ICE. One CPU job per baseline trial, "
        "at most 10 in flight."
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=32,
        help="CPUs for each trial job (default: 32). Slurm affinity is what MJPC uses.",
    )
    parser.add_argument("--mem", default="16G", help="Memory for each trial job (default: 16G).")
    parser.add_argument(
        "--time",
        default="00:20:00",
        help="Wall time for each trial job (default: 00:20:00).",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=10,
        help="Maximum concurrent trial jobs (default: 10).",
    )
    parser.add_argument("--partition", default=None, help="Slurm partition. Leave unset to use the cluster default.")
    parser.add_argument("--account", default=None, help="Slurm account, if your cluster requires one.")
    parser.add_argument("--qos", default=None, help="Slurm QoS, if your cluster requires one.")
    parser.add_argument(
        "--constraint",
        default=None,
        help="Slurm constraint. Find feature names with: sinfo -o '%P %c %f'",
    )
    parser.add_argument(
        "--conda-env",
        default="loka",
        help="Named conda env activated on the compute node (default: loka). Ignored when --conda-prefix is set.",
    )
    parser.add_argument(
        "--conda-prefix",
        default=None,
        help="Path of the conda env to activate, for example ~/scratch/envs/loka. Overrides --conda-env.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the Slurm scripts and job list, and do not call sbatch.",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Run the Slurm array task for this node. Requires SLURM_ARRAY_TASK_ID and --run-dir.",
    )
    parser.add_argument(
        "--merge",
        default=None,
        help="Merge partials in this run directory and write plots. Skips submission.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker:
        if not args.run_dir:
            print("--worker requires --run-dir", file=sys.stderr)
            return 2
        return run_worker(Path(args.run_dir))
    if args.merge:
        missing = merge_partials(Path(args.merge))
        return 1 if missing else 0

    try:
        suite = load_suite_config(args)
    except Exception as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2
    if args.cpus < 1 or args.max_in_flight < 1:
        print("--cpus and --max-in-flight must be >= 1", file=sys.stderr)
        return 2
    conda_prefix = None
    if args.conda_prefix:
        prefix = Path(args.conda_prefix).expanduser()
        if not prefix.is_dir():
            print(f"--conda-prefix is not a directory: {prefix}", file=sys.stderr)
            return 2
        conda_prefix = str(prefix)

    repo = Path(__file__).resolve().parent.parent
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(suite.output_dir) / f"pace_{stamp}"
    try:
        worker, merge, jobs = prepare_run(
            suite,
            run_dir,
            repo=repo,
            conda_env=args.conda_env,
            conda_prefix=conda_prefix,
            cpus=args.cpus,
            mem=args.mem,
            time=args.time,
            max_in_flight=args.max_in_flight,
            partition=args.partition,
            account=args.account,
            qos=args.qos,
            constraint=args.constraint,
        )
    except Exception as exc:
        print(f"Submit error: {exc}", file=sys.stderr)
        return 2

    print(f"[pace] {len(jobs)} jobs, at most {args.max_in_flight} in flight, {args.cpus} CPUs each")
    print(f"[pace] run directory {run_dir}")
    if args.dry_run:
        print(f"[pace] dry run, scripts at {worker} and {merge}")
        return 0
    try:
        array_id = _sbatch(worker)
        _sbatch(merge, ["--dependency", f"afterany:{array_id}"])
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
