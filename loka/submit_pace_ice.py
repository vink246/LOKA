#!/usr/bin/env python3
"""Queue the G1 evaluation suite on PACE-ICE.

Run this on a login node. One CPU Slurm array task is one work unit
(task x perturbation x baseline x trial). At most ``--max-in-flight`` tasks
run at once. A follow-up job recomputes metrics and writes the plots.

Jobs do not request a GPU.

    bash scripts/pace_ice_eval.sh --num-trials 5
    bash scripts/pace_ice_eval.sh --num-trials 5 --cpus 2 \\
        --conda-prefix /home/hice1/vkulkarni46/scratch/envs/loka
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from loka.harness.cli import add_suite_arguments
from loka.harness.config import SuiteConfig, load_suite_config
from loka.harness.pipeline import recompute, write_resolved_config
from loka.harness.units import iter_units


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Submit the G1 eval suite to PACE-ICE. One CPU job per work unit, "
            "at most 10 in flight."
        )
    )
    add_suite_arguments(parser)
    parser.add_argument("--cpus", type=int, default=2, help="CPUs per unit (default: 2).")
    parser.add_argument("--mem", default="8G", help="Memory per unit (default: 8G).")
    parser.add_argument(
        "--time",
        default="02:00:00",
        help="Wall time per unit (default: 02:00:00).",
    )
    parser.add_argument("--max-in-flight", type=int, default=10)
    parser.add_argument("--partition", default=None)
    parser.add_argument("--account", default=None)
    parser.add_argument("--qos", default=None)
    parser.add_argument("--constraint", default=None)
    parser.add_argument("--conda-env", default="loka")
    parser.add_argument(
        "--conda-prefix",
        default=None,
        help="Conda env directory to activate, for example ~/scratch/envs/loka.",
    )
    parser.add_argument(
        "--mujoco-gl",
        default=None,
        help="MUJOCO_GL for the job (egl or osmesa). Unset keeps the cluster default.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--merge", default=None, help="Recompute this run directory.")
    parser.add_argument(
        "--resubmit-missing",
        default=None,
        help="Submit a new array for units in this run that have no done.json.",
    )
    return parser


def _sbatch_headers(
    *,
    name: str,
    cpus: int,
    mem: str,
    time_limit: str,
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
        f"#SBATCH --time={time_limit}",
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


def _preamble(
    repo: Path,
    conda_env: str,
    conda_prefix: str | None,
    mujoco_gl: str | None,
    jobs_file: Path,
) -> list[str]:
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
        f"export LOKA_JOBS_FILE={shlex.quote(str(jobs_file))}",
    ]
    if mujoco_gl:
        lines.append(f"export MUJOCO_GL={shlex.quote(mujoco_gl)}")
    return lines


def render_worker_script(
    *,
    run_dir: Path,
    repo: Path,
    conda_env: str,
    conda_prefix: str | None,
    n_jobs: int,
    cpus: int,
    mem: str,
    time_limit: str,
    max_in_flight: int,
    jobs_file: Path,
    partition: str | None = None,
    account: str | None = None,
    qos: str | None = None,
    constraint: str | None = None,
    mujoco_gl: str | None = None,
    label: str = "loka-eval",
) -> str:
    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1")
    if max_in_flight < 1:
        raise ValueError("max_in_flight must be >= 1")
    if cpus < 1:
        raise ValueError("cpus must be >= 1")
    run_q = shlex.quote(str(run_dir))
    lines = _sbatch_headers(
        name=label,
        cpus=cpus,
        mem=mem,
        time_limit=time_limit,
        output=f"{run_dir}/slurm/eval_%A_%a.out",
        error=f"{run_dir}/slurm/eval_%A_%a.err",
        array=f"0-{n_jobs - 1}%{int(max_in_flight)}",
        partition=partition,
        account=account,
        qos=qos,
        constraint=constraint,
    )
    lines += [
        "# CPU only. This script does not request a GPU.",
        *_preamble(repo, conda_env, conda_prefix, mujoco_gl, jobs_file),
        f'"$CONDA_PREFIX/bin/python" -m loka.submit_pace_ice --worker --run-dir {run_q}',
        "",
    ]
    return "\n".join(lines)


def render_merge_script(
    *,
    run_dir: Path,
    repo: Path,
    conda_env: str,
    conda_prefix: str | None,
    partition: str | None = None,
    account: str | None = None,
    qos: str | None = None,
    constraint: str | None = None,
    mujoco_gl: str | None = None,
) -> str:
    run_q = shlex.quote(str(run_dir))
    lines = _sbatch_headers(
        name="loka-eval-merge",
        cpus=1,
        mem="4G",
        time_limit="00:30:00",
        output=f"{run_dir}/slurm/merge_%j.out",
        error=f"{run_dir}/slurm/merge_%j.err",
        partition=partition,
        account=account,
        qos=qos,
        constraint=constraint,
    )
    lines += [
        "# CPU only. This script does not request a GPU.",
        *_preamble(repo, conda_env, conda_prefix, mujoco_gl, run_dir / "slurm" / "jobs.json"),
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


def _require_openai(suite: SuiteConfig, repo: Path) -> None:
    if not any(item.type == "loka" for item in suite.baselines):
        return
    load_dotenv(repo / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "baseline 'loka' needs OPENAI_API_KEY in the environment or in .env. "
            "The key is read on the compute node; it is not written into the Slurm script."
        )


def prepare_run(
    suite: SuiteConfig,
    run_dir: Path,
    *,
    repo: Path,
    conda_env: str,
    conda_prefix: str | None,
    cpus: int,
    mem: str,
    time_limit: str,
    max_in_flight: int,
    partition: str | None,
    account: str | None,
    qos: str | None,
    constraint: str | None,
    mujoco_gl: str | None,
    jobs: list[dict] | None = None,
    jobs_name: str = "jobs.json",
    worker_name: str = "worker.sh",
) -> tuple[Path, Path, list[dict]]:
    run_dir = Path(run_dir)
    slurm_dir = run_dir / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(run_dir, suite)
    if jobs is None:
        jobs = [unit.to_dict() for unit in iter_units(suite)]
    if not jobs:
        raise ValueError("No jobs to submit")
    jobs_path = slurm_dir / jobs_name
    jobs_path.write_text(
        json.dumps({"jobs": jobs, "num_trials": suite.num_trials}, indent=2),
        encoding="utf-8",
    )
    worker = slurm_dir / worker_name
    worker.write_text(
        render_worker_script(
            run_dir=run_dir,
            repo=repo,
            conda_env=conda_env,
            conda_prefix=conda_prefix,
            n_jobs=len(jobs),
            cpus=cpus,
            mem=mem,
            time_limit=time_limit,
            max_in_flight=max_in_flight,
            jobs_file=jobs_path,
            partition=partition,
            account=account,
            qos=qos,
            constraint=constraint,
            mujoco_gl=mujoco_gl,
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
            mujoco_gl=mujoco_gl,
        ),
        encoding="utf-8",
    )
    return worker, merge, jobs


def missing_jobs(run_dir: Path) -> list[dict]:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "slurm" / "jobs.json").read_text(encoding="utf-8"))
    missing = []
    for job in manifest.get("jobs") or []:
        if not (run_dir / str(job["id"]) / "done.json").is_file():
            missing.append(job)
    return missing


def worker_job(run_dir: Path) -> dict:
    """The manifest entry for this Slurm array task."""
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task is None:
        raise RuntimeError("SLURM_ARRAY_TASK_ID is not set")
    jobs_file = Path(os.environ.get("LOKA_JOBS_FILE", Path(run_dir) / "slurm" / "jobs.json"))
    manifest = json.loads(jobs_file.read_text(encoding="utf-8"))
    return manifest["jobs"][int(task)]


def run_worker(run_dir: Path) -> int:
    from loka.harness.pipeline import run_suite

    run_dir = Path(run_dir)
    job = worker_job(run_dir)
    suite = load_suite_config(_namespace(config=str(run_dir / "config.resolved.yaml")))
    suite.unit = str(job["id"])
    suite.run_dir = run_dir
    run_suite(suite)
    return 0


def _namespace(**kwargs):
    return argparse.Namespace(**kwargs)


def _slurm_kwargs(args) -> dict:
    return {
        "conda_env": args.conda_env,
        "conda_prefix": args.conda_prefix,
        "cpus": args.cpus,
        "mem": args.mem,
        "time_limit": args.time,
        "max_in_flight": args.max_in_flight,
        "partition": args.partition,
        "account": args.account,
        "qos": args.qos,
        "constraint": args.constraint,
        "mujoco_gl": args.mujoco_gl,
    }


def _submit_pair(worker: Path, merge: Path) -> None:
    array_id = _sbatch(worker)
    _sbatch(merge, ["--dependency", f"afterany:{array_id}"])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker:
        if not args.run_dir:
            print("--worker requires --run-dir", file=sys.stderr)
            return 2
        try:
            return run_worker(Path(args.run_dir))
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args.merge:
        missing = recompute(Path(args.merge))
        return 1 if missing else 0

    repo = Path(__file__).resolve().parent.parent
    if args.resubmit_missing:
        return _resubmit(args, repo)

    try:
        suite = load_suite_config(args)
    except Exception as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2
    if args.cpus < 1 or args.max_in_flight < 1:
        print("--cpus and --max-in-flight must be >= 1", file=sys.stderr)
        return 2
    conda_prefix = _check_prefix(args.conda_prefix)
    if conda_prefix is None and args.conda_prefix:
        return 2
    args.conda_prefix = conda_prefix
    try:
        _require_openai(suite, repo)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    from datetime import datetime

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(suite.output_dir) / f"pace_{stamp}"
    suite.run_dir = run_dir
    try:
        worker, merge, jobs = prepare_run(suite, run_dir, repo=repo, **_slurm_kwargs(args))
    except Exception as exc:
        print(f"Submit error: {exc}", file=sys.stderr)
        return 2
    print(
        f"[pace] {len(jobs)} jobs, at most {args.max_in_flight} in flight, "
        f"{args.cpus} CPUs each"
    )
    print(f"[pace] run directory {run_dir}")
    if args.dry_run:
        print(f"[pace] dry run, scripts at {worker} and {merge}")
        return 0
    try:
        _submit_pair(worker, merge)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(run_dir)
    return 0


def _resubmit(args, repo: Path) -> int:
    run_dir = Path(args.resubmit_missing)
    jobs = missing_jobs(run_dir)
    if not jobs:
        print(f"[pace] nothing missing in {run_dir}")
        return 0
    suite = load_suite_config(_namespace(config=str(run_dir / "config.resolved.yaml")))
    suite.run_dir = run_dir
    conda_prefix = _check_prefix(args.conda_prefix)
    if conda_prefix is None and args.conda_prefix:
        return 2
    args.conda_prefix = conda_prefix
    try:
        _require_openai(suite, repo)
        worker, merge, jobs = prepare_run(
            suite,
            run_dir,
            repo=repo,
            jobs=jobs,
            jobs_name="resubmit_jobs.json",
            worker_name="resubmit.sh",
            **_slurm_kwargs(args),
        )
    except Exception as exc:
        print(f"Submit error: {exc}", file=sys.stderr)
        return 2
    print(f"[pace] resubmitting {len(jobs)} unfinished units")
    if args.dry_run:
        print(f"[pace] dry run, scripts at {worker} and {merge}")
        return 0
    try:
        _submit_pair(worker, merge)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


def _check_prefix(text: str | None) -> str | None:
    if not text:
        return None
    prefix = Path(text).expanduser()
    if not prefix.is_dir():
        print(f"--conda-prefix is not a directory: {prefix}", file=sys.stderr)
        return None
    return str(prefix)


if __name__ == "__main__":
    raise SystemExit(main())
