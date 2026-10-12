#!/usr/bin/env python3
"""Queue the G1 evaluation suite on PACE-ICE.

Run this on a login node. One CPU Slurm array task is one work unit
(task x perturbation x baseline x trial). At most ``--max-in-flight`` tasks
run at once. A follow-up job recomputes metrics and writes the plots.

PACE counts each array task as a submitted job, and a user can have about 50
in the queue. ``--max-submitted`` (default 50) keeps a wave under that cap,
including its one follow-up job. When the suite is larger, that follow-up
submits the next wave only after the previous tasks have left the queue.

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

# PACE rejects a submission once a user has about this many jobs queued.
# Each element of a job array counts as one job.
DEFAULT_MAX_SUBMITTED = 25


def first_wave(n_remaining: int, max_submitted: int, *, self_counts: bool) -> tuple[int, str]:
    """How many array tasks to submit now, and which follow-up job to attach.

    The follow-up counts toward the cap. It is ``merge`` when every remaining
    unit fits with that one extra job, and ``chain`` otherwise. ``self_counts``
    is true when the caller is itself a queued Slurm job (the previous wave's
    chain job), so one slot is already taken.
    """
    if n_remaining < 1:
        raise ValueError("n_remaining must be >= 1")
    if max_submitted < 3:
        raise ValueError("--max-submitted must be >= 3")
    slots = max_submitted - (1 if self_counts else 0)
    if n_remaining + 1 <= slots:
        return n_remaining, "merge"
    workers = slots - 1
    if workers < 1:
        raise ValueError(
            f"--max-submitted {max_submitted} cannot fit a worker and a follow-up "
            "while this job is still queued"
        )
    return workers, "chain"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Submit the G1 eval suite to PACE-ICE. One CPU job per work unit, "
            "at most 10 running and 50 submitted."
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
    parser.add_argument(
        "--max-submitted",
        type=int,
        default=DEFAULT_MAX_SUBMITTED,
        help=(
            "Most jobs to have queued at once, counting each array task and "
            f"the follow-up job (default: {DEFAULT_MAX_SUBMITTED})."
        ),
    )
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
    parser.add_argument(
        "--advance",
        action="store_true",
        help="Submit the next wave. The chaining job passes this; it is not needed on the login node.",
    )
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
    # Headless compute nodes have no DISPLAY. Export this before Python
    # starts so MuJoCo does not bind GLFW during import.
    lines.append(f"export MUJOCO_GL={shlex.quote(mujoco_gl or 'egl')}")
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
    offset: int = 0,
) -> str:
    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1")
    if max_in_flight < 1:
        raise ValueError("max_in_flight must be >= 1")
    if cpus < 1:
        raise ValueError("cpus must be >= 1")
    if offset < 0:
        raise ValueError("offset must be >= 0")
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
        f"export LOKA_JOB_OFFSET={int(offset)}",
        f'"$CONDA_PREFIX/bin/python" -m loka.submit_pace_ice --worker --run-dir {run_q}',
        "",
    ]
    return "\n".join(lines)


def render_chain_script(
    *,
    run_dir: Path,
    repo: Path,
    conda_env: str,
    conda_prefix: str | None,
    chain_args: Path,
    partition: str | None = None,
    account: str | None = None,
    qos: str | None = None,
    constraint: str | None = None,
    mujoco_gl: str | None = None,
    jobs_file: Path,
) -> str:
    """A one-task job that submits the next wave after this one has finished."""
    run_q = shlex.quote(str(run_dir))
    lines = _sbatch_headers(
        name="loka-eval-next",
        cpus=1,
        mem="4G",
        time_limit="00:15:00",
        output=f"{run_dir}/slurm/chain_%j.out",
        error=f"{run_dir}/slurm/chain_%j.err",
        partition=partition,
        account=account,
        qos=qos,
        constraint=constraint,
    )
    lines += [
        "# CPU only. This script does not request a GPU.",
        "# Submits the next wave once the previous array has left the queue.",
        *_preamble(repo, conda_env, conda_prefix, mujoco_gl, jobs_file),
        f"export LOKA_CHAIN_ARGS={shlex.quote(str(chain_args))}",
        f'"$CONDA_PREFIX/bin/python" -m loka.submit_pace_ice --advance --run-dir {run_q}',
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
        json.dumps(
            {"jobs": jobs, "num_trials": suite.num_trials, "next_offset": 0},
            indent=2,
        ),
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
    index = int(task) + int(os.environ.get("LOKA_JOB_OFFSET", "0"))
    jobs = manifest["jobs"]
    if index < 0 or index >= len(jobs):
        raise RuntimeError(f"job index {index} is outside {jobs_file} ({len(jobs)} units)")
    return jobs[index]


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


def _chain_payload(args, repo: Path, jobs_name: str) -> dict:
    return {
        "repo": str(repo),
        "jobs_name": jobs_name,
        "max_submitted": int(args.max_submitted),
        "max_in_flight": int(args.max_in_flight),
        "cpus": int(args.cpus),
        "mem": args.mem,
        "time_limit": args.time,
        "partition": args.partition,
        "account": args.account,
        "qos": args.qos,
        "constraint": args.constraint,
        "conda_env": args.conda_env,
        "conda_prefix": args.conda_prefix,
        "mujoco_gl": args.mujoco_gl,
    }


def _chain_args_path(run_dir: Path, jobs_name: str) -> Path:
    stem = "chain_args" if jobs_name == "jobs.json" else f"{Path(jobs_name).stem}_chain_args"
    return run_dir / "slurm" / f"{stem}.json"


def _write_wave_scripts(
    run_dir: Path, chain: dict, chain_path: Path, offset: int, count: int, kind: str
) -> tuple[Path, Path]:
    slurm_dir = run_dir / "slurm"
    jobs_name = str(chain["jobs_name"])
    jobs_file = slurm_dir / jobs_name
    if offset == 0 and jobs_name == "jobs.json":
        worker_name = "worker.sh"
    elif offset == 0 and jobs_name == "resubmit_jobs.json":
        worker_name = "resubmit.sh"
    else:
        stem = "resubmit" if jobs_name.startswith("resubmit") else "wave"
        worker_name = f"{stem}_{offset:04d}.sh"
    common = dict(
        run_dir=run_dir,
        repo=Path(chain["repo"]),
        conda_env=chain["conda_env"],
        conda_prefix=chain["conda_prefix"],
        partition=chain["partition"],
        account=chain["account"],
        qos=chain["qos"],
        constraint=chain["constraint"],
        mujoco_gl=chain["mujoco_gl"],
    )
    worker = slurm_dir / worker_name
    worker.write_text(
        render_worker_script(
            **common,
            n_jobs=count,
            cpus=int(chain["cpus"]),
            mem=chain["mem"],
            time_limit=chain["time_limit"],
            max_in_flight=int(chain["max_in_flight"]),
            jobs_file=jobs_file,
            offset=offset,
        ),
        encoding="utf-8",
    )
    if kind == "merge":
        follow = slurm_dir / "merge.sh"
        follow.write_text(render_merge_script(**common), encoding="utf-8")
    else:
        follow_name = "chain.sh" if jobs_name == "jobs.json" else "resubmit_chain.sh"
        follow = slurm_dir / follow_name
        follow.write_text(
            render_chain_script(**common, chain_args=chain_path, jobs_file=jobs_file),
            encoding="utf-8",
        )
    return worker, follow


def queue_wave(
    run_dir: Path,
    *,
    dry_run: bool,
    self_counts: bool,
    chain_path: Path | None = None,
) -> int:
    """Submit one wave, plus either the merge job or the job that queues the next wave."""
    run_dir = Path(run_dir)
    if chain_path is None:
        env = os.environ.get("LOKA_CHAIN_ARGS")
        chain_path = Path(env) if env else run_dir / "slurm" / "chain_args.json"
    try:
        chain = json.loads(chain_path.read_text(encoding="utf-8"))
        manifest_path = run_dir / "slurm" / str(chain["jobs_name"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        jobs = list(manifest["jobs"])
        pending = manifest.get("pending_wave")
        if pending:
            if dry_run:
                print("[pace] a follow-up submit was interrupted; dry run leaves it pending")
                return 0
            follow = Path(pending["follow_path"])
            follow_id = _sbatch(follow, ["--dependency", f"afterany:{pending['array_id']}"])
            manifest["next_offset"] = int(pending["offset"]) + int(pending["count"])
            manifest["pending_wave"] = None
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"[pace] submitted the pending follow-up {follow_id}")
            return 0
        offset = int(manifest.get("next_offset") or 0)
        remaining = len(jobs) - offset
        if remaining <= 0:
            print(f"[pace] every unit in {chain['jobs_name']} is already queued")
            return 0
        count, kind = first_wave(
            remaining, int(chain["max_submitted"]), self_counts=self_counts
        )
        worker, follow = _write_wave_scripts(run_dir, chain, chain_path, offset, count, kind)
        end = offset + count
        print(
            f"[pace] wave units {offset}:{end} of {len(jobs)} "
            f"({count} array tasks + 1 {kind} job; cap {chain['max_submitted']})"
        )
        if dry_run:
            print(f"[pace] dry run, scripts at {worker} and {follow}")
            return 0
        array_id = _sbatch(worker)
        manifest["pending_wave"] = {
            "offset": offset,
            "count": count,
            "array_id": array_id,
            "kind": kind,
            "follow_path": str(follow),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        follow_id = _sbatch(follow, ["--dependency", f"afterany:{array_id}"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["next_offset"] = end
        manifest["pending_wave"] = None
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[pace] follow-up {kind} job {follow_id}")
        return 0
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


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
    if args.advance:
        if not args.run_dir:
            print("--advance requires --run-dir", file=sys.stderr)
            return 2
        return queue_wave(Path(args.run_dir), dry_run=args.dry_run, self_counts=True)
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
    if args.max_submitted < 3:
        print("--max-submitted must be >= 3", file=sys.stderr)
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
        _worker, _merge, jobs = prepare_run(suite, run_dir, repo=repo, **_slurm_kwargs(args))
    except Exception as exc:
        print(f"Submit error: {exc}", file=sys.stderr)
        return 2
    recording = "on" if suite.recording.enabled else "off"
    print(
        f"[pace] {len(jobs)} units, at most {args.max_in_flight} running, "
        f"at most {args.max_submitted} submitted, {args.cpus} CPUs each, "
        f"recording {recording}"
    )
    print(f"[pace] run directory {run_dir}")
    chain_path = _chain_args_path(run_dir, "jobs.json")
    chain_path.write_text(
        json.dumps(_chain_payload(args, repo, "jobs.json"), indent=2),
        encoding="utf-8",
    )
    code = queue_wave(run_dir, dry_run=args.dry_run, self_counts=False, chain_path=chain_path)
    if code == 0 and not args.dry_run:
        print(run_dir)
    return code


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
        _worker, _merge, jobs = prepare_run(
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
    chain_path = _chain_args_path(run_dir, "resubmit_jobs.json")
    chain_path.write_text(
        json.dumps(_chain_payload(args, repo, "resubmit_jobs.json"), indent=2),
        encoding="utf-8",
    )
    return queue_wave(
        run_dir, dry_run=args.dry_run, self_counts=False, chain_path=chain_path
    )


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
