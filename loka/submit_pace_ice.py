#!/usr/bin/env python3
"""Queue the G1 evaluation suite on PACE-ICE.

Run this on a login node. One CPU Slurm array task is one work unit
(task x perturbation x baseline x trial). At most ``--max-in-flight`` tasks
run at once. A follow-up job recomputes metrics and writes the plots.

PACE counts each array task as a submitted job, and a user can have about 50
in the queue. ``--max-submitted`` (default 25) keeps a wave under that cap,
including its one follow-up job. When the suite is larger, that follow-up
submits the next wave only after the previous tasks have left the queue.

Baselines on an API orchestrator (OpenAI, Anthropic) and baselines without
one run on CPU and request no GPU. A LOKA baseline whose orchestrator is a
local model (``loka/config/orchestrator_models.yaml``, ``provider: local``)
runs as GPU jobs instead: ``--gpu-units-per-job`` units of one model share a
job, which starts the model's OpenAI-compatible server once, waits for it,
and runs the units in turn. Each local model has its own array and wave chain
(GPU types differ per model). Models are queued one after another in
``--gpu-lanes`` lanes (default 1): a model's last wave hands over to the next
model's first, so at most lanes x ``--gpu-max-in-flight`` GPUs are busy. The
lanes hold ``--gpu-max-submitted`` of the ``--max-submitted`` slots. Every
chain ends in the same merge job, so the last one to finish writes the full
summary.

    bash scripts/pace_ice_eval.sh --num-trials 5
    bash scripts/pace_ice_eval.sh --num-trials 5 --cpus 2 \\
        --conda-prefix /home/hice1/vkulkarni46/scratch/envs/loka
    bash scripts/pace_ice_eval.sh --config loka/config/eval_ablations.yaml \\
        --conda-prefix ~/scratch/envs/loka --gpu-max-in-flight 2
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
from loka.harness.config import LOKA_BASELINE_TYPES, SuiteConfig, load_suite_config
from loka.harness.pipeline import recompute, write_resolved_config
from loka.harness.units import iter_units

# PACE rejects a submission once a user has about this many jobs queued.
# Each element of a job array counts as one job.
DEFAULT_MAX_SUBMITTED = 25
#: Of --max-submitted, the slots the local-model GPU chains may hold together.
DEFAULT_GPU_MAX_SUBMITTED = 6
DEFAULT_GPU_MAX_IN_FLIGHT = 2
DEFAULT_GPU_UNITS_PER_JOB = 5
#: Local models run one after another in each lane; one lane keeps at most
#: --gpu-max-in-flight GPUs busy for the whole suite.
DEFAULT_GPU_LANES = 1
CPU_JOBS_NAME = "jobs.json"


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
            "Submit the G1 eval suite to PACE-ICE. One CPU job per work unit "
            "(API orchestrators and no orchestrator), and GPU jobs that bundle "
            "the units of each local orchestrator model."
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
    gpu = parser.add_argument_group("local-model GPU jobs")
    gpu.add_argument(
        "--gpu-units-per-job",
        type=int,
        default=DEFAULT_GPU_UNITS_PER_JOB,
        help=f"Units of one local model per GPU job (default: {DEFAULT_GPU_UNITS_PER_JOB}).",
    )
    gpu.add_argument(
        "--gpu-max-in-flight",
        type=int,
        default=DEFAULT_GPU_MAX_IN_FLIGHT,
        help=f"Running GPU jobs per local model (default: {DEFAULT_GPU_MAX_IN_FLIGHT}).",
    )
    gpu.add_argument(
        "--gpu-max-submitted",
        type=int,
        default=DEFAULT_GPU_MAX_SUBMITTED,
        help=(
            "Of --max-submitted, the queued jobs the local-model lanes may hold "
            f"together (default: {DEFAULT_GPU_MAX_SUBMITTED}; at least 3 per lane)."
        ),
    )
    gpu.add_argument(
        "--gpu-lanes",
        type=int,
        default=DEFAULT_GPU_LANES,
        help=(
            "Local models are queued one after another in this many parallel lanes; "
            "at most lanes x --gpu-max-in-flight GPUs are busy at once "
            f"(default: {DEFAULT_GPU_LANES})."
        ),
    )
    gpu.add_argument("--gpu-gres", default=None, help="Override the registry gres, e.g. gpu:A100:1.")
    gpu.add_argument("--gpu-time", default=None, help="Override the registry wall time per GPU job.")
    gpu.add_argument("--gpu-mem", default=None, help="Override the registry memory per GPU job.")
    gpu.add_argument("--gpu-cpus", type=int, default=None, help="Override the registry CPUs per GPU job.")
    gpu.add_argument("--gpu-partition", default=None, help="Default: --partition.")
    gpu.add_argument("--gpu-account", default=None, help="Default: --account.")
    gpu.add_argument("--gpu-qos", default=None, help="Default: --qos.")
    gpu.add_argument("--gpu-constraint", default=None, help="Default: --constraint.")
    gpu.add_argument("--vllm-prefix", default=None, help="Server env for engine vllm (default: registry).")
    gpu.add_argument("--hf-prefix", default=None, help="Server env for engine hf (default: registry).")
    gpu.add_argument(
        "--no-local-check",
        action="store_true",
        help="Do not check on this node that local checkpoints and server envs exist.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--gpu-worker", action="store_true", help=argparse.SUPPRESS)
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
    gres: str | None = None,
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
    if gres:
        lines.append(f"#SBATCH --gres={gres}")
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
        *_preamble(repo, conda_env, conda_prefix, jobs_file),
        f"export LOKA_JOB_OFFSET={int(offset)}",
        f'"$CONDA_PREFIX/bin/python" -m loka.submit_pace_ice --worker --run-dir {run_q}',
        "",
    ]
    return "\n".join(lines)


def render_gpu_worker_script(
    *,
    run_dir: Path,
    repo: Path,
    conda_env: str,
    conda_prefix: str | None,
    n_jobs: int,
    entry: dict,
    gres: str,
    cpus: int,
    mem: str,
    time_limit: str,
    max_in_flight: int,
    jobs_file: Path,
    partition: str | None = None,
    account: str | None = None,
    qos: str | None = None,
    constraint: str | None = None,
    offset: int = 0,
) -> str:
    """One GPU array task: start the model server, then run a bundle of units."""
    from loka.harness.serving import render_server_block

    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1")
    if max_in_flight < 1:
        raise ValueError("max_in_flight must be >= 1")
    if not gres:
        raise ValueError(f"local model {entry['key']!r} has no GPU gres")
    key = str(entry["key"])
    run_q = shlex.quote(str(run_dir))
    lines = _sbatch_headers(
        name=f"loka-gpu-{key}",
        cpus=cpus,
        mem=mem,
        time_limit=time_limit,
        output=f"{run_dir}/slurm/gpu_{key}_%A_%a.out",
        error=f"{run_dir}/slurm/gpu_{key}_%A_%a.err",
        array=f"0-{n_jobs - 1}%{int(max_in_flight)}",
        partition=partition,
        account=account,
        qos=qos,
        constraint=constraint,
        gres=gres,
    )
    log = f"{run_dir}/slurm/server/{key}_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.log"
    lines += [
        f"# GPU job: serves local orchestrator {key} and runs a bundle of units against it.",
        *_preamble(repo, conda_env, conda_prefix, jobs_file),
        f"export LOKA_JOB_OFFSET={int(offset)}",
        f"mkdir -p {shlex.quote(str(run_dir))}/slurm/server",
        "nvidia-smi || true",
        *render_server_block(entry, log_path=log),
        f'"$CONDA_PREFIX/bin/python" -m loka.submit_pace_ice --gpu-worker --run-dir {run_q}',
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
        *_preamble(repo, conda_env, conda_prefix, jobs_file),
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
        *_preamble(repo, conda_env, conda_prefix, run_dir / "slurm" / "jobs.json"),
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


# -- which units need what ---------------------------------------------------


def baseline_model_entry(suite: SuiteConfig, baseline: str) -> dict | None:
    """The resolved orchestrator model of a LOKA baseline; None for the others."""
    from loka.agent.model_registry import resolve_model

    spec = suite.baseline(baseline)
    if spec.type not in LOKA_BASELINE_TYPES:
        return None
    entry = (spec.options or {}).get("model_entry")
    if entry:
        return dict(entry)
    orch = suite.orchestrator_for(spec)
    return resolve_model(orch.model, provider=orch.provider)


def local_model_entries(suite: SuiteConfig) -> dict[str, dict]:
    """Local orchestrator models used by the suite's baselines, by key."""
    entries: dict[str, dict] = {}
    for spec in suite.baselines:
        entry = baseline_model_entry(suite, spec.name)
        if entry is not None and entry["provider"] == "local":
            entries[str(entry["key"])] = entry
    return entries


def split_jobs(suite: SuiteConfig, jobs: list[dict]) -> tuple[list[dict], dict[str, list[dict]]]:
    """``(cpu_jobs, {model_key: units})`` in manifest order."""
    model_of: dict[str, str | None] = {}
    for spec in suite.baselines:
        entry = baseline_model_entry(suite, spec.name)
        model_of[spec.name] = str(entry["key"]) if entry and entry["provider"] == "local" else None
    cpu: list[dict] = []
    gpu: dict[str, list[dict]] = {}
    for job in jobs:
        key = model_of.get(str(job["baseline"]))
        if key is None:
            cpu.append(job)
        else:
            gpu.setdefault(key, []).append(job)
    return cpu, gpu


def bundle_units(key: str, units: list[dict], per_job: int, *, tag: str = "") -> list[dict]:
    if per_job < 1:
        raise ValueError("--gpu-units-per-job must be >= 1")
    return [
        {
            "id": f"gpu{tag}__{key}__{index:03d}",
            "model": key,
            "units": units[start:start + per_job],
        }
        for index, start in enumerate(range(0, len(units), per_job))
    ]


def _gpu_jobs_name(key: str, *, resubmit: bool = False) -> str:
    return f"{'resubmit_' if resubmit else ''}gpu_{key}_jobs.json"


def _gpu_resources(entry: dict, args) -> dict:
    slurm = dict(entry.get("slurm") or {})
    return {
        "gres": args.gpu_gres or slurm.get("gres"),
        "cpus": int(args.gpu_cpus or slurm.get("cpus") or 4),
        "mem": args.gpu_mem or slurm.get("mem") or "64G",
        "time_limit": args.gpu_time or slurm.get("time") or "04:00:00",
    }


def _with_server_env(entry: dict, args) -> dict:
    entry = dict(entry)
    override = args.vllm_prefix if entry.get("engine") == "vllm" else args.hf_prefix
    if override:
        entry["server_env"] = override
    return entry


def _require_credentials(suite: SuiteConfig, repo: Path, *, check_local: bool = True) -> None:
    """Fail on the login node, before anything is queued, if a provider cannot run.

    API keys are read on the compute node from the environment or ``.env``;
    they are never written into a Slurm script.
    """
    from loka.agent.model_registry import expand

    load_dotenv(repo / ".env")
    needed: dict[str, list[str]] = {}
    for spec in suite.baselines:
        entry = baseline_model_entry(suite, spec.name)
        if entry is None:
            continue
        needed.setdefault(str(entry["provider"]), []).append(spec.name)
        if entry["provider"] == "local" and check_local:
            for label, path in (("checkpoint", entry["path"]), ("server env", entry["server_env"])):
                if not expand(path).is_dir():
                    raise RuntimeError(
                        f"baseline {spec.name!r} ({entry['key']}) needs its {label} at {expand(path)}. "
                        "See scripts/fetch_local_models.sh and scripts/setup_vllm_env.sh, "
                        "or pass --no-local-check."
                    )
    for provider, variable in (("openai", "OPENAI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY")):
        if provider in needed and not os.environ.get(variable):
            names = ", ".join(repr(n) for n in needed[provider])
            raise RuntimeError(
                f"baseline {names} needs {variable} in the environment or in .env. "
                "The key is read on the compute node; it is not written into the Slurm script."
            )


def _require_openai(suite: SuiteConfig, repo: Path) -> None:
    """Kept for callers of the old name; checks every provider now."""
    _require_credentials(suite, repo, check_local=False)


# -- manifests -----------------------------------------------------------------


def _write_manifest(path: Path, jobs: list[dict], num_trials: int) -> None:
    path.write_text(
        json.dumps({"jobs": jobs, "num_trials": num_trials, "next_offset": 0}, indent=2),
        encoding="utf-8",
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
    jobs: list[dict] | None = None,
    jobs_name: str = CPU_JOBS_NAME,
    worker_name: str = "worker.sh",
) -> tuple[Path, Path, list[dict]]:
    """Write the CPU manifest, its first worker script and the merge script."""
    run_dir = Path(run_dir)
    slurm_dir = run_dir / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(run_dir, suite)
    if jobs is None:
        jobs = suite_jobs(suite)
    if not jobs:
        raise ValueError("No jobs to submit")
    jobs_path = slurm_dir / jobs_name
    _write_manifest(jobs_path, jobs, suite.num_trials)
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
        ),
        encoding="utf-8",
    )
    merge = _write_merge(run_dir, repo, conda_env, conda_prefix, partition, account, qos, constraint)
    return worker, merge, jobs


def _write_merge(run_dir, repo, conda_env, conda_prefix, partition, account, qos, constraint) -> Path:
    merge = Path(run_dir) / "slurm" / "merge.sh"
    merge.write_text(
        render_merge_script(
            run_dir=Path(run_dir),
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
    return merge


def suite_jobs(suite: SuiteConfig) -> list[dict]:
    jobs = [unit.to_dict() for unit in iter_units(suite)]
    if suite.unit is not None:
        jobs = [job for job in jobs if job["id"] == suite.unit]
        if not jobs:
            raise ValueError(f"Unknown unit {suite.unit!r}")
    return jobs


def missing_jobs(run_dir: Path) -> list[dict]:
    """CPU units in this run that have no done.json."""
    run_dir = Path(run_dir)
    path = run_dir / "slurm" / CPU_JOBS_NAME
    if not path.is_file():
        return []
    manifest = json.loads(path.read_text(encoding="utf-8"))
    missing = []
    for job in manifest.get("jobs") or []:
        if not (run_dir / str(job["id"]) / "done.json").is_file():
            missing.append(job)
    return missing


def missing_gpu_units(run_dir: Path) -> dict[str, list[dict]]:
    """Units without done.json in each local model's GPU manifest."""
    run_dir = Path(run_dir)
    missing: dict[str, list[dict]] = {}
    for path in sorted((run_dir / "slurm").glob("gpu_*_jobs.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for bundle in manifest.get("jobs") or []:
            for unit in bundle.get("units") or []:
                if not (run_dir / str(unit["id"]) / "done.json").is_file():
                    missing.setdefault(str(bundle["model"]), []).append(unit)
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


def run_gpu_worker(run_dir: Path) -> int:
    """Run every unit of this task's bundle against the already-started server.

    A failing unit is reported and the rest still run; finished units (with
    ``done.json``) are skipped, so a resubmitted bundle resumes.
    """
    from loka.harness.pipeline import run_suite

    run_dir = Path(run_dir)
    bundle = worker_job(run_dir)
    failed = []
    for unit in bundle.get("units") or []:
        suite = load_suite_config(_namespace(config=str(run_dir / "config.resolved.yaml")))
        suite.unit = str(unit["id"])
        suite.run_dir = run_dir
        try:
            run_suite(suite)
        except Exception as exc:
            print(f"[pace] unit {unit['id']} failed: {exc}", file=sys.stderr)
            failed.append(str(unit["id"]))
    if failed:
        print(f"[pace] {len(failed)} unit(s) of {bundle['id']} failed: {', '.join(failed)}", file=sys.stderr)
        return 1
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
    }


def _chain_payload(args, repo: Path, jobs_name: str, *, max_submitted: int | None = None) -> dict:
    return {
        "kind": "cpu",
        "repo": str(repo),
        "jobs_name": jobs_name,
        "prefix": "",
        "max_submitted": int(max_submitted if max_submitted is not None else args.max_submitted),
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
    }


def _gpu_chain_payload(args, repo: Path, entry: dict, jobs_name: str, max_submitted: int) -> dict:
    resources = _gpu_resources(entry, args)
    return {
        "kind": "gpu",
        "repo": str(repo),
        "jobs_name": jobs_name,
        "prefix": f"gpu_{entry['key']}_",
        "entry": entry,
        "gres": resources["gres"],
        "max_submitted": int(max_submitted),
        "max_in_flight": int(args.gpu_max_in_flight),
        "cpus": resources["cpus"],
        "mem": resources["mem"],
        "time_limit": resources["time_limit"],
        "partition": args.gpu_partition or args.partition,
        "account": args.gpu_account or args.account,
        "qos": args.gpu_qos or args.qos,
        "constraint": args.gpu_constraint or args.constraint,
        "conda_env": args.conda_env,
        "conda_prefix": args.conda_prefix,
    }


def _chain_args_path(run_dir: Path, jobs_name: str) -> Path:
    stem = "chain_args" if jobs_name == CPU_JOBS_NAME else f"{Path(jobs_name).stem}_chain_args"
    return run_dir / "slurm" / f"{stem}.json"


def _script_names(chain: dict, offset: int) -> tuple[str, str]:
    """``(worker, chain)`` script names for a wave starting at ``offset``."""
    prefix = str(chain.get("prefix") or "")
    resubmit = str(chain["jobs_name"]).startswith("resubmit")
    stem = "resubmit" if resubmit else "wave"
    if offset == 0:
        worker = f"{prefix}{'resubmit' if resubmit else 'worker'}.sh"
    else:
        worker = f"{prefix}{stem}_{offset:04d}.sh"
    follow = f"{prefix}{'resubmit_chain' if resubmit else 'chain'}.sh"
    return worker, follow


def _write_wave_scripts(
    run_dir: Path, chain: dict, chain_path: Path, offset: int, count: int, kind: str
) -> tuple[Path, Path]:
    slurm_dir = run_dir / "slurm"
    jobs_file = slurm_dir / str(chain["jobs_name"])
    worker_name, chain_name = _script_names(chain, offset)
    common = dict(
        run_dir=run_dir,
        repo=Path(chain["repo"]),
        conda_env=chain["conda_env"],
        conda_prefix=chain["conda_prefix"],
        partition=chain["partition"],
        account=chain["account"],
        qos=chain["qos"],
        constraint=chain["constraint"],
    )
    worker = slurm_dir / worker_name
    if chain.get("kind") == "gpu":
        text = render_gpu_worker_script(
            **common,
            n_jobs=count,
            entry=chain["entry"],
            gres=chain["gres"],
            cpus=int(chain["cpus"]),
            mem=chain["mem"],
            time_limit=chain["time_limit"],
            max_in_flight=int(chain["max_in_flight"]),
            jobs_file=jobs_file,
            offset=offset,
        )
    else:
        text = render_worker_script(
            **common,
            n_jobs=count,
            cpus=int(chain["cpus"]),
            mem=chain["mem"],
            time_limit=chain["time_limit"],
            max_in_flight=int(chain["max_in_flight"]),
            jobs_file=jobs_file,
            offset=offset,
        )
    worker.write_text(text, encoding="utf-8")
    # Follow-up jobs are short CPU jobs on the CPU partition settings.
    follow_common = dict(common)
    if chain.get("kind") == "gpu":
        cpu = chain.get("cpu_slurm") or {}
        follow_common.update({k: cpu.get(k) for k in ("partition", "account", "qos", "constraint")})
    if kind == "merge" and chain.get("next_chain"):
        # This model's last wave: the follow-up starts the next model of the lane.
        next_path = Path(chain["next_chain"])
        next_chain = json.loads(next_path.read_text(encoding="utf-8"))
        follow = slurm_dir / f"{chain.get('prefix') or ''}next.sh"
        follow.write_text(
            render_chain_script(
                **follow_common,
                chain_args=next_path,
                jobs_file=slurm_dir / str(next_chain["jobs_name"]),
            ),
            encoding="utf-8",
        )
    elif kind == "merge":
        follow = slurm_dir / "merge.sh"
        follow.write_text(render_merge_script(**follow_common), encoding="utf-8")
    else:
        follow = slurm_dir / chain_name
        follow.write_text(
            render_chain_script(**follow_common, chain_args=chain_path, jobs_file=jobs_file),
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
        noun = "GPU array tasks" if chain.get("kind") == "gpu" else "array tasks"
        follow_kind = "next-model" if kind == "merge" and chain.get("next_chain") else kind
        print(
            f"[pace] {chain['jobs_name']}: wave {offset}:{end} of {len(jobs)} "
            f"({count} {noun} + 1 {follow_kind} job; cap {chain['max_submitted']})"
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


def _lane_count(args, n_gpu_models: int) -> int:
    if int(args.gpu_lanes) < 1:
        raise ValueError("--gpu-lanes must be >= 1")
    return min(int(args.gpu_lanes), n_gpu_models)


def _cap_split(args, n_gpu_models: int, has_cpu: bool) -> tuple[int, int]:
    """``(cpu_cap, cap_per_gpu_lane)`` within --max-submitted."""
    if n_gpu_models == 0:
        return int(args.max_submitted), 0
    lanes = _lane_count(args, n_gpu_models)
    gpu_total = int(args.gpu_max_submitted)
    per_lane = gpu_total // lanes
    if per_lane < 3:
        raise ValueError(
            f"--gpu-max-submitted {gpu_total} leaves {per_lane} slots for each of "
            f"{lanes} GPU lanes; each needs at least 3"
        )
    cpu_cap = int(args.max_submitted) - per_lane * lanes
    if has_cpu and cpu_cap < 3:
        raise ValueError(
            f"--max-submitted {args.max_submitted} minus the GPU share "
            f"{per_lane * lanes} leaves {cpu_cap} CPU slots; at least 3 are needed"
        )
    return cpu_cap, per_lane


def _queue_gpu_models(
    args,
    repo: Path,
    run_dir: Path,
    suite: SuiteConfig,
    gpu_units: dict[str, list[dict]],
    per_model_cap: int,
    *,
    resubmit: bool = False,
) -> int:
    entries = local_model_entries(suite)
    lanes = _lane_count(args, len(gpu_units))
    chains: list[tuple[str, Path, dict]] = []
    for key, units in gpu_units.items():
        entry = _with_server_env(entries[key], args)
        tag = "_resubmit" if resubmit else ""
        bundles = bundle_units(key, units, int(args.gpu_units_per_job), tag=tag)
        jobs_name = _gpu_jobs_name(key, resubmit=resubmit)
        _write_manifest(run_dir / "slurm" / jobs_name, bundles, suite.num_trials)
        payload = _gpu_chain_payload(args, repo, entry, jobs_name, per_model_cap)
        payload["cpu_slurm"] = {
            "partition": args.partition,
            "account": args.account,
            "qos": args.qos,
            "constraint": args.constraint,
        }
        if not payload["gres"]:
            print(f"[pace] local model {key} has no slurm.gres; pass --gpu-gres", file=sys.stderr)
            return 2
        print(
            f"[pace] {key}: {len(units)} units in {len(bundles)} GPU jobs "
            f"({payload['gres']}, {payload['cpus']} CPUs, {payload['mem']}, {payload['time_limit']}; "
            f"at most {args.gpu_max_in_flight} running)"
        )
        chains.append((key, _chain_args_path(run_dir, jobs_name), payload))
    # Round-robin into lanes; within a lane each model's last wave starts the next model.
    lane_members: list[list[int]] = [list(range(i, len(chains), lanes)) for i in range(lanes)]
    for members in lane_members:
        for position, index in enumerate(members):
            key, path, payload = chains[index]
            if position + 1 < len(members):
                payload["next_chain"] = str(chains[members[position + 1]][1])
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        order = " -> ".join(chains[i][0] for i in members)
        print(f"[pace] GPU lane: {order}")
    code = 0
    for members in lane_members:
        chain_path = chains[members[0]][1]
        code = queue_wave(run_dir, dry_run=args.dry_run, self_counts=False, chain_path=chain_path) or code
    return code


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.worker or args.gpu_worker:
        if not args.run_dir:
            print("--worker requires --run-dir", file=sys.stderr)
            return 2
        try:
            if args.gpu_worker:
                return run_gpu_worker(Path(args.run_dir))
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
    if args.cpus < 1 or args.max_in_flight < 1 or args.gpu_max_in_flight < 1:
        print("--cpus, --max-in-flight and --gpu-max-in-flight must be >= 1", file=sys.stderr)
        return 2
    if args.max_submitted < 3:
        print("--max-submitted must be >= 3", file=sys.stderr)
        return 2
    conda_prefix = _check_prefix(args.conda_prefix)
    if conda_prefix is None and args.conda_prefix:
        return 2
    args.conda_prefix = conda_prefix
    try:
        _require_credentials(suite, repo, check_local=not args.no_local_check)
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
        jobs = suite_jobs(suite)
        cpu_jobs, gpu_units = split_jobs(suite, jobs)
        cpu_cap, per_model_cap = _cap_split(args, len(gpu_units), bool(cpu_jobs))
        (run_dir / "slurm").mkdir(parents=True, exist_ok=True)
        write_resolved_config(run_dir, suite)
        if cpu_jobs:
            prepare_run(suite, run_dir, repo=repo, jobs=cpu_jobs, **_slurm_kwargs(args))
        else:
            _write_merge(
                run_dir, repo, args.conda_env, args.conda_prefix,
                args.partition, args.account, args.qos, args.constraint,
            )
    except Exception as exc:
        print(f"Submit error: {exc}", file=sys.stderr)
        return 2
    recording = "on" if suite.recording.enabled else "off"
    print(
        f"[pace] {len(jobs)} units: {len(cpu_jobs)} on CPU (at most {args.max_in_flight} running, "
        f"{cpu_cap} submitted, {args.cpus} CPUs each), "
        f"{sum(len(u) for u in gpu_units.values())} on GPU for {len(gpu_units)} local model(s); "
        f"recording {recording}"
    )
    print(f"[pace] run directory {run_dir}")
    code = 0
    if cpu_jobs:
        chain_path = _chain_args_path(run_dir, CPU_JOBS_NAME)
        chain_path.write_text(
            json.dumps(_chain_payload(args, repo, CPU_JOBS_NAME, max_submitted=cpu_cap), indent=2),
            encoding="utf-8",
        )
        code = queue_wave(run_dir, dry_run=args.dry_run, self_counts=False, chain_path=chain_path)
    if gpu_units and code == 0:
        code = _queue_gpu_models(args, repo, run_dir, suite, gpu_units, per_model_cap)
    if code == 0 and not args.dry_run:
        print(run_dir)
    return code


def _resubmit(args, repo: Path) -> int:
    run_dir = Path(args.resubmit_missing)
    jobs = missing_jobs(run_dir)
    gpu_units = missing_gpu_units(run_dir)
    if not jobs and not gpu_units:
        print(f"[pace] nothing missing in {run_dir}")
        return 0
    suite = load_suite_config(_namespace(config=str(run_dir / "config.resolved.yaml")))
    suite.run_dir = run_dir
    conda_prefix = _check_prefix(args.conda_prefix)
    if conda_prefix is None and args.conda_prefix:
        return 2
    args.conda_prefix = conda_prefix
    try:
        _require_credentials(suite, repo, check_local=not args.no_local_check)
        cpu_cap, per_model_cap = _cap_split(args, len(gpu_units), bool(jobs))
    except (RuntimeError, ValueError) as exc:
        print(f"Submit error: {exc}", file=sys.stderr)
        return 2
    code = 0
    if jobs:
        try:
            prepare_run(
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
        print(f"[pace] resubmitting {len(jobs)} unfinished CPU units")
        chain_path = _chain_args_path(run_dir, "resubmit_jobs.json")
        chain_path.write_text(
            json.dumps(
                _chain_payload(args, repo, "resubmit_jobs.json", max_submitted=cpu_cap), indent=2
            ),
            encoding="utf-8",
        )
        code = queue_wave(run_dir, dry_run=args.dry_run, self_counts=False, chain_path=chain_path)
    if gpu_units and code == 0:
        print(f"[pace] resubmitting {sum(len(u) for u in gpu_units.values())} unfinished GPU units")
        code = _queue_gpu_models(
            args, repo, run_dir, suite, gpu_units, per_model_cap, resubmit=True
        )
    return code


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
