#!/usr/bin/env python3
"""Download a VLA / VLM checkpoint to a local directory (PACE only).

    python scripts/fetch_navila_checkpoint.py --out ~/scratch/models/navila-llama3-8b-8f

Do **not** run this on a laptop: these checkpoints are 16-22 GB. After it
finishes, pass the same path to ``run_loka``:

* NaVILA (default repo): ``run_loka --navila --navila-model <path>``
* Llama 3.2 Vision orchestrator: ``run_loka --orchestrator-mllama <path>``.
  The repo is gated: accept the license on its Hugging Face page, then log in
  once on PACE (``huggingface-cli login``; the token is stored under HF_HOME)::

      python scripts/fetch_navila_checkpoint.py \\
          --repo-id meta-llama/Llama-3.2-11B-Vision-Instruct \\
          --out ~/scratch/models/llama-3.2-11b-vision-instruct

* VILA1.5 orchestrator: ``run_loka --orchestrator-vila <path>``::

      python scripts/fetch_navila_checkpoint.py \\
          --repo-id Efficient-Large-Model/Llama-3-VILA1.5-8B --out ~/scratch/models/llama-3-vila1.5-8b

Files under ``original/`` (Meta's consolidated ``.pth`` copy of the weights,
21 GB for Llama 3.2 Vision) are skipped by default; transformers reads the
safetensors.

HF_HOME defaults to ``~/scratch/huggingface`` and XDG_CACHE_HOME to
``~/scratch/.cache``. An exported value or ``--hf-home`` / ``--cache-dir`` wins.

Requires ``huggingface_hub`` (installed by both PACE env setup scripts).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

HUB_ID = "a8cheng/navila-llama3-8b-8f"
MIN_FREE_GB = 30.0
DEFAULT_HF_HOME = "~/scratch/huggingface"
DEFAULT_CACHE_HOME = "~/scratch/.cache"
DEFAULT_EXCLUDE = ("original/*",)


def _free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def _is_auth_error(exc: Exception) -> bool:
    name = type(exc).__name__
    return "Gated" in name or "Unauthorized" in name or "401" in str(exc) or "403" in str(exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Local directory for the checkpoint (e.g. ~/scratch/models/navila-llama3-8b-8f)",
    )
    parser.add_argument(
        "--repo-id",
        default=HUB_ID,
        help=f"Hugging Face repo id (default: {HUB_ID})",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=MIN_FREE_GB,
        help=f"Refuse to download if free disk under --out is below this (default: {MIN_FREE_GB})",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        metavar="GLOB",
        help=f"Repo paths to skip; repeatable (default: {' '.join(DEFAULT_EXCLUDE)}). "
        "Pass --exclude '' to download everything",
    )
    parser.add_argument(
        "--hf-home",
        default=os.environ.get("HF_HOME") or DEFAULT_HF_HOME,
        help=f"HF_HOME for the download (default: $HF_HOME, else {DEFAULT_HF_HOME})",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("XDG_CACHE_HOME") or DEFAULT_CACHE_HOME,
        help=f"XDG_CACHE_HOME (default: $XDG_CACHE_HOME, else {DEFAULT_CACHE_HOME})",
    )
    args = parser.parse_args(argv)
    exclude = [g for g in (args.exclude if args.exclude is not None else DEFAULT_EXCLUDE) if g]
    # huggingface_hub reads these at import time, so set them before importing it.
    os.environ["HF_HOME"] = str(Path(args.hf_home).expanduser())
    os.environ["XDG_CACHE_HOME"] = str(Path(args.cache_dir).expanduser())
    print(f"HF_HOME={os.environ['HF_HOME']} XDG_CACHE_HOME={os.environ['XDG_CACHE_HOME']}")

    out = args.out.expanduser().resolve()
    free = _free_gb(out)
    if free < float(args.min_free_gb):
        print(
            f"Only {free:.1f} GB free under {out}; need >= {args.min_free_gb:g} GB. Aborting.",
            file=sys.stderr,
        )
        return 1

    # Discourage accidental laptop runs when HOME looks like a personal desktop
    # and scratch is not in the path. Still allow an explicit --out anywhere.
    home = Path.home()
    if "scratch" not in str(out).lower() and out.is_relative_to(home):
        print(
            f"Warning: {out} is under $HOME and not on scratch. "
            "On PACE, prefer ~/scratch/... so home quota is not filled.",
            file=sys.stderr,
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub is not installed. Activate loka-navila or loka-mllama, "
            "or: pip install huggingface_hub",
            file=sys.stderr,
        )
        return 1

    skipped = f", skipping {' '.join(exclude)}" if exclude else ""
    print(f"Downloading {args.repo_id} -> {out} ({free:.1f} GB free{skipped}) …")
    try:
        snapshot_download(
            repo_id=args.repo_id,
            local_dir=str(out),
            ignore_patterns=exclude or None,
        )
    except Exception as exc:
        if not _is_auth_error(exc):
            raise
        print(
            f"Hugging Face refused {args.repo_id}: {exc}\n"
            f"Accept the license at https://huggingface.co/{args.repo_id}, then run "
            f"`huggingface-cli login` (or export HF_TOKEN) with HF_HOME={os.environ['HF_HOME']} "
            "and retry.",
            file=sys.stderr,
        )
        return 1
    print(
        f"Done: {out}\n"
        f"  NaVILA:                   run_loka --navila --navila-model {out} …\n"
        f"  Llama 3.2 Vision orch.:   run_loka --orchestrator-mllama {out} …\n"
        f"  VILA orchestrator:        run_loka --orchestrator-vila {out} …"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
