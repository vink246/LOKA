#!/usr/bin/env python3
"""Download the NaVILA-8B checkpoint to a local directory (PACE only).

    python scripts/fetch_navila_checkpoint.py --out ~/scratch/models/navila-llama3-8b-8f

Do **not** run this on a laptop: the checkpoint is ~16 GB. After it finishes,
pass the same path to ``run_loka --navila --navila-model <path>``.

The same script fetches NaVILA's base model for ``run_loka --orchestrator-vila``:

    python scripts/fetch_navila_checkpoint.py \\
        --repo-id Efficient-Large-Model/Llama-3-VILA1.5-8B --out ~/scratch/models/llama-3-vila1.5-8b

HF_HOME defaults to ``~/scratch/huggingface`` and XDG_CACHE_HOME to
``~/scratch/.cache``. An exported value or ``--hf-home`` / ``--cache-dir`` wins.

Requires ``huggingface_hub`` (installed by ``scripts/setup_navila_env.sh``).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

HUB_ID = "a8cheng/navila-llama3-8b-8f"
MIN_FREE_GB = 20.0
DEFAULT_HF_HOME = "~/scratch/huggingface"
DEFAULT_CACHE_HOME = "~/scratch/.cache"


def _free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


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
            "huggingface_hub is not installed. Activate loka-navila "
            "(scripts/setup_navila_env.sh) or: pip install huggingface_hub",
            file=sys.stderr,
        )
        return 1

    print(f"Downloading {args.repo_id} -> {out} ({free:.1f} GB free) …")
    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(out),
        local_dir_use_symlinks=False,
    )
    print(
        f"Done: {out}\n  NaVILA:            run_loka --navila --navila-model {out} …\n"
        f"  VILA orchestrator: run_loka --orchestrator-vila {out} …"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
