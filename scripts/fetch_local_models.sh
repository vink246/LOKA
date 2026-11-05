#!/usr/bin/env bash
# Download local orchestrator checkpoints listed in loka/config/orchestrator_models.yaml
# to their registry path (default ~/scratch/models/KEY). PACE only: the
# checkpoints are 8-240 GB.
#
#   bash scripts/fetch_local_models.sh gemma-3-4b-it gemma-3-12b-it gemma-3-27b-it
#   bash scripts/fetch_local_models.sh --list
#
# Run it from the loka-vllm env (it has huggingface_hub and LOKA's deps). Gated
# repos (Meta Llama, Google Gemma, some Mistral) need the license accepted on
# the model's Hugging Face page and a login with the same HF_HOME:
#   HF_HOME=~/scratch/huggingface huggingface-cli login
# The free-space check reads the filesystem, not your PACE quota: check
# `pace-quota` before a large fetch. Approximate sizes: gemma-3-4b-it 9 GB,
# gemma-3-12b-it 24 GB, gemma-3-27b-it 55 GB, gpt-oss-20b 14 GB,
# llama-3.2-11b-vision 21 GB, gpt-oss-120b 65 GB.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export HF_HOME="${HF_HOME:-$HOME/scratch/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/scratch/.cache}"
PY="${LOKA_PYTHON:-python}"

if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
  sed -n '2,14p' "$0"
  exit 0
fi
if [[ "$1" == "--list" ]]; then
  "$PY" -c "from loka.agent.model_registry import known_local_models; print('\n'.join(known_local_models()))"
  exit 0
fi

status=0
for key in "$@"; do
  line="$("$PY" -m loka.harness.serving "$key" --print-fetch)" || { status=1; continue; }
  IFS=$'\t' read -r repo out excludes <<< "$line"
  exclude_args=()
  if [[ -n "$excludes" ]]; then
    IFS=',' read -ra globs <<< "$excludes"
    for g in "${globs[@]}"; do exclude_args+=(--exclude "$g"); done
  fi
  if [[ -z "$repo" ]]; then
    echo "[fetch] $key has no hf_repo in the registry; place the checkpoint at $out yourself" >&2
    status=1
    continue
  fi
  if [[ -f "$out/config.json" || -f "$out/params.json" ]]; then
    echo "[fetch] $key already at $out (delete it to fetch again)"
    continue
  fi
  echo "[fetch] $key: $repo -> $out${excludes:+ (skipping $excludes)}"
  if ! "$PY" scripts/fetch_navila_checkpoint.py --repo-id "$repo" --out "$out" --min-free-gb 20 "${exclude_args[@]}"; then
    echo "[fetch] $key failed. If the repo is gated, accept its license at https://huggingface.co/$repo and log in." >&2
    status=1
  fi
done
exit "$status"
