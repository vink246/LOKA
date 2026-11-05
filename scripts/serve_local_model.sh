#!/usr/bin/env bash
# Smoke-test a local orchestrator model on a PACE GPU node: start its server,
# wait for /health, send one real LOKA operator turn, check the reply parses,
# and stop the server. Run it inside a GPU allocation, from the loka env:
#
#   salloc --gres=gpu:H100:1 --cpus-per-task=4 --mem=96G -t 01:00:00
#   conda activate ~/scratch/envs/loka
#   bash scripts/serve_local_model.sh gemma-3-27b-it
#   bash scripts/serve_local_model.sh gemma-3-27b-it --keep   # leave the server up (Ctrl-C stops it)
#
# The server log goes to logs/serve/KEY.log.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  sed -n '2,12p' "$0"
  exit 0
fi
KEY="$1"; shift
KEEP=0
for arg in "$@"; do
  case "$arg" in
    --keep) KEEP=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 1 ;;
  esac
done

PY="${LOKA_PYTHON:-python}"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "no nvidia-smi: run this inside a PACE GPU allocation (salloc --gres=gpu:...)" >&2
  exit 1
fi
mkdir -p logs/serve
LOG="logs/serve/${KEY}.log"
BLOCK="$("$PY" -m loka.harness.serving "$KEY" --print-block --log "$LOG")"
eval "$BLOCK"

"$PY" -m loka.agent.providers --smoke "$KEY"
if [[ "$KEEP" == 1 ]]; then
  echo "[serve] server stays up at $LOKA_LOCAL_BASE_URL (Ctrl-C to stop)"
  wait "$LOKA_SERVER_PID"
fi
