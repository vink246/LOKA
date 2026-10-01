#!/usr/bin/env bash
# Submit the Walker suite on a PACE-ICE login node.
# Each Slurm job is one trial of one baseline, CPU only, at most 10 in flight.
#
#   bash scripts/pace_ice_walker_suite.sh --num-trials 5
#   bash scripts/pace_ice_walker_suite.sh --num-trials 5 --partition cpu --cpus 32

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if command -v module >/dev/null 2>&1; then
  module load anaconda3 >/dev/null 2>&1 || module load anaconda >/dev/null 2>&1 || true
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not on PATH; cannot activate the loka env" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${LOKA_CONDA_ENV:-loka}"
if [ "${CONDA_DEFAULT_ENV:-}" != "${LOKA_CONDA_ENV:-loka}" ]; then
  echo "expected conda env ${LOKA_CONDA_ENV:-loka}, got ${CONDA_DEFAULT_ENV:-none}" >&2
  exit 1
fi

exec "$CONDA_PREFIX/bin/python" -m loka.submit_pace_ice "$@"
