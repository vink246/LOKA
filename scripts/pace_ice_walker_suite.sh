#!/usr/bin/env bash
# Submit the Walker suite on a PACE-ICE login node.
# Each Slurm job is one perturbation of one baseline and trial, CPU only.
#
#   bash scripts/pace_ice_walker_suite.sh --num-trials 5
#   bash scripts/pace_ice_walker_suite.sh --num-trials 5 --cpus 32 \
#       --conda-prefix /home/hice1/vkulkarni46/scratch/envs/loka

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONDA_PREFIX_ARG=""
prev=""
for arg in "$@"; do
  if [ "$prev" = "--conda-prefix" ]; then
    CONDA_PREFIX_ARG="$arg"
  fi
  case "$arg" in
    --conda-prefix=*)
      CONDA_PREFIX_ARG="${arg#--conda-prefix=}"
      ;;
  esac
  prev="$arg"
done

if command -v module >/dev/null 2>&1; then
  module load anaconda3 >/dev/null 2>&1 || module load anaconda >/dev/null 2>&1 || true
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not on PATH; cannot activate the loka env" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
if [ -n "$CONDA_PREFIX_ARG" ]; then
  conda activate "$CONDA_PREFIX_ARG"
  expected="$(cd "$CONDA_PREFIX_ARG" && pwd)"
  actual="$(cd "$CONDA_PREFIX" && pwd)"
  if [ "$actual" != "$expected" ]; then
    echo "expected conda prefix $expected, got ${CONDA_PREFIX:-none}" >&2
    exit 1
  fi
else
  conda activate "${LOKA_CONDA_ENV:-loka}"
  if [ "${CONDA_DEFAULT_ENV:-}" != "${LOKA_CONDA_ENV:-loka}" ]; then
    echo "expected conda env ${LOKA_CONDA_ENV:-loka}, got ${CONDA_DEFAULT_ENV:-none}" >&2
    exit 1
  fi
fi

exec "$CONDA_PREFIX/bin/python" -m loka.submit_pace_ice "$@"
