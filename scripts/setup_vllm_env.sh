#!/usr/bin/env bash
# Create the loka-vllm conda env: the one GPU env that serves every local
# orchestrator model in loka/config/orchestrator_models.yaml, through vLLM
# (engine vllm) or loka.vla.serve_hf (engine hf, Llama-3.2-11B-Vision; this
# replaces the older loka-mllama env). LOKA's evaluation itself runs in the
# regular CPU loka env and talks to the server over HTTP. NaVILA keeps its own
# loka-navila env (VILA pins an older transformers/torch). PACE only.
#
#   bash scripts/setup_vllm_env.sh                    # env at ~/scratch/envs/loka-vllm
#   bash scripts/setup_vllm_env.sh --prefix DIR       # env somewhere else
#   bash scripts/setup_vllm_env.sh --vllm 0.11.0      # another vLLM release
#
# Everything large goes to scratch so the home quota is not filled. An exported
# value or the flag wins:
#   env             ~/scratch/envs/loka-vllm   (--prefix / --name)
#   HF_HOME         ~/scratch/huggingface      (--hf-home)
#   XDG_CACHE_HOME  ~/scratch/.cache           (--cache-dir; pip, conda packages and pip's temp dir go under it)
# The vLLM pin (LOKA_VLLM_VERSION) must serve every vllm entry in the
# registry: gpt-oss needs >= 0.10.2; Gemma 3, Mistral Small 3.x and Llama 3.x
# are older. vLLM wheels bring their own CUDA torch build. Rerunning reuses an
# existing env; --recreate rebuilds it.
#
# Then (on PACE):
#   bash scripts/fetch_local_models.sh gemma-3-27b-it
#   salloc --gres=gpu:H100:1 --cpus-per-task=4 --mem=96G -t 01:00:00
#   bash scripts/serve_local_model.sh gemma-3-27b-it

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="$HOME/scratch/envs/loka-vllm"
ENV_NAME=""
RECREATE=0
HF_HOME_DIR="${HF_HOME:-$HOME/scratch/huggingface}"
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/scratch/.cache}"
VLLM_VERSION="${LOKA_VLLM_VERSION:-0.11.0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --prefix=*) PREFIX="${1#*=}"; shift ;;
    --name) ENV_NAME="$2"; PREFIX=""; shift 2 ;;
    --name=*) ENV_NAME="${1#*=}"; PREFIX=""; shift ;;
    --hf-home) HF_HOME_DIR="$2"; shift 2 ;;
    --hf-home=*) HF_HOME_DIR="${1#*=}"; shift ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --cache-dir=*) CACHE_DIR="${1#*=}"; shift ;;
    --vllm) VLLM_VERSION="$2"; shift 2 ;;
    --vllm=*) VLLM_VERSION="${1#*=}"; shift ;;
    --recreate) RECREATE=1; shift ;;
    -h|--help)
      sed -n '2,25p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if command -v module >/dev/null 2>&1; then
  module load anaconda3 >/dev/null 2>&1 || module load anaconda >/dev/null 2>&1 || true
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not on PATH (on PACE: module load anaconda3)" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

export HF_HOME="$HF_HOME_DIR"
export XDG_CACHE_HOME="$CACHE_DIR"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_DIR/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$CACHE_DIR/conda/pkgs}"
# pip unpacks multi-GB CUDA wheels in TMPDIR; node-local /tmp can be small.
export TMPDIR="${LOKA_TMPDIR:-$CACHE_DIR/tmp}"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$TMPDIR"
echo "HF_HOME=$HF_HOME"
echo "XDG_CACHE_HOME=$XDG_CACHE_HOME (pip: $PIP_CACHE_DIR, conda pkgs: $CONDA_PKGS_DIRS, tmp: $TMPDIR)"

if [[ -n "$PREFIX" ]]; then
  TARGET=(-p "$PREFIX")
  EXISTS=0; [[ -d "$PREFIX/conda-meta" ]] && EXISTS=1
  echo "env: $PREFIX"
else
  TARGET=(-n "$ENV_NAME")
  EXISTS=0; conda env list | awk '{print $1}' | grep -qx "$ENV_NAME" && EXISTS=1
  echo "env: $ENV_NAME"
fi

if [[ "$EXISTS" == 1 && "$RECREATE" == 1 ]]; then
  conda env remove -y "${TARGET[@]}"
  EXISTS=0
fi
if [[ "$EXISTS" == 1 ]]; then
  echo "reusing the existing env (pass --recreate to rebuild)"
else
  conda create -y "${TARGET[@]}" -c conda-forge --override-channels python=3.12 pip
fi
conda activate "${TARGET[1]}"

python -m pip install -U pip
python -m pip install "vllm==${VLLM_VERSION}" "huggingface_hub>=0.26"
# LOKA's light runtime deps, so this env also runs loka.vla.serve_hf (the
# in-process Llama-3.2-11B-Vision server, engine hf) and fetch_local_models.sh.
# Installed after vLLM so they fit around its torch/transformers/numpy pins.
python -m pip install "mujoco==3.2.3" "scipy>=1.10" "pyyaml>=6.0" "osqp>=1.1" \
  "python-dotenv>=1.0" "Pillow" "accelerate"
python -m pip check || echo "pip check reported conflicts (vLLM pins its own stack); see above"
(cd "$ROOT" && python - <<'PY'
import mujoco, torch, transformers, vllm
from transformers import MllamaForConditionalGeneration  # noqa: F401
import loka.vla.serve_hf  # noqa: F401
print(
    f"vllm env ok: vllm {vllm.__version__}, torch {torch.__version__} "
    f"(cuda build {torch.version.cuda}, cuda available here: {torch.cuda.is_available()}), "
    f"transformers {transformers.__version__} (Mllama ok), mujoco {mujoco.__version__}"
)
PY
)
# Wheels and pip's temp files are not needed once installed.
python -m pip cache purge >/dev/null 2>&1 || true
rm -rf "${TMPDIR:?}"/* 2>/dev/null || true
echo
echo "Env ready: ${TARGET[1]}"
echo "Next (on PACE):"
echo "  bash $ROOT/scripts/fetch_local_models.sh gemma-3-27b-it"
echo "  bash $ROOT/scripts/serve_local_model.sh gemma-3-27b-it   # inside a GPU salloc"
