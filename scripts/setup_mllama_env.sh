#!/usr/bin/env bash
# Create the loka-mllama conda env: LOKA plus Llama-3.2-11B-Vision-Instruct as
# a local orchestrator (run_loka --orchestrator-mllama) on a PACE GPU node.
# Does NOT download the ~21 GB checkpoint; use scripts/fetch_navila_checkpoint.py.
#
#   bash scripts/setup_mllama_env.sh                    # env at ~/scratch/envs/loka-mllama
#   bash scripts/setup_mllama_env.sh --prefix DIR       # env somewhere else
#   bash scripts/setup_mllama_env.sh --name loka-mllama # named env under conda's envs dir
#
# Everything large goes to scratch so the home quota is not filled. An exported
# value or the flag wins:
#   env             ~/scratch/envs/loka-mllama   (--prefix / --name)
#   HF_HOME         ~/scratch/huggingface        (--hf-home)
#   XDG_CACHE_HOME  ~/scratch/.cache             (--cache-dir; pip, conda packages and pip's temp dir go under it)
# Torch wheels come from LOKA_TORCH_INDEX (default: CUDA 12.1 wheels).
# Rerunning reuses an existing env; --recreate rebuilds it.
#
# Then:
#   conda activate ~/scratch/envs/loka-mllama
#   huggingface-cli login   # once; the repo is gated, accept the license on its HF page first
#   python scripts/fetch_navila_checkpoint.py --repo-id meta-llama/Llama-3.2-11B-Vision-Instruct \
#       --out ~/scratch/models/llama-3.2-11b-vision-instruct
#   python -m loka.run_loka --orchestrator-mllama ~/scratch/models/llama-3.2-11b-vision-instruct \
#       --vision --headless -T 60 --scene obstacle_course --record --operator "walk to the chair"

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="$HOME/scratch/envs/loka-mllama"
ENV_NAME=""
RECREATE=0
HF_HOME_DIR="${HF_HOME:-$HOME/scratch/huggingface}"
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/scratch/.cache}"
TORCH_INDEX="${LOKA_TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"

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
    --recreate) RECREATE=1; shift ;;
    -h|--help)
      sed -n '2,24p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

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

# OSMesa (MUJOCO_GL=osmesa fallback; the GPU path defaults to EGL) and the
# ffmpeg binary mediapy writes --record mp4s with. One channel avoids mixing.
CONDA_PKGS=(python=3.10 pip mesalib ffmpeg)
if [[ "$EXISTS" == 1 && "$RECREATE" == 0 ]]; then
  echo "reusing the existing env (pass --recreate to rebuild)"
  conda install -y "${TARGET[@]}" -c conda-forge --override-channels mesalib ffmpeg
else
  conda create -y "${TARGET[@]}" -c conda-forge --override-channels "${CONDA_PKGS[@]}"
fi
conda activate "${TARGET[1]}"

python -m pip install -U pip
# Torch first, from the CUDA index, so nothing below swaps in a CPU build.
python -m pip install "torch==2.4.1" "torchvision==0.19.1" --index-url "$TORCH_INDEX"
# LOKA plant deps (environment.yml). MuJoCo is pinned: the MjSpec scene
# builder uses the 3.2.x API (3.16 has no MjSpec.find_body).
# transformers 4.45 added Llama 3.2 Vision (Mllama); 4.46.3 is the tested pin.
python -m pip install \
  "mujoco==3.2.3" "numpy>=1.24,<2" "scipy>=1.10" "pyyaml>=6.0" "osqp>=1.1" \
  "pytest>=7.4" "dearpygui>=1.11" "openai>=1.30" "python-dotenv>=1.0" \
  "Pillow" "mediapy" \
  "transformers==4.46.3" "accelerate==1.1.1" "safetensors>=0.4.3" "huggingface_hub>=0.26,<1.0"

python -m pip check
python - <<'PY'
import torch, transformers, mujoco
from transformers import AutoProcessor, MllamaForConditionalGeneration  # noqa: F401
print(
    f"mllama env ok: torch {torch.__version__} (cuda build {torch.version.cuda}, "
    f"cuda available here: {torch.cuda.is_available()}), transformers {transformers.__version__}, "
    f"mujoco {mujoco.__version__}"
)
PY
echo
echo "Env ready. Next (on PACE):"
echo "  conda activate ${TARGET[1]}"
echo "  huggingface-cli login   # once; accept the license at https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct first"
echo "  python $ROOT/scripts/fetch_navila_checkpoint.py --repo-id meta-llama/Llama-3.2-11B-Vision-Instruct --out \$HOME/scratch/models/llama-3.2-11b-vision-instruct"
