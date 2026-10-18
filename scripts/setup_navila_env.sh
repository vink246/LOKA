#!/usr/bin/env bash
# Create the loka-navila conda env for running NaVILA on a PACE GPU node.
# Does NOT download the ~16 GB checkpoint; use scripts/fetch_navila_checkpoint.py.
#
#   bash scripts/setup_navila_env.sh
#   bash scripts/setup_navila_env.sh --prefix ~/scratch/envs/loka-navila
#
# Caches default to scratch so home quota is not filled. An exported value or
# the flag wins:
#   HF_HOME         ~/scratch/huggingface   (--hf-home)
#   XDG_CACHE_HOME  ~/scratch/.cache        (--cache-dir; pip and conda packages go under it)
#
# Then:
#   conda activate loka-navila   # or the --prefix path
#   python scripts/fetch_navila_checkpoint.py --out ~/scratch/models/navila-llama3-8b-8f
#   python -m loka.run_loka --navila --navila-model ~/scratch/models/navila-llama3-8b-8f \
#       --headless -T 60 --scene obstacle_course --record --operator "walk to the chair"

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${LOKA_NAVILA_ENV:-loka-navila}"
PREFIX=""
HF_HOME_DIR="${HF_HOME:-$HOME/scratch/huggingface}"
CACHE_DIR="${XDG_CACHE_HOME:-$HOME/scratch/.cache}"
NAVILA_DIR="${NAVILA_SRC:-$ROOT/third_party/NaVILA}"
# Pin to a known NaVILA commit so flash-attn / transformers pins stay valid.
NAVILA_REPO="${NAVILA_REPO:-https://github.com/AnjieCheng/NaVILA.git}"
NAVILA_REF="${NAVILA_REF:-main}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --prefix=*) PREFIX="${1#*=}"; shift ;;
    --navila-dir) NAVILA_DIR="$2"; shift 2 ;;
    --name) ENV_NAME="$2"; shift 2 ;;
    --hf-home) HF_HOME_DIR="$2"; shift 2 ;;
    --hf-home=*) HF_HOME_DIR="${1#*=}"; shift ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --cache-dir=*) CACHE_DIR="${1#*=}"; shift ;;
    -h|--help)
      sed -n '2,17p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

export HF_HOME="$HF_HOME_DIR"
export XDG_CACHE_HOME="$CACHE_DIR"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_DIR/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$CACHE_DIR/conda/pkgs}"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"
echo "HF_HOME=$HF_HOME"
echo "XDG_CACHE_HOME=$XDG_CACHE_HOME (pip: $PIP_CACHE_DIR, conda pkgs: $CONDA_PKGS_DIRS)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not on PATH" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ -n "$PREFIX" ]]; then
  conda create -y -p "$PREFIX" python=3.10 pip
  conda activate "$PREFIX"
else
  conda create -y -n "$ENV_NAME" python=3.10 pip
  conda activate "$ENV_NAME"
fi

# OSMesa (MUJOCO_GL=osmesa fallback; --navila defaults to EGL on the GPU) and
# the ffmpeg binary mediapy writes --record mp4s with.
conda install -y -c conda-forge mesalib ffmpeg

python -m pip install -U pip
# LOKA plant deps (environment.yml). MuJoCo is pinned: the MjSpec scene
# builder uses the 3.2.x API (3.16 has no MjSpec.find_body).
python -m pip install \
  "mujoco==3.2.3" "numpy>=1.24" "scipy>=1.10" "pyyaml>=6.0" "osqp>=1.1" \
  "pytest>=7.4" "dearpygui>=1.11" "openai>=1.30" "python-dotenv>=1.0" \
  "Pillow" "mediapy" "huggingface_hub"

# Torch 2.3 + CUDA 12.1 wheels (Ampere+). Adjust the index if the cluster CUDA differs.
python -m pip install torch==2.3.0 torchvision==0.18.0 \
  --index-url https://download.pytorch.org/whl/cu121

# FlashAttention 2 wheel pinned by the NaVILA README (cu122 / torch2.3 / cp310).
FLASH_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
python -m pip install "$FLASH_URL" || {
  echo "flash-attn wheel install failed; try building from source or pick a matching wheel." >&2
  exit 1
}

if [[ ! -d "$NAVILA_DIR/.git" ]]; then
  mkdir -p "$(dirname "$NAVILA_DIR")"
  git clone "$NAVILA_REPO" "$NAVILA_DIR"
fi
git -C "$NAVILA_DIR" fetch --tags
git -C "$NAVILA_DIR" checkout "$NAVILA_REF"

python -m pip install -e "$NAVILA_DIR"
python -m pip install -e "$NAVILA_DIR[train]" || true
python -m pip install -e "$NAVILA_DIR[eval]" || true
python -m pip install "transformers==4.37.2"
# Upstream requires monkey-patched transformers / deepspeed files.
SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
if [[ -d "$NAVILA_DIR/llava/train/transformers_replace" ]]; then
  cp -rv "$NAVILA_DIR/llava/train/transformers_replace/"* "$SITE/transformers/" || true
fi
if [[ -d "$NAVILA_DIR/llava/train/deepspeed_replace" ]]; then
  mkdir -p "$SITE/deepspeed"
  cp -rv "$NAVILA_DIR/llava/train/deepspeed_replace/"* "$SITE/deepspeed/" || true
fi

python -c "import torch, llava; print('navila env ok', torch.__version__, torch.cuda.is_available())"
echo
echo "Env ready. Next (on PACE):"
echo "  python $ROOT/scripts/fetch_navila_checkpoint.py --out \$HOME/scratch/models/navila-llama3-8b-8f"
