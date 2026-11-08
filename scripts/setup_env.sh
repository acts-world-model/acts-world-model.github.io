#!/usr/bin/env bash
# Create the conda env (torch 2.10 + cu128, cosmos-framework deps WITHOUT flash-attn /
# transformer-engine / natten: the Flex-Triton attention path needs none) and clone the official
# NVIDIA cosmos-framework at the exact commit the models were trained with (never modified).
#   bash scripts/setup_env.sh            # env name acts_wm under $CONDA_ROOT (default ~/miniconda3)
# Needs an NVIDIA driver that supports CUDA 12.8 (tested on H100).
set -euo pipefail
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-acts_wm}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
COSMOS_COMMIT=3f94d25
PY="${PYTHON_VERSION:-3.12}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
if [[ ! -d "$CONDA_ROOT/envs/$ENV_NAME" ]]; then
  conda create -y -n "$ENV_NAME" "python=$PY"
fi
PIP="$CONDA_ROOT/envs/$ENV_NAME/bin/python -m pip"
export PYTHONNOUSERSITE=1
# ignore any user-level pip.conf (an unreachable extra index breaks resolution)
export PIP_CONFIG_FILE=/dev/null
$PIP install --upgrade pip
$PIP install --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.10.0" "torchvision==0.25.0" "triton==3.6.0"
$PIP install \
  "transformers>=4.57.1,<5.0.0" "diffusers>=0.39.0" accelerate \
  nvidia-ml-py nvidia-cudnn-frontend \
  omegaconf hydra-core pydantic tyro termcolor loguru einops attrs cattrs msgpack \
  scipy requests av imageio "imageio-ffmpeg==0.6.0" websockets obstore \
  pyyaml numpy safetensors huggingface_hub \
  lpips scikit-image opencv-python-headless pytest iopath \
  pyarrow matplotlib
echo "ENV READY: $CONDA_ROOT/envs/$ENV_NAME"

# PyPI deps (accelerate/lpips) resolve "torch>=2.0" to the latest cu130 wheel; pin back.
$PIP install --no-deps --force-reinstall --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.10.0" "torchvision==0.25.0" "triton==3.6.0"
$CONDA_ROOT/envs/$ENV_NAME/bin/python -c "import torch; assert torch.__version__.startswith('2.10.0+cu128'), torch.__version__; print('torch', torch.__version__)"
echo "ENV PINNED: torch 2.10.0+cu128"

# The transient torch-2.14+cu130 install leaves nvidia-*-cu13 packages that share the
# nvidia/cudnn (etc.) namespace dirs with the cu12 ones and overwrite libcudnn.so.9 ->
# "CUDNN_STATUS_NOT_INITIALIZED" on every conv. Remove them and restore the cu12 libraries.
CU13=$($PIP list 2>/dev/null | awk '/^nvidia-/{print $1}' | grep -v -- "-cu12$" | grep -v "nvidia-ml-py\|nvidia-cudnn-frontend" | tr '\n' ' ')
[ -n "$CU13" ] && $PIP uninstall -y -q $CU13
$PIP install -q --force-reinstall --no-deps --index-url https://download.pytorch.org/whl/cu128 \
  nvidia-cudnn-cu12==9.10.2.21 nvidia-cusparselt-cu12==0.7.1 nvidia-nccl-cu12==2.27.5 nvidia-nvshmem-cu12==3.4.5 \
  nvidia-cublas-cu12==12.8.4.1 nvidia-cuda-cupti-cu12==12.8.90 nvidia-cuda-nvrtc-cu12==12.8.93 nvidia-cuda-runtime-cu12==12.8.90 \
  nvidia-cufft-cu12==11.3.3.83 nvidia-cufile-cu12==1.13.1.3 nvidia-curand-cu12==10.3.9.90 nvidia-cusolver-cu12==11.7.3.90 \
  nvidia-cusparse-cu12==12.5.8.93 nvidia-nvjitlink-cu12==12.8.93 nvidia-nvtx-cu12==12.8.90
$CONDA_ROOT/envs/$ENV_NAME/bin/python -c "import torch; assert torch.backends.cudnn.version() == 91002, torch.backends.cudnn.version(); print('cudnn', torch.backends.cudnn.version())"
echo "ENV CUDA LIBS OK"

# official cosmos-framework (used through PYTHONPATH, not pip-installed)
CF="$REPO/third_party/cosmos-framework"
if [[ ! -d "$CF/.git" ]]; then
  git clone https://github.com/NVIDIA/cosmos-framework.git "$CF"
fi
git -C "$CF" fetch --quiet origin || true
git -C "$CF" checkout --quiet "$COSMOS_COMMIT"
echo "cosmos-framework @ $(git -C "$CF" rev-parse --short HEAD)"
echo "DONE. Set PYTHON_BIN=$CONDA_ROOT/envs/$ENV_NAME/bin/python (env.sh default) and: source $REPO/env.sh"
