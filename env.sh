# Source from anywhere:  source /path/to/acts-world-model/env.sh
# Edit the three roots below (or export them before sourcing) to match your machine.
export ACTS_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ACTS_DATA_ROOT="${ACTS_DATA_ROOT:-$ACTS_REPO/data}"          # WitneyWW/vtm-cosmos-data snapshot
export ACTS_CKPT_ROOT="${ACTS_CKPT_ROOT:-$ACTS_REPO/checkpoints}"   # Cosmos3-Nano, Wan2.2_VAE.pth, vae/, nano_action_causal_*_v1/
export ACTS_RUNS_ROOT="${ACTS_RUNS_ROOT:-$ACTS_REPO/runs}"          # training runs + eval outputs
export VTM_SOURCE_VIDEO_ROOT="${VTM_SOURCE_VIDEO_ROOT:-$ACTS_DATA_ROOT/raw/videos}"  # mp4s for VAE anchor-frame decode
export COSMOS_FRAMEWORK_ROOT="${COSMOS_FRAMEWORK_ROOT:-$ACTS_REPO/third_party/cosmos-framework}"
export VTM_WAN22_DIR="${VTM_WAN22_DIR:-$ACTS_REPO/third_party/wan22_vae}"
export PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/acts_wm/bin/python}"  # env from scripts/setup_env.sh
export PYTHONPATH="$ACTS_REPO:$COSMOS_FRAMEWORK_ROOT:$VTM_WAN22_DIR:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
