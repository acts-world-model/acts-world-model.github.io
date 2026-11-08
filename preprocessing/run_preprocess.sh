#!/usr/bin/env bash
# Rebuild the training inputs of one task from the raw mp4/parquet files:
#   raw/ (download.py --what raw)  ->  <OUT>/<tree>_latents/{tac,visual,wrist}/<ep>.pt
#                                  ->  <OUT>/<tree>_actions/<ep>/{sensor_actions_with_dforce_fps15.pt,
#                                        sensor_pose7d_force_fps15.pt, view_actions_fps15.pt, ...}
#   bash preprocessing/run_preprocess.sh rope [OUT] [episode ...]
# Default OUT = $ACTS_DATA_ROOT/rebuilt (the released trees in $ACTS_DATA_ROOT are not touched).
# Point training at the result with data.latent_root=... data.action_root=... overrides.
# Recipe (identical to the released data): 30 -> 15 fps (stride 2), 8+8-frame windows, stride 1,
# 256x256 scale+center-crop; tactile = Wan2.2 VAE + tactile LoRA, view_middle + wrist = base VAE.
# Encoding needs 1 GPU (~10-40 min per episode group); action building is CPU-only.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ACTS_REPO:-$HERE}/env.sh"
TASK="${1:?usage: run_preprocess.sh <umi_all|rope|pusht> [OUT] [episode ...]}"
OUT="${2:-$ACTS_DATA_ROOT/rebuilt}"; shift $(( $# >= 2 ? 2 : 1 ))
declare -A TREE=([umi_all]=mixed_mb [rope]=mixed_rope [pusht]=mixed_pushT)
T="${TREE[$TASK]}"; RAW="$ACTS_DATA_ROOT/raw"
VAE="${WAN22_VAE_PTH:-$ACTS_CKPT_ROOT/Wan2.2_VAE.pth}"; LORA="${VTM_TACTILE_LORA:-$ACTS_CKPT_ROOT/vae/lora_last.pt}"
LAT="$OUT/${T}_latents"; ACT="$OUT/${T}_actions"; MAN="$OUT/episodes_${TASK}.jsonl"
mkdir -p "$LAT" "$ACT"

echo "=== [1/3] manifest"
"$PYTHON_BIN" "$ACTS_REPO/preprocessing/make_manifest.py" --raw-root "$RAW" --task "$TASK" --out "$MAN" ${@:+--episodes "$@"}
EPS=$("$PYTHON_BIN" -c "import json,sys; print(' '.join(json.loads(l)['episode'] for l in open(sys.argv[1]) if l.strip()))" "$MAN")

echo "=== [2/3] Wan2.2 latents (GPU)"
COMMON=(--vae-pth "$VAE" --fps 15 --src-fps 30 --history 8 --prediction 8 --stride 1 --resize 256
        --manifest "$MAN" --output-root "$LAT" --episodes $EPS)
cd "$VTM_WAN22_DIR"
"$PYTHON_BIN" -u encode_tactile_wan22.py --modality tactile --adapter "$LORA" "${COMMON[@]}"
"$PYTHON_BIN" -u encode_tactile_wan22.py --modality visual  --adapter none    "${COMMON[@]}"
"$PYTHON_BIN" -u encode_tactile_wan22.py --modality wrist   --adapter none    "${COMMON[@]}"

echo "=== [3/3] actions (CPU)"
export ACTS_PARQUET_ROOT="$RAW/parquet"
P="$ACTS_REPO/preprocessing"
export PYTHONPATH="$P:$P/data_preprocessing:$PYTHONPATH"
for ep in $EPS; do mkdir -p "$ACT/$ep"; done
AARGS=(--parquet-root "$RAW/parquet" --videos-root "$ACT" --latent-dir "$LAT/tac" --target-fps 15 --episodes $EPS)
# order matters: the two force twins first, pose7d cross-checks its valid mask against them
"$PYTHON_BIN" -u "$P/vm_diffusion/data/build_force_actions.py" --force-term delta    "${AARGS[@]}"
"$PYTHON_BIN" -u "$P/vm_diffusion/data/build_force_actions.py" --force-term absolute "${AARGS[@]}"
"$PYTHON_BIN" -u "$P/vm_diffusion/data/build_pose7d_force.py" "${AARGS[@]}"
# camera-frame view actions (gel centre / normal / up points + projections), 15 fps
"$PYTHON_BIN" -u "$P/data_preprocessing/build_view_actions.py" --videos-root "$ACT" --episodes $EPS \
    --stride 2 --out-name view_actions_fps15.pt
echo "PREPROCESS DONE -> $LAT  $ACT"
echo "train on it:  bash scripts/train.sh configs/nano_action_causal_${TASK}_v1.yaml data.latent_root=$LAT data.action_root=$ACT"
