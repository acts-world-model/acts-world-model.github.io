#!/usr/bin/env bash
# Train one of the released models (single node, one DDP rank per GPU).
#   bash scripts/train.sh configs/nano_action_causal_rope_v1.yaml [key=value ...]
#   NGPU=4 bash scripts/train.sh configs/nano_action_causal_pusht_v1.yaml training.epochs=1
# Under Slurm (sbatch/srun with --ntasks-per-node == #GPUs) it uses srun instead of torchrun:
#   sbatch -N1 --ntasks-per-node=4 --gres=gpu:4 scripts/train.sh configs/...yaml
# Paper runs: 4x H100 80GB, batch_size 2/GPU, 20 epochs (~2.3 h (pushT) - 3.3 h (rope) per epoch).
# Re-launching with the same run_name auto-resumes from $ACTS_RUNS_ROOT/<run_name>/last.pt.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ACTS_REPO:-$HERE}/env.sh"
cd "$ACTS_REPO"
CONFIG="${1:?usage: train.sh <config.yaml> [key=value ...]}"; shift || true
if [[ -n "${SLURM_JOB_ID:-}" && "${SLURM_NTASKS:-1}" -gt 1 ]]; then
  export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
  export MASTER_PORT=$((29500 + SLURM_JOB_ID % 35500))
  echo "[train.sh] slurm job=$SLURM_JOB_ID ntasks=$SLURM_NTASKS master=$MASTER_ADDR:$MASTER_PORT config=$CONFIG $*"
  exec srun --ntasks="$SLURM_NTASKS" --cpu-bind=none "$PYTHON_BIN" -u -m vtm_cosmos.train.train --config "$CONFIG" "$@"
fi
NGPU="${NGPU:-$("$PYTHON_BIN" -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
echo "[train.sh] torchrun nproc=$NGPU config=$CONFIG $*"
exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NGPU" \
  -m vtm_cosmos.train.train --config "$CONFIG" "$@"
