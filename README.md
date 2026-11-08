---
license: other
tags:
- world-model
- robotics
- tactile
- cosmos
---

# ACTS world model — code

Code for the three action-conditioned visuo-tactile world models trained on the React bimanual dataset:

| run | task | ckpt (HF) | train / val / test episodes | train windows | steps (4×H100, bs 2/GPU) |
|---|---|---|---|---:|---:|
| `nano_action_causal_umi_all_v1` | motherboard (UMI-style) | `nano_action_causal_umi_all_v1/epoch_019.pt` | 21 / 2 / 5 | 71,386 | 178,460 |
| `nano_action_causal_rope_v1` | rope | `nano_action_causal_rope_v1/epoch_019.pt` | 53 / 8 / 9 | 77,188 | 192,960 |
| `nano_action_causal_pusht_v1` | pushT | `nano_action_causal_pusht_v1/epoch_019.pt` | 56 / 4 / 8 | 55,199 | 138,000 |

All three use one recipe (only the data paths differ, see `configs/`):

* **Backbone**: NVIDIA **Cosmos3-Nano** Generator (Qwen3-VL-8B MoT, 36 layers, D = 4096), frozen; only the GEN tower is used.
* **Trainable** (45.3 M, 0.30 %): LoRA r16/α32 on attention+FFN of the GEN tower, tactile in/out projections (initialised from
  the RGB `vae2llm/llm2vae`), two 9-D action projections (initialised from Nano's UMI action head), a stream embedding.
* **Streams**: 5 image latent streams — main camera `video` (view_middle), `wrist_left/right`, GelSight `tactile_left/right`
  (Wan2.2 VAE latents `[48,4,16,16]` per 16-frame window, tactile with a GelSight LoRA of the VAE) — plus 4 action streams:
  per-hand 9-D sensor action `[dp(3) + k_f·Δf·n, rot6d(6)]` (force enters as the change of normal force) and per-hand 9-D view action.
* **Window**: 15 fps, 8 observed + 8 predicted frames = 2 + 2 latent blocks.
* **Objective**: flow matching on future blocks with **causal teacher forcing** (`[H0,H1,~F0]` → loss on F0, `[H0,H1,F0,~F1]` → loss on F1);
  attention mask = same sample ∧ `key_tick ≤ query_tick` over all streams (FlexAttention).
* **Inference**: causal autoregressive, UniPC 35 steps per latent block; long horizon = 15 chained windows (120 generated frames, 8 s),
  each window conditioned on the recorded actions and on the model's own previous output.

Companion Hugging Face repos:

| what | repo |
|---|---|
| this code | `WitneyWW/acts-world-model` (also on GitHub: `acts-world-model/acts-world-model.github.io`, branch `code`) |
| preprocessed latents + actions, raw inputs, splits | dataset `WitneyWW/vtm-cosmos-data` |
| trained checkpoints (+ tactile VAE LoRA) | model `WitneyWW/vtm-cosmos-nano-ckpts` |
| base model | `nvidia/Cosmos3-Nano` |
| VAE | `Wan-AI/Wan2.2-TI2V-5B` (`Wan2.2_VAE.pth`) |

## Repository layout

```
env.sh                    paths + PYTHONPATH; source this before anything else
configs/                  the exact training configs of the 3 runs (machine paths -> ${oc.env:...})
splits/                   train/val/test episode lists (mixed_{mb,rope,pusht}_wrist_holdout.json)
vtm_cosmos/               model / data / training / inference package
  model/                    Cosmos3 Generator adapter, token layout, causal mask, attention dispatch, Nano weight loading
  data/                     window dataset, action loading + quantile normaliser
  train/                    train.py (DDP), causal teacher-forcing loss, noising
  infer/                    sampler (UniPC), rollout.py (short + long rollouts, metrics), VAE decode
eval/
  run_test_case.py          run the fixed test cases (short + 15-window long rollout) and render GT|PRED videos
  summarize.py              PSNR/SSIM/LPIPS tables (+ paired diff vs the shipped reference numbers)
  cases/                    <task>_test_cases.json (84 cases total) + <task>_reference_metrics.json
scripts/
  setup_env.sh              conda env (torch 2.10 + cu128) + clone cosmos-framework @ 3f94d25
  download.py               fetch data / raw / ckpts / Cosmos3-Nano / Wan2.2 VAE
  train.sh                  torchrun (single node) or srun (Slurm) launcher
  select_contact_windows.py contact-stratified window selection used to build eval/cases
preprocessing/            raw mp4 + parquet -> latents + actions
  run_preprocess.sh         end-to-end for one task
  make_manifest.py          encoder manifest from raw/
  vm_diffusion/data/        build_force_actions.py (dforce / force 9-D actions), build_pose7d_force.py
  vm_diffusion/model/       pose7d_action.py (SE(3) + force action maths)
  data_preprocessing/       build_view_actions.py, project_pose_to_camera.py, build_episode_actions.py
  common/                   rotation helpers, gel frame
  data/<task>/calibration/  camera / gel calibration used for the view actions
third_party/wan22_vae/    standalone Wan2.2 VAE + VAE LoRA + encoder (encode_tactile_wan22.py) + pad_latent.py
third_party/cosmos-framework/   (created by setup_env.sh; NVIDIA/cosmos-framework, unmodified)
```

## 1. Setup

Tested on Linux, NVIDIA H100 80 GB, driver with CUDA 12.8 support, conda.

```bash
git clone https://huggingface.co/WitneyWW/acts-world-model     # or the GitHub `code` branch
cd acts-world-model
bash scripts/setup_env.sh          # creates conda env `acts_wm` (override: ENV_NAME=..., CONDA_ROOT=...)
                                   # and clones NVIDIA/cosmos-framework @ 3f94d25 into third_party/
```

Edit the roots at the top of `env.sh` (or export them before sourcing) — defaults are inside the repo:

| variable | default | content |
|---|---|---|
| `ACTS_DATA_ROOT` | `./data` | snapshot of `WitneyWW/vtm-cosmos-data` |
| `ACTS_CKPT_ROOT` | `./checkpoints` | `Cosmos3-Nano/`, `Wan2.2_VAE.pth`, `vae/lora_last.pt`, `nano_action_causal_*_v1/epoch_019.pt` |
| `ACTS_RUNS_ROOT` | `./runs` | training runs and eval outputs |
| `PYTHON_BIN` | `~/miniconda3/envs/acts_wm/bin/python` | interpreter of the env |

```bash
source env.sh
export HF_TOKEN=hf_...             # or `hf auth login`
$PYTHON_BIN scripts/download.py --what all                  # everything, ~165 GB
# or only what you need, e.g. evaluating the rope model on its test split (~40 GB, mostly Cosmos3-Nano):
$PYTHON_BIN scripts/download.py --what data raw --tasks rope --splits test
$PYTHON_BIN scripts/download.py --what ckpts nano wan_vae --tasks rope
```

Sizes: Cosmos3-Nano 29 GB, Wan2.2 VAE 2.8 GB, one task checkpoint 0.55 GB; latents+actions of the split episodes ≈ 40 GB (motherboard),
31 GB (pushT), 43 GB (rope); raw mp4+parquet for all 166 split episodes 16 GB (≈ 165 GB in total).

> `raw/videos` is needed for **evaluation** too, not only for preprocessing: latents drop the Wan2.2 "anchor" latent of each
> window, and decoding rebuilds it from frame 0 of the source mp4 (`third_party/wan22_vae/pad_latent.py`). The mp4 paths stored
> in the latent files point at the original cluster; `VTM_SOURCE_VIDEO_ROOT` (set by `env.sh`) redirects them to
> `$ACTS_DATA_ROOT/raw/videos/<episode>/<stream>.mp4`.

## 2. Inference / evaluation

The paper test protocol: per task a fixed set of contact-stratified cases (`none` / `onset` / `release` / `full`, defined from the
GelSight normal force, `max(F_L, F_R) > 0.5 N`); each case = one short prediction (8 observed → 8 predicted frames) and one long
autoregressive rollout of 15 windows (120 frames) from the same start, conditioned on the recorded actions; seed 0, UniPC 35.

```bash
source env.sh
$PYTHON_BIN eval/run_test_case.py --task rope --case 0                 # one case (~7 min on H100; ~40 s with --short-only)
$PYTHON_BIN eval/run_test_case.py --task rope --case all               # 34 cases, sequential, resumable
$PYTHON_BIN eval/summarize.py --task rope --reference                  # table + paired diff vs reference numbers
```

`--task` ∈ `umi_all` (20 cases), `rope` (34), `pusht` (30). Outputs per case in
`$ACTS_RUNS_ROOT/<run>/test_epoch_019/<NN_episode_class>/`: `window_eval.json` (short metrics), `reference_rollout.json`
(per-window long metrics), `*_pred.pt` (predicted latents), GT/PRED strip PNGs, `short_gt_pred.mp4`, `long_gt_pred.mp4`.
Cases are independent, so on a cluster run them as an array job, e.g. `sbatch --array=0-33 --gres=gpu:1 --wrap "source env.sh; $PYTHON_BIN eval/run_test_case.py --task rope --case \$SLURM_ARRAY_TASK_ID"`.

Arbitrary windows (any split) with the lower-level CLI:

```bash
$PYTHON_BIN -m vtm_cosmos.infer.rollout --ckpt $ACTS_CKPT_ROOT/nano_action_causal_rope_v1/epoch_019.pt \
  --split test --window-stride 1 --windows 0 500 --rollout-windows 15 --steps 35 --solver unipc --seed 0 \
  --out-dir runs/my_eval --override data.latent_root=$ACTS_DATA_ROOT/mixed_rope_latents \
  data.action_root=$ACTS_DATA_ROOT/mixed_rope_actions data.split_file=splits/mixed_rope_wrist_holdout.json \
  model.nano_checkpoint=$ACTS_CKPT_ROOT/Cosmos3-Nano
```

(The checkpoint stores the training config with the original absolute paths, hence the `--override`s; the action normaliser is
stored in the checkpoint. `--windows` are indices into the window list of the split after invalid-action windows are dropped;
with `--window-stride 1` the first episode's index equals `t_start`.)

### Reference results (epoch 019, test split)

| task | n | horizon | metric | video | wrist_left | wrist_right | tactile_left | tactile_right |
|---|---:|---|---|---:|---:|---:|---:|---:|
| rope | 34 | short | PSNR ↑ | 28.92 | 22.30 | 23.70 | 46.08 | 46.51 |
| rope | 34 | long | PSNR ↑ | 23.68 | 16.40 | 18.22 | 38.26 | 40.09 |
| pushT | 30 | short | PSNR ↑ | 29.68 | 22.86 | 22.76 | 46.43 | 48.37 |
| pushT | 28 | long | PSNR ↑ | 23.28 | 14.77 | 15.50 | 39.21 | 39.65 |
| motherboard | 20 | short | PSNR ↑ | 27.74 | 17.68 | 20.77 | 41.65 | 45.70 |
| motherboard | 20 | long | PSNR ↑ | 21.26 | 14.08 | 15.02 | 34.53 | 35.98 |

SSIM / LPIPS and per-case numbers: `eval/cases/<task>_reference_metrics.json`. pushT cases 5 and 29 are short-only (segment too
short for 15 windows). 12 of the 20 motherboard cases come from the 09-09 session (pre-v8 force calibration, out of distribution).

## 3. Training

```bash
source env.sh
bash scripts/train.sh configs/nano_action_causal_rope_v1.yaml                     # all local GPUs (torchrun)
NGPU=1 bash scripts/train.sh configs/nano_action_causal_rope_v1.yaml \
     run_name=smoke training.max_steps=20                                          # quick smoke test
sbatch -N1 --ntasks-per-node=4 --gres=gpu:4 scripts/train.sh configs/nano_action_causal_pusht_v1.yaml  # Slurm
```

Any `key=value` after the config is an OmegaConf dot-list override. The paper runs used 4× H100 80 GB, batch 2 per GPU
(global 8), AdamW lr 5e-5 (constant after 100 warm-up steps), wd 0.01, bf16, activation checkpointing, 20 epochs
(≈ 1.2 s/step, ~30 GiB per GPU; 2.3 h/epoch pushT … 3.3 h/epoch rope). The run directory gets `config.yaml`, `train.log`,
`metrics.jsonl`, `action_normalizer.json` (quantile normaliser fitted on the train split), `epoch_XXX.pt`, `last.pt`, `best.pt`
(lowest val flow loss; for these runs best = last = epoch 19). Re-launching with the same `run_name` resumes from `last.pt`.
Initial weights: Cosmos3-Nano (`$ACTS_CKPT_ROOT/Cosmos3-Nano`), loaded strictly (804 tensors); UND tower stays frozen.

## 4. Preprocessing (raw → latents + actions)

The released `mixed_*_latents` / `mixed_*_actions` were produced by this pipeline; it is only needed to rebuild or extend them.

```bash
source env.sh
$PYTHON_BIN scripts/download.py --what raw ckpts wan_vae --tasks rope
bash preprocessing/run_preprocess.sh rope                       # -> $ACTS_DATA_ROOT/rebuilt/mixed_rope_{latents,actions}
bash preprocessing/run_preprocess.sh rope /out/dir rope_0911_episode_004   # subset
```

Steps (`preprocessing/run_preprocess.sh`):

1. **Latents** (`third_party/wan22_vae/encode_tactile_wan22.py`, 1 GPU): 30 → 15 fps (every 2nd frame), 256×256 scale + centre crop,
   sliding 16-frame windows with stride 1, each window padded to 17 frames by repeating frame 0, encoded with the Wan2.2 VAE, the first
   (anchor) latent dropped → `(48,4,16,16)` fp16 per window. `tactile_left/right` use the GelSight LoRA (`vae/lora_last.pt`),
   `view_middle` and `wrist_left/right` the base VAE. Output `<tree>_latents/{tac,visual,wrist}/<episode>.pt`.
2. **Sensor actions** (`build_force_actions.py --force-term delta|absolute`, `build_pose7d_force.py`): per hand, row k = frame k→k+1 at
   15 fps, `dp = R_kᵀ(p_{k+1} − p_k) + k_f·(f_{k+1} − f_k)·n` (n = (0,−1,0), k_f = 5e-4 m/N, i.e. 2 N/mm gel stiffness) and the
   first two columns of `R_kᵀR_{k+1}` → `sensor_actions_with_dforce_fps15.pt` (used for training);
   `sensor_pose7d_force_fps15.pt` = absolute pose + normal force per frame (used for the contact classes).
3. **View actions** (`build_view_actions.py --stride 2`): gel centre / normal / up points in the mocap frame and their projections into
   the calibrated cameras → `view_actions_fps15.pt`.

Verification done for this release: rebuilding the actions of all 166 split episodes from `raw/parquet` reproduces the released action
files bit-exactly (996 files); see the bottom of this README for the latent check.

## 5. Data format (`WitneyWW/vtm-cosmos-data`)

```
mixed_{mb,rope,pushT}_latents/{tac,visual,wrist}/<episode>.pt
    z_tactile_left/right | z_visual_middle | z_wrist_left/right : (N_windows, 48, 4, 16, 16) fp16
    starts (window start frame @15 fps), fps=15, src_fps=30, fps_stride=2, history=8, prediction=8, num_frames, native_num_frames, source, adapter
mixed_{mb,rope,pushT}_actions/<episode>/
    sensor_actions_with_dforce_fps15.pt   per hand (num_frames-1, 9) + valid / rot_deg / frame_idx / meta   <- training input
    view_actions_fps15.pt                 per hand per frame view action (gel points + projections)          <- training input
    sensor_pose7d_force_fps15.pt          per hand (num_frames, 7 pose + 1 force N)                          <- contact classes
    sensor_actions_with_{dforce,force}{,_fps15}.pt                                                         (30 fps / absolute-force twins)
raw/videos/<episode>/{view_middle,tactile_left,tactile_right,wrist_left,wrist_right}.mp4      30 fps source videos
raw/parquet/<episode>.parquet   per-native-frame sensor poses (OptiTrack) + calibrated normal force
raw/episodes.json               [{episode, tree, num_frames}]
splits/*.json                   same files as ./splits
```

Episode names: `motherboard_MMDD_episode_XXX[_segYY]`, `rope_MMDD_...`, `pushT_MMDD_...` (MMDD = 2026 recording date).
`mixed_mb_latents` also contains 32 older episodes (05-10/11/19) that are not in any split (no v8 force) and are unused.
Raw inputs come from the React dataset (HF `yxma/React`): motherboard at revision 990a6787, pushT 360392a1, rope d8990e0a,
validation 09-09 session; `raw/` freezes exactly the files the release was built from.

## Notes and pitfalls

* Run everything with `env.sh` sourced: it sets `PYTHONNOUSERSITE=1` and puts `third_party/cosmos-framework` and
  `third_party/wan22_vae` on `PYTHONPATH` (cosmos-framework is used unmodified via `PYTHONPATH`, not pip-installed).
* `setup_env.sh` pins torch 2.10.0+cu128; some PyPI deps try to upgrade torch to a cu130 build, which older drivers reject, and leave
  `nvidia-*-cu13` libraries behind that break cuDNN convolutions (`CUDNN_STATUS_NOT_INITIALIZED` in VAE decode). The script removes them
  and re-pins; if you install extra packages, re-run its last block.
* Attention uses FlexAttention compiled with Triton; the first step compiles kernels (a few minutes).
* ffmpeg: frames are read through the ffmpeg CLI (`$FFMPEG`, else `PATH`, else the `imageio-ffmpeg` binary).
* `rollout.py --num-windows N` takes the first N windows of the split (not random); use `--windows` for explicit indices.
* Training-time `best.pt` is selected on val flow loss, which is only a monitor; the reported models are the last epoch (019).

## Reproducibility checks performed for this release

On a clean directory (code + data + checkpoints from the Hub, fresh env from `setup_env.sh`): test-case inference reproduces
the reference metrics exactly (short and 15-window long rollouts), re-encoding latents from `raw/` is bit-identical, rebuilding
the actions of all 166 split episodes is bit-identical, and a 30-step training smoke run works. Details: `REPRO_CHECKS.md`.
