# Reproducibility checks (2026-11-08)

Run on a clean directory: code downloaded from `WitneyWW/acts-world-model`, data / checkpoints downloaded from the Hub with
`scripts/download.py`, a fresh conda env created by `scripts/setup_env.sh` (`acts_wm`: torch 2.10.0+cu128, numpy 2.4,
imageio-ffmpeg 0.6.0 = ffmpeg 7.0.2), cosmos-framework @ 3f94d25, 1× H100 80 GB per job.

| check | what | result |
|---|---|---|
| inference, short | `eval/run_test_case.py` cases 0-3 of every task (12 cases, 8 → 8 frames) | PSNR/SSIM/LPIPS identical to the reference numbers (Δ = 0.000 dB on every stream) |
| inference, long | case 0 of every task, 15 chained windows (120 frames) | per-window metrics identical to the reference (Δ = 0.000 dB) |
| training | `NGPU=1 scripts/train.sh configs/nano_action_causal_rope_v1.yaml training.max_steps=30` from Cosmos3-Nano | runs, loss 5.2 → 4.1, val pass OK, 30.5 GiB peak |
| latents | `preprocessing/run_preprocess.sh` on one episode per task (motherboard_0911_episode_004_seg01, rope_0914_episode_003_seg03, pushT_0915_episode_000_seg00) from `raw/` | tac / visual / wrist latents **bit-identical** to the released files |
| actions | all 6 action files of all 166 split episodes rebuilt from `raw/parquet` (original env) | 996 files bit-identical |
| actions (new env) | the 3 episodes above in `acts_wm` | `action` / `valid` / `frame_idx` bit-identical; diagnostic `rot_deg` differs by 1.5e-8 deg (float32 ulp, numpy version) |

Notes
* Bit-exact decoding depends on the ffmpeg build: frames are read via `imageio-ffmpeg` (pinned to 0.6.0); `$FFMPEG` overrides.
  Newer ffmpeg (≥ 8) is supported (`-fps_mode passthrough`), but frames may differ in the last bit of the scaler output.
* The 32 unused 05-10/11/19 motherboard episodes in `mixed_mb_latents` and the `sensor_actions_fps15_noforce.pt` ablation
  files are not part of this release.
