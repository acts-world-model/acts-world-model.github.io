#!/usr/bin/env python3
"""Encode mode1_v2 tactile mp4s into **Wan2.2 VAE** latent windows.

Wan2.2 port of data_preprocessing/encode_from_mp4_cosmos.py — same sliding-window logic,
output layout and metadata, so the result drops into vm_diffusion's WindowVideoLatentDataset
(tac side). Differences from the Cosmos version:

  * Encoder = Wan2.2 causal 3D VAE + a tactile LoRA adapter (default lora_out_refflow), the
    fps6-trained tactile VAE. 48 latent channels, spatial 16x -> latent grid 256/16 = 16.
  * --fps 6: each clip is subsampled stride round(src_fps/fps)=5 from 30fps BEFORE windowing,
    matching the rate the adapter trained at (native fps would be a domain mismatch). `starts`
    are indices into the fps6-subsampled frames.
  * Frames are read with the training-consistent scale+center-crop to 256x256
    (wan22_vae.tactile_dataset.read_video_ffmpeg), NOT the Cosmos square-squash, so the LoRA
    sees in-domain input.
  * Wan2.2 is causal (valid len 1+4k): each 16-frame (8+8) window is padded to 17 by
    replicating the first frame, encoded to (48,5,16,16), then the first latent frame is
    dropped -> (48,4,16,16). The 4 latent frames each summarise 4 raw frames (matches Cosmos).

Output (per episode):
    <output_root>/tac/<episode>.pt:
        z_tactile_left/right: (Nw, 48, 4, 16, 16) fp16
        + starts/window_size/history/prediction/stride/fps/src_fps/source/num_frames/... meta

Run on a GPU (env from scripts/setup_env.sh); see preprocessing/run_preprocess.sh.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wan22_vae import Wan2_2_VAE
from wan22_vae.tactile_dataset import read_video_ffmpeg
from wan22_vae.vae_lora import inject_lora, load_lora_state_dict

_CK = Path(os.environ.get("ACTS_CKPT_ROOT", "checkpoints"))
DEFAULT_MANIFEST = Path("episodes.jsonl")
DEFAULT_OUTPUT_ROOT = Path("latents")
DEFAULT_VAE = os.environ.get("WAN22_VAE_PTH", str(_CK / "Wan2.2_VAE.pth"))       # Wan-AI/Wan2.2-TI2V-5B
DEFAULT_ADAPTER = os.environ.get("VTM_TACTILE_LORA", str(_CK / "vae" / "lora_last.pt"))  # tactile LoRA
TEMPORAL_FACTOR = 4          # Wan2.2 causal temporal compression: valid input len = 1 + 4k
SPATIAL_FACTOR = 16          # Wan2.2 spatial compression -> latent grid = resize // 16
TACTILE_KEYS = (("tactile_left", "z_tactile_left"), ("tactile_right", "z_tactile_right"))
# The visual stream is the middle scene camera. It is encoded with the BASE Wan2.2 VAE:
# the default LoRA (lora_out_refflow) was fine-tuned on tactile gel images and is out of
# domain for RGB scene video, so --adapter none is the correct setting for --modality visual.
VISUAL_KEYS = (("view_middle", "z_visual_middle"),)
# Wrist cameras (added 2026-09-10 for the val0909 set): RGB, so base VAE (--adapter none) like
# the visual stream. Written to their own wrist/ subdir so existing visual/ files stay untouched.
WRIST_KEYS = (("wrist_left", "z_wrist_left"), ("wrist_right", "z_wrist_right"))
MODALITY = {"tactile": ("tac", TACTILE_KEYS), "visual": ("visual", VISUAL_KEYS),
            "wrist": ("wrist", WRIST_KEYS)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--vae-pth", default=DEFAULT_VAE)
    p.add_argument("--adapter", default=DEFAULT_ADAPTER,
                   help="LoRA .pt to inject; 'none' for the frozen base VAE")
    p.add_argument("--modality", choices=sorted(MODALITY), default="tactile",
                   help="tactile -> tac/ (tactile_left/right, needs the tactile LoRA); "
                        "visual -> visual/ (view_middle, use --adapter none)")
    p.add_argument("--episodes", nargs="*", default=None, help="explicit episode names")
    p.add_argument("--limit", type=int, default=None, help="encode only first N episodes")
    p.add_argument("--resize", type=int, default=256, help="square-side after scale+center-crop")
    p.add_argument("--fps", type=float, default=6.0, help="subsample rate (0 = native)")
    p.add_argument("--src-fps", type=float, default=30.0)
    p.add_argument("--history", type=int, default=8)
    p.add_argument("--prediction", type=int, default=8)
    p.add_argument("--stride", type=int, default=1, help="window stride in fps-subsampled frames")
    p.add_argument("--max-windows", type=int, default=None, help="per-episode cap for smoke tests")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--storage-dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_vae(vae_pth, adapter, device, dtype):
    vae = Wan2_2_VAE(vae_pth=vae_pth, dtype=dtype, device=device)
    tag = "base"
    if adapter and adapter.lower() != "none":
        ckpt = torch.load(adapter, map_location="cpu")
        cfg = ckpt.get("config", {})
        inject_lora(vae.model, r=cfg.get("r", 16), alpha=cfg.get("alpha", 16),
                    dropout=0.0, preset=cfg.get("targets", "residual_attn"))
        vae.model.to(device)
        load_lora_state_dict(vae.model, ckpt["state_dict"], strict=True)
        tag = f"{Path(adapter).parent.name}/{Path(adapter).name} (step {ckpt.get('step', '?')})"
    vae.model.eval()
    return vae, tag


def largest_valid_len(t):
    """Largest 1+4k that is <= t (Wan2.2 causal input length)."""
    return t if (t - 1) % TEMPORAL_FACTOR == 0 else t - ((t - 1) % TEMPORAL_FACTOR)


def n_pad_for(t):
    """Frames to prepend (replicate first) so t -> 1+4k."""
    rem = (t - 1) % TEMPORAL_FACTOR
    return 0 if rem == 0 else TEMPORAL_FACTOR - rem


def make_starts(num_frames, window_size, stride):
    if num_frames < window_size:
        return torch.empty(0, dtype=torch.long)
    return torch.arange(0, num_frames - window_size + 1, stride, dtype=torch.long)


def subsample_fps(frames_thwc, src_fps, fps):
    """(T,H,W,3) uint8 -> (T6,H,W,3) keeping every round(src/fps)-th frame (fps>0)."""
    if fps and fps > 0:
        stride = max(1, int(round(src_fps / fps)))
        return frames_thwc[::stride], stride
    return frames_thwc, 1


def windows_to_batch(frames_thwc, starts_batch, window):
    """-> (B,3,window,256,256) float in [-1,1]."""
    idx = starts_batch[:, None] + np.arange(window)[None, :]      # (B, window)
    x = frames_thwc[idx]                                          # (B,window,H,W,3) uint8
    x = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 4, 1, 2, 3).contiguous().float()
    return x.div_(255.0).mul_(2.0).sub_(1.0)                      # (B,3,window,H,W) in [-1,1]


@torch.no_grad()
def encode_stream(vae, scale, frames_thwc, starts, *, window, batch_size, device, dtype,
                  storage_dtype):
    """All windows of one stream -> (Nw, 48, 4, latent_grid, latent_grid)."""
    n_pad = n_pad_for(window)                       # 16 -> 1
    n_drop = (n_pad + TEMPORAL_FACTOR - 1) // TEMPORAL_FACTOR     # 1 latent frame from padding
    starts_np = starts.numpy()
    chunks = []
    for off in range(0, len(starts_np), batch_size):
        bs = starts_np[off:off + batch_size]
        x = windows_to_batch(frames_thwc, bs, window).to(device, dtype)   # (B,3,16,256,256)
        if n_pad:
            x = torch.cat([x[:, :, :1].repeat(1, 1, n_pad, 1, 1), x], dim=2)  # -> 1+4k frames
        with torch.autocast("cuda", dtype=dtype):
            z = vae.model.encode(x, scale)          # (B,48,T'+?,g,g)
        z = z.float()[:, :, n_drop:]                # drop padded-frame latent -> (B,48,4,g,g)
        chunks.append(z.to("cpu", storage_dtype).contiguous())
        del x, z
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0)


def atomic_save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, str(tmp))
    tmp.replace(path)


def encode_episode(row, *, args, vae, scale, device, dtype, storage_dtype, adapter_tag):
    subdir, keys = MODALITY[args.modality]
    out = args.output_root / subdir / (row["episode"] + ".pt")
    if out.exists() and not args.overwrite:
        print(f"[skip] {row['episode']}: exists")
        return
    window = args.history + args.prediction
    grid = args.resize // SPATIAL_FACTOR
    t0 = time.time()

    streams = {}
    T6 = native_T = None
    for src_key, z_name in keys:
        frames = read_video_ffmpeg(row[src_key], args.resize, args.resize)  # (T,256,256,3) center-crop
        native_T = frames.shape[0]
        frames6, sub_stride = subsample_fps(frames, args.src_fps, args.fps)
        T6 = frames6.shape[0]
        starts = make_starts(T6, window, args.stride)
        if args.max_windows is not None:
            starts = starts[: args.max_windows]
        if starts.numel() == 0:
            raise ValueError(f"{row['episode']}:{src_key} has {T6} fps{args.fps:g} frames "
                             f"< window={window}")
        streams[z_name] = encode_stream(vae, scale, frames6, starts, window=window,
                                        batch_size=args.batch_size, device=device, dtype=dtype,
                                        storage_dtype=storage_dtype)
        del frames, frames6

    starts = make_starts(T6, window, args.stride)
    if args.max_windows is not None:
        starts = starts[: args.max_windows]
    meta = {
        "starts": starts.cpu(), "window_size": int(window),
        "history": int(args.history), "prediction": int(args.prediction),
        "stride": int(args.stride), "fps": float(args.fps), "src_fps": float(args.src_fps),
        "fps_stride": int(round(args.src_fps / args.fps)) if args.fps else 1,
        "num_frames": int(T6), "native_num_frames": int(native_T),
        "tactile_frame_shift": 0, "resize": int(args.resize),
        "source": str(row.get(keys[0][0], row["episode"])),
        "vae": "Wan2.2-TI2V-5B", "adapter": adapter_tag,
        "modality": args.modality,
        "visual_keys": [z for _, z in keys] if args.modality == "visual" else None,
        # explicit, so these can never again be mistaken for build_placeholder_visual.py output
        "placeholder": False,
    }
    meta.update(streams)
    atomic_save(meta, out)
    dt = time.time() - t0
    shp = tuple(streams[keys[0][1]].shape)
    names = "/".join(z for _, z in keys)
    print(f"[write] {out}  {names}={shp} grid={grid} "
          f"windows={starts.numel()} ({native_T}->{T6} fps{args.fps:g})  [{dt:.1f}s]")
    del streams
    gc.collect()


def main():
    args = parse_args()
    rows = [json.loads(l) for l in args.manifest.read_text().splitlines() if l.strip()]
    if args.episodes:
        rows = [r for r in rows if r["episode"] in set(args.episodes)]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise FileNotFoundError(f"no episodes selected from {args.manifest}")
    print(f"episodes={len(rows)} manifest={args.manifest} output={args.output_root}/tac")
    if args.dry_run:
        for r in rows:
            print(" ", r["episode"])
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if device.type == "cpu":
        dtype = torch.float32
    storage_dtype = {"float16": torch.float16, "float32": torch.float32}[args.storage_dtype]

    vae, adapter_tag = load_vae(args.vae_pth, args.adapter, device, dtype)
    scale = [s.to(device) if torch.is_tensor(s) else s for s in vae.scale]
    print(f"[vae] device={device} dtype={dtype} adapter={adapter_tag}")

    for row in rows:
        encode_episode(row, args=args, vae=vae, scale=scale, device=device, dtype=dtype,
                       storage_dtype=storage_dtype, adapter_tag=adapter_tag)
    print("done")


if __name__ == "__main__":
    main()
