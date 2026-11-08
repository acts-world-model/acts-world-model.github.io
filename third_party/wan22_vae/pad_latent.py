"""Restore the pad latent the windowed Wan2.2 tree dropped, so a window decodes to 4T frames.

encode_tactile_wan22.py pads every 16-frame window to 17 by replicating frame 0 in front, encodes,
and DROPS the first latent (which summarises only that pad frame). Decoding the 4 stored latents as
is makes the causal decoder treat latent 0 (= raw frames 0-3) as its single-frame first chunk: 13
frames, 3 lost, and every frame decoded from a wrong cache state (view -8 dB, tactile -3.5 dB vs
raw; checkpoints/eval/pad_decode_check, 2026-09-15). The dropped latent is bit-exactly the t=1
encode of raw frame 0, so it is rebuilt here from the source mp4 named in the latent tree.

Importable from both envs (twm for training-time strips, effecterase for the eval scripts):
numpy + torch + ffmpeg only.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

STREAM_TREE = {"view": ("visual_latent_dir", "visual"), "tl": ("tactile_latent_dir", "tac"),
               "tr": ("tactile_latent_dir", "tac"), "wl": ("wrist_latent_dir", "wrist"),
               "wr": ("wrist_latent_dir", "wrist")}
STREAM_SIDE = {"tr": ("tactile_left", "tactile_right"), "wr": ("wrist_left", "wrist_right")}
_FALLBACK_FFMPEG = "/usr/bin/ffmpeg"


def remap_source(src: str, episode_key) -> str:
    """Latent metas store the ABSOLUTE mp4 path of the machine that encoded them. If that path
    machine is not this one, so when $VTM_SOURCE_VIDEO_ROOT is set the mp4 is taken from the release layout
    <VTM_SOURCE_VIDEO_ROOT>/<episode>/<stream>.mp4 (episode = latent file stem, stream = mp4 basename;
    what `scripts/download.py --what raw` puts on disk), falling back to the stored path."""
    root = os.environ.get("VTM_SOURCE_VIDEO_ROOT")
    if root:
        alt = Path(root) / Path(str(episode_key)).stem / Path(src).name
        if alt.is_file():
            return str(alt)
    return src


def _passthrough_flags() -> list:
    """`-vsync 0` (ffmpeg <= 7) == `-fps_mode passthrough` (ffmpeg >= 5.1; -vsync removed in 8)."""
    global _PT
    if _PT is None:
        out = subprocess.run([find_ffmpeg(), "-hide_banner", "-h", "long"], capture_output=True, text=True).stdout
        _PT = ["-vsync", "0"] if "-vsync" in out else ["-fps_mode", "passthrough"]
    return _PT


_PT = None


def find_ffmpeg() -> str:
    """$FFMPEG, then imageio-ffmpeg's bundled binary (setup_env.sh pins imageio-ffmpeg==0.6.0 = ffmpeg 7.0.2,
    the build the released latents/metrics were made with), then PATH."""
    exe = os.environ.get("FFMPEG")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    return _FALLBACK_FFMPEG if Path(_FALLBACK_FFMPEG).is_file() else "ffmpeg"


def latent_dirs_from_config(cfg) -> dict:
    """cfg: path to a run config.yaml, a mapping with a 'data' key, a 'data' mapping itself, or an
    attribute object (OmegaConf/SimpleNamespace cfg.data). -> {visual/tactile/wrist_latent_dir}."""
    keys = ("visual_latent_dir", "tactile_latent_dir", "wrist_latent_dir")
    if isinstance(cfg, (str, Path)):
        import yaml
        cfg = yaml.safe_load(Path(cfg).read_text())
    if isinstance(cfg, dict):
        d = cfg.get("data", cfg)
        return {k: d.get(k) for k in keys}
    d = getattr(cfg, "data", cfg)
    return {k: getattr(d, k, None) for k in keys}


def read_frames_at(mp4, native_indices, size=256):
    """Decode exactly the given 0-based frame indices of <mp4> with the SAME ffmpeg filter the
    encoder used (wan22_vae.tactile_dataset.read_video_ffmpeg: scale to cover + centre crop), in
    one sequential pass (bit-exact vs the full decode). -> {index: (size,size,3) uint8 RGB}."""
    idx = sorted(set(int(i) for i in native_indices))
    sel = "+".join(f"eq(n\\,{i})" for i in idx)
    vf = (f"select='{sel}',scale={size}:{size}:force_original_aspect_ratio=increase,"
          f"crop={size}:{size}")
    cmd = [find_ffmpeg(), "-v", "error", "-i", str(mp4), "-vf", vf, *_passthrough_flags(),
           "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {mp4}: {proc.stderr.decode(errors='ignore')[:300]}")
    fb = size * size * 3
    n = len(proc.stdout) // fb
    if n != len(idx):
        raise RuntimeError(f"{mp4}: asked {len(idx)} frames {idx[:3]}..{idx[-3:]}, got {n}")
    arr = np.frombuffer(proc.stdout[: n * fb], dtype=np.uint8).reshape(n, size, size, 3).copy()
    return {i: arr[j] for j, i in enumerate(idx)}


def frame_to_tensor(frame_u8):
    """(H,W,3) uint8 RGB -> (3,1,H,W) float in [-1,1] (one-frame clip)."""
    return torch.from_numpy(np.ascontiguousarray(frame_u8)).permute(2, 0, 1)[:, None].float().div(255).mul(2).sub(1)


class PadLatents:
    """Per (episode, stream): source mp4 from the latent tree meta, frame reads, t=1 encodes -- cached.
    ``episode_key`` is the latent file name (``motherboard_..._episode_000.pt``); ``t_start`` the
    window start in the tree's (fps-subsampled) frame index, mapped to native via ``fps_stride``."""

    def __init__(self, latent_dirs: dict, size: int = 256):
        self.dirs, self.size = dict(latent_dirs), int(size)
        self._meta, self._frames, self._z = {}, {}, {}

    def source(self, episode_key, st):
        key = (str(episode_key), st)
        if key not in self._meta:
            dkey, _ = STREAM_TREE[st]
            root = self.dirs.get(dkey)
            if not root:
                raise RuntimeError(f"no data.{dkey} configured; needed to rebuild the pad latent of stream {st}")
            meta = torch.load(Path(root) / str(episode_key), map_location="cpu", weights_only=False, mmap=True)
            src, fs = str(meta["source"]), int(meta.get("fps_stride", 1))
            if st in STREAM_SIDE:
                a, b = STREAM_SIDE[st]
                if a not in src:
                    raise RuntimeError(f"{st}: expected '{a}' in latent source {src}")
                src = src.replace(a, b)
            if st == "view":
                src = src.replace("view_left", "view_middle").replace("view_right", "view_middle")
            src = remap_source(src, episode_key)
            if not Path(src).is_file():
                raise FileNotFoundError(f"{st} source video for {episode_key}: {src} "
                                        "(set VTM_SOURCE_VIDEO_ROOT to the downloaded raw/videos dir)")
            self._meta[key] = (src, fs)
        return self._meta[key]

    def prefetch(self, samples_meta):
        """samples_meta: iterable of (episode_key, t_start, [streams]) -> one ffmpeg pass per source."""
        need = {}
        for ep, t0, sts in samples_meta:
            for st in sts:
                src, fs = self.source(ep, st)
                need.setdefault(src, set()).add(int(t0) * fs)
        for src, idxs in need.items():
            missing = [i for i in idxs if (src, i) not in self._frames]
            if not missing:
                continue
            for i, fr in read_frames_at(src, missing, self.size).items():
                self._frames[(src, i)] = fr
            print(f"[pad] {Path(src).parent.name}/{Path(src).name}: read {len(missing)} frame(s)", flush=True)

    def frame0(self, episode_key, st, t_start):
        src, fs = self.source(episode_key, st)
        k = (src, int(t_start) * fs)
        if k not in self._frames:
            self.prefetch([(episode_key, t_start, [st])])
        return self._frames[k]

    def window(self, episode_key, st, t_start, n=16):
        """Raw (n, size, size, 3) window frames at the tree's stride (verification only)."""
        src, fs = self.source(episode_key, st)
        idx = [int(t_start) * fs + j * fs for j in range(n)]
        fr = read_frames_at(src, idx, self.size)
        return np.stack([fr[i] for i in idx])

    @torch.no_grad()
    def latent(self, vae, episode_key, st, t_start, device, dtype=torch.bfloat16):
        """(C,1,g,g) float cpu = t=1 encode of raw frame 0 of the window. Cached per (vae, ep, st, t).
        Use the SAME vae instance the stream is decoded with (LoRA for the gels, base otherwise)."""
        k = (id(vae), str(episode_key), st, int(t_start))
        if k not in self._z:
            x = frame_to_tensor(self.frame0(episode_key, st, t_start))
            self._z[k] = vae.encode([x.to(device, dtype)])[0].float().cpu()
        return self._z[k]


@torch.no_grad()
def decode_padded(vae, z, pad, device, dtype=torch.bfloat16):
    """(C,T,g,g) stored latents + (C,1,g,g) pad -> (3, 4T, H, W) float cpu in [-1,1]."""
    zz = torch.cat([pad.to(z.device, z.dtype), z], dim=1).to(device, dtype)
    px = vae.decode([zz])[0].float().cpu()
    if px.shape[1] != 1 + 4 * z.shape[1]:
        raise RuntimeError(f"decode returned {px.shape[1]} frames for {z.shape[1]}+1 latents")
    return px[:, 1:]
