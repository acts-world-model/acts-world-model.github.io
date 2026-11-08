"""Wan2.2 VAE decode with anchor restoration (spec §9.2) and re-encode for reference rollouts.

Uses ``third_party/wan22_vae`` (override with $VTM_WAN22_DIR): ``wan22_vae.Wan2_2_VAE`` (+ tactile LoRA via
``vae_lora``) and ``pad_latent.PadLatents`` which rebuilds the dropped anchor latent
from raw frame 0 of the window (bit-exact with the pad-17 encode's chunk 0)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from vtm_cosmos.streams import PAD_STREAM_KEY, TACTILE_STREAMS

_REPO = Path(__file__).resolve().parents[2]
_CKPT = Path(os.environ.get("ACTS_CKPT_ROOT", _REPO / "checkpoints"))
WAN22_VAE_DIR = Path(os.environ.get("VTM_WAN22_DIR", _REPO / "third_party" / "wan22_vae"))
# Wan-AI/Wan2.2-TI2V-5B :: Wan2.2_VAE.pth (base VAE, all streams)
DEFAULT_VAE_PTH = os.environ.get("WAN22_VAE_PTH", str(_CKPT / "Wan2.2_VAE.pth"))
# tactile LoRA of the VAE (WitneyWW/vtm-cosmos-nano-ckpts :: vae/lora_last.pt)
DEFAULT_TACTILE_LORA = os.environ.get("VTM_TACTILE_LORA", str(_CKPT / "vae" / "lora_last.pt"))


def _import_wan22():
    if str(WAN22_VAE_DIR) not in sys.path:
        sys.path.insert(0, str(WAN22_VAE_DIR))
    from wan22_vae import Wan2_2_VAE  # noqa
    from wan22_vae.vae_lora import inject_lora, load_lora_state_dict  # noqa
    import pad_latent  # noqa

    return Wan2_2_VAE, inject_lora, load_lora_state_dict, pad_latent


def fresh_vae(vae_pth: str, device, dtype, adapter: str | None = None):
    Wan2_2_VAE, inject_lora, load_lora_state_dict, _ = _import_wan22()
    vae = Wan2_2_VAE(vae_pth=vae_pth, dtype=dtype, device=device)
    if adapter is None:
        return vae
    ckpt = torch.load(adapter, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    inject_lora(vae.model, r=cfg.get("r", 16), alpha=cfg.get("alpha", 16), dropout=0.0, preset=cfg.get("targets", "residual_attn"))
    vae.model.to(device)
    load_lora_state_dict(vae.model, ckpt["state_dict"], strict=True)
    return vae


class WanDecoder:
    def __init__(
        self,
        latent_root: str | Path,
        device,
        dtype=torch.bfloat16,
        vae_pth: str = DEFAULT_VAE_PTH,
        tactile_adapter: str | None = DEFAULT_TACTILE_LORA,
        size: int = 256,
    ):
        _, _, _, pad_latent = _import_wan22()
        root = Path(latent_root)
        self.device, self.dtype = device, dtype
        self.base = fresh_vae(vae_pth, device, dtype, None)
        self.tac = fresh_vae(vae_pth, device, dtype, tactile_adapter) if tactile_adapter else self.base
        self.pads = pad_latent.PadLatents(
            {"visual_latent_dir": str(root / "visual"), "tactile_latent_dir": str(root / "tac"), "wrist_latent_dir": str(root / "wrist")},
            size=size,
        )
        self._decode_padded = pad_latent.decode_padded

    def vae_for(self, stream: str):
        return self.tac if stream in TACTILE_STREAMS else self.base

    @torch.no_grad()
    def anchor(self, stream: str, episode_key: str, t_start: int) -> torch.Tensor:
        """(48,1,16,16) cpu float: the dropped pad latent (frame 0 encoded alone)."""
        return self.pads.latent(self.vae_for(stream), episode_key, PAD_STREAM_KEY[stream], int(t_start), self.device, self.dtype)

    @torch.no_grad()
    def decode_window(self, stream: str, episode_key: str, t_start: int, z4: torch.Tensor) -> torch.Tensor:
        """z4 (48,4,16,16) -> (3,16,H,W) float cpu in [-1,1] (anchor restored, pad frame dropped)."""
        pad = self.anchor(stream, episode_key, t_start)
        return self._decode_padded(self.vae_for(stream), z4.float().cpu(), pad, self.device, self.dtype)

    @torch.no_grad()
    def decode_with_anchor(self, stream: str, anchor: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Explicit anchor (48,1,g,g) + z (48,T,g,g) -> (3,4T,H,W); used by free-running rollouts."""
        return self._decode_padded(self.vae_for(stream), z.float().cpu(), anchor, self.device, self.dtype)

    @torch.no_grad()
    def encode_frames(self, stream: str, frames: torch.Tensor) -> torch.Tensor:
        """frames (3,T,H,W) in [-1,1] with T = 4k+1 -> (48, k+1, g, g) float cpu (index 0 = anchor)."""
        if (frames.shape[1] - 1) % 4 != 0:
            raise ValueError(f"need 4k+1 frames, got {frames.shape[1]}")
        z = self.vae_for(stream).encode([frames.to(self.device, self.dtype)])[0]
        return z.float().cpu()

    def raw_window(self, stream: str, episode_key: str, t_start: int, n: int = 16):
        """Raw uint8 (n,H,W,3) frames at the tree stride (for fidelity checks)."""
        return self.pads.window(episode_key, PAD_STREAM_KEY[stream], int(t_start), n)
