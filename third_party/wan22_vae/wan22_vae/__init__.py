"""Standalone Wan2.2 VAE architecture.

Extracted verbatim from the Wan2.2 repo (``wan/modules/vae2_2.py``). The module
is self-contained — it depends only on ``torch`` and ``einops``.

Public API:
    Wan2_2_VAE   -- high-level wrapper (loads Wan2.2_VAE.pth, encode/decode lists)
    WanVAE_      -- the raw nn.Module (encoder + decoder), if you need the graph
"""
from .model import Wan2_2_VAE, WanVAE_

__all__ = ["Wan2_2_VAE", "WanVAE_"]
