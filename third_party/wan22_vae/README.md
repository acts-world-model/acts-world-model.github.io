# Wan2.2 VAE (standalone)

The Wan2.2 video VAE architecture, extracted verbatim from the
[Wan2.2](https://github.com/Wan-Video/Wan2.2) repo
(`wan/modules/vae2_2.py`) into a dependency-free package. It depends only on
`torch` and `einops` — no `easydict`, no rest of the Wan codebase.

## What it is

A **causal 3D VAE** for video, used by Wan2.2 TI2V-5B:

- **Compression**: `vae_stride = (4, 16, 16)` — 4× temporal, 16× spatial each
  dim. A `[3, T, H, W]` clip encodes to `[48, 1+(T-1)//4, H/16, W/16]`.
- **Latent channels**: `z_dim = 48`.
- **Causal in time**: `CausalConv3d` + a chunked feat-cache so encode/decode
  stream frame-chunks left-to-right (no lookahead), enabling long videos.
- **Patchify 2×** at the pixel boundary (12 = 3·2·2 input channels to the conv
  stack), attention in the bottleneck, RMSNorm throughout.
- Per-channel `mean`/`std` normalization baked into the wrapper's `scale`.

Architecture defaults (from `Wan2_2_VAE.__init__`):

| param | value |
|-------|-------|
| `z_dim` | 48 |
| `c_dim` (encoder base dim) | 160 |
| `dec_dim` | 256 |
| `dim_mult` | `[1, 2, 4, 4]` |
| `num_res_blocks` | 2 |
| `temperal_downsample` | `[False, True, True]` |

## Weights

Not included (2.8 GB). Download `Wan2.2_VAE.pth` from
[Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B).
On this machine it is already at:

```
/group2/ct/weihanx/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
```

## Install

```bash
pip install -r requirements.txt   # torch, einops
```

## Usage

```python
import torch
from wan22_vae import Wan2_2_VAE

vae = Wan2_2_VAE(
    vae_pth="/group2/ct/weihanx/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
    dtype=torch.bfloat16,
    device="cuda",
)

video = torch.rand(3, 17, 256, 256, device="cuda") * 2 - 1  # [C, T, H, W] in [-1,1]
latents = vae.encode([video])   # list -> list; latents[0]: [48, 5, 16, 16]
recon   = vae.decode(latents)   # list -> list; recon[0]:   [3, 17, 256, 256]
```

`encode`/`decode` take and return **lists** (one entry per clip) and run under
autocast at `dtype`. The raw `nn.Module` is exposed as `WanVAE_` /
`vae.model` if you need the graph directly (e.g. `.encode(x, scale)` /
`.decode(z, scale)` on a single batched tensor).

## Files

```
wan22-vae/
├── README.md
├── requirements.txt
├── example.py          # encode -> decode round-trip smoke test
└── wan22_vae/
    ├── __init__.py
    └── model.py        # the architecture (verbatim from wan/modules/vae2_2.py)
```

## Provenance

`wan22_vae/model.py` is a byte-for-byte copy of
`/group2/ct/weihanx/Wan2.2/wan/modules/vae2_2.py`. Copyright Alibaba Wan Team,
Apache-2.0 (see the Wan2.2 repo `LICENSE.txt`).
