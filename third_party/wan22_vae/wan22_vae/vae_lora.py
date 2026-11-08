"""Causal-preserving LoRA for the Wan2.2 VAE (WanVAE_).

Why a custom LoRA instead of peft
---------------------------------
The VAE's temporal convs are ``CausalConv3d``, which set ``self.padding = (0,0,0)``
and apply the (causal, left-only) padding manually inside ``forward``. peft's
conv-LoRA reads ``base_layer.padding`` to size its ``lora_A``; with padding 0 it
would produce a spatially smaller delta than the base output and the residual add
would fail. It also would not preserve causality (its delta uses symmetric
padding). So we implement a tiny LoRA that:

  * wraps the frozen base conv and calls it unchanged (forwarding any extra
    positional args such as the feature cache), and
  * adds a low-rank residual ``lora_B(lora_A(x))`` whose ``lora_A`` is *the same
    conv class* as the base (a ``CausalConv3d`` for temporal convs, an
    ``nn.Conv2d`` for the attention 1x1 convs). Same class + same kernel/stride/
    padding => the delta has identical output geometry AND the same causal
    behavior as the layer it adapts.

``lora_B`` is initialized to zero, so at step 0 the model reproduces the frozen
VAE exactly and finetuning departs smoothly from the pretrained reconstruction.
"""

import math

import torch
import torch.nn as nn

from .model import CausalConv3d


def _causal_orig_padding(conv: CausalConv3d):
    """Recover the (t, h, w) padding a CausalConv3d was constructed with.

    CausalConv3d.__init__ stores
        self._padding = (pad_w, pad_w, pad_h, pad_h, 2*pad_t, 0)
    and then zeroes self.padding. Invert that here.
    """
    p = conv._padding
    return (p[4] // 2, p[2], p[0])  # (pad_t, pad_h, pad_w)


class LoRACausalConv3d(CausalConv3d):
    """Frozen CausalConv3d + low-rank causal residual.

    Subclasses CausalConv3d (but skips its __init__ via nn.Module.__init__, so no
    redundant conv weight is allocated) so that `isinstance(layer, CausalConv3d)`
    stays True at the three cache-threading sites in ResidualBlock/Encoder3d/
    Decoder3d. That is what lets the frozen base keep receiving its temporal
    feature cache after wrapping — otherwise causality/caching silently breaks.
    """

    def __init__(self, base: CausalConv3d, r: int, alpha: float, dropout: float = 0.0):
        nn.Module.__init__(self)  # NOT CausalConv3d.__init__ (would alloc a weight)
        self.base = base  # frozen; kept as-is (incl. its cache-aware forward)
        self.r = r
        self.scaling = alpha / r
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_ch, out_ch = base.in_channels, base.out_channels
        kernel = base.kernel_size
        stride = base.stride
        pad = _causal_orig_padding(base)
        # lora_A mirrors the base geometry (causal, same kernel/stride/padding);
        # lora_B is a 1x1x1 mixing conv. Both bias-free.
        self.lora_A = CausalConv3d(in_ch, r, kernel, stride=stride, padding=pad, bias=False)
        self.lora_B = CausalConv3d(r, out_ch, 1, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x, *args, **kwargs):
        out = self.base(x, *args, **kwargs)
        delta = self.lora_B(self.lora_A(self.drop(x)))
        return out + delta * self.scaling


class LoRAConv2d(nn.Module):
    """Frozen nn.Conv2d + low-rank residual (used for the attention 1x1 convs)."""

    def __init__(self, base: nn.Conv2d, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base  # frozen
        self.r = r
        self.scaling = alpha / r
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_ch, out_ch = base.in_channels, base.out_channels
        self.lora_A = nn.Conv2d(in_ch, r, base.kernel_size, stride=base.stride,
                                padding=base.padding, bias=False)
        self.lora_B = nn.Conv2d(r, out_ch, 1, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x, *args, **kwargs):
        out = self.base(x, *args, **kwargs)
        delta = self.lora_B(self.lora_A(self.drop(x)))
        return out + delta * self.scaling


# ---- target selection & injection -------------------------------------------

# Preset filters over the fully-qualified module name (relative to WanVAE_).
TARGET_PRESETS = {
    # ResidualBlock feature convs (the VAE's conv "MLP") + attention 1x1 convs.
    "residual_attn": lambda n: (".residual." in n) or n.endswith(("to_qkv", "proj")),
    # everything conv-shaped we can safely adapt (adds shortcuts, resample, time_conv)
    "all": lambda n: True,
    # decoder only (cheapest path to better pixels)
    "decoder": lambda n: n.startswith("decoder."),
}


def _iter_target_convs(model, name_filter):
    for name, mod in model.named_modules():
        if isinstance(mod, (CausalConv3d, nn.Conv2d)) and name_filter(name):
            yield name, mod


def inject_lora(model, r=16, alpha=16, dropout=0.0, preset="residual_attn",
                name_filter=None):
    """Replace selected conv layers in `model` (a WanVAE_) with LoRA wrappers.

    Returns (num_wrapped, wrapped_names). Freezes all non-LoRA params.
    """
    if name_filter is None:
        if preset not in TARGET_PRESETS:
            raise ValueError(f"unknown preset {preset!r}; choose from {list(TARGET_PRESETS)}")
        name_filter = TARGET_PRESETS[preset]

    # collect first (can't mutate while iterating named_modules)
    targets = list(_iter_target_convs(model, name_filter))
    wrapped_names = []
    for name, mod in targets:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        if isinstance(mod, CausalConv3d):
            wrapper = LoRACausalConv3d(mod, r, alpha, dropout)
        else:
            wrapper = LoRAConv2d(mod, r, alpha, dropout)
        setattr(parent, child, wrapper)
        wrapped_names.append(name)

    # freeze everything, then re-enable only LoRA params
    model.requires_grad_(False)
    for n, p in model.named_parameters():
        if ".lora_A." in n or ".lora_B." in n:
            p.requires_grad_(True)
    return len(wrapped_names), wrapped_names


def lora_parameters(model):
    return [p for n, p in model.named_parameters()
            if p.requires_grad and (".lora_A." in n or ".lora_B." in n)]


def lora_state_dict(model):
    return {n: p.detach().cpu()
            for n, p in model.named_parameters()
            if ".lora_A." in n or ".lora_B." in n}


def load_lora_state_dict(model, state_dict, strict=True):
    own = dict(model.named_parameters())
    missing, unexpected = [], []
    for k, v in state_dict.items():
        if k in own:
            with torch.no_grad():
                own[k].copy_(v.to(own[k].device, own[k].dtype))
        else:
            unexpected.append(k)
    lora_keys = {n for n in own if ".lora_A." in n or ".lora_B." in n}
    missing = [k for k in lora_keys if k not in state_dict]
    if strict and (missing or unexpected):
        raise RuntimeError(f"LoRA load mismatch: missing={missing[:4]} unexpected={unexpected[:4]}")
    return missing, unexpected
