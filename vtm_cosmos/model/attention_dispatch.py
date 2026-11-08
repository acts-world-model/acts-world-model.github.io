"""Per-layer attention dispatch for the GEN-only pack.

Installed on every ``PackedAttentionMoT.dispatch_attention_fn`` (an instance
attribute the official code itself replaces for AR/KV-memory variants), so all
layers see the same mask semantics (spec §7.3 item 5). Two implementations:

* ``flex``: the production path — the official ``flex_attention`` kernel with the
  BlockMask from ``vtm_cosmos.model.mask``.
* ``sdpa``: a correctness reference with an explicit boolean mask (True = may
  read), used by the unit tests to validate the flex path.

Both skip the UND (reasoner) branch entirely: our packs carry zero causal
tokens, and the official ``multiview_attention`` would still run
``und_self_attention`` over the empty stream, which not every backend accepts.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.sequence_packing.runtime import from_mode_splits, get_gen_seq


class VtmAttentionDispatch:
    def __init__(self, mode: str = "flex"):
        if mode not in ("flex", "sdpa"):
            raise ValueError(mode)
        self.mode = mode
        self.block_mask = None
        self.backend = None
        self.dense_mask: torch.Tensor | None = None
        self.calls = 0

    def set_mask(self, *, block_mask=None, backend=None, dense_mask: torch.Tensor | None = None) -> None:
        self.block_mask = block_mask
        self.backend = backend
        self.dense_mask = dense_mask

    def __call__(
        self,
        q_pack,
        k_pack,
        v_pack,
        attention_mask,
        natten_metadata=None,
        memory_value=None,
        packed_key_states_normalized=None,
    ):
        if int(q_pack["_num_causal_tokens"]) != 0:
            raise RuntimeError("VtmAttentionDispatch expects a GEN-only pack (no UND tokens)")
        q = get_gen_seq(q_pack)  # [N, H, Dh]
        k = get_gen_seq(k_pack)  # [N, Hkv, Dh]
        v = get_gen_seq(v_pack)
        n, heads, dh = q.shape
        # Under autocast with fp32 base weights the official RMSNorm promotes q/k to fp32 while v
        # stays bf16; both kernels require one dtype, so follow the query.
        k, v = k.to(q.dtype), v.to(q.dtype)
        self.calls += 1
        if self.mode == "flex":
            if self.block_mask is None or self.backend is None:
                raise RuntimeError("flex dispatch called before set_mask()")
            from cosmos_framework.model.generator.mot.flex_attention import flex_attention

            out = flex_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), self.block_mask, self.backend)[0]
        else:
            if self.dense_mask is None:
                raise RuntimeError("sdpa dispatch called before set_mask()")
            if self.dense_mask.shape != (n, n):
                raise RuntimeError(f"dense mask {tuple(self.dense_mask.shape)} does not match N={n}")
            rep = heads // k.shape[1]
            qh = q.permute(1, 0, 2).unsqueeze(0)  # [1,H,N,Dh]
            kh = k.repeat_interleave(rep, dim=1).permute(1, 0, 2).unsqueeze(0)
            vh = v.repeat_interleave(rep, dim=1).permute(1, 0, 2).unsqueeze(0)
            out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=self.dense_mask)  # [1,H,N,Dh]
            out = out[0].permute(1, 0, 2)  # [N,H,Dh]
        out = out.reshape(n, heads * dh)
        causal_out = out.new_zeros(0, heads * dh)
        return from_mode_splits(causal_out, out, q_pack), None


def install_dispatch(language_model, fn) -> int:
    layers = language_model.model.layers
    for layer in layers:
        layer.self_attn.dispatch_attention_fn = fn
    return len(layers)
