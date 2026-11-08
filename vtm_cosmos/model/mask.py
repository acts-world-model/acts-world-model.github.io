"""Attention mask: configured stream topology x sample x time (spec §7).

One predicate serves both the block classification (``pair_allowed``) and the
element-level ``mask_mod`` FlexAttention evaluates inside partial blocks, so the
two can never disagree (spec §7.3 item 6). The closure captures tensors only
(FlashAttention-4 rejects captured Python scalars).
"""

from __future__ import annotations

import torch

from vtm_cosmos.model.layout import TEMPORAL_MODES, TokenLayout
from vtm_cosmos.streams import stream_allow_tensor


def make_pair_predicate(
    sample_id: torch.Tensor,
    stream_id: torch.Tensor,
    tick: torch.Tensor,
    allow: torch.Tensor,
    temporal_mode: str,
):
    if temporal_mode not in TEMPORAL_MODES:
        raise ValueError(f"temporal_mode must be one of {TEMPORAL_MODES}, got {temporal_mode}")
    causal = torch.tensor(temporal_mode == "causal", device=sample_id.device)

    def pred(b, h, q_idx, kv_idx):
        same = sample_id[q_idx] == sample_id[kv_idx]
        ok = same & allow[stream_id[q_idx], stream_id[kv_idx]]
        time_ok = tick[kv_idx] <= tick[q_idx]
        return ok & torch.where(causal, time_ok, torch.ones_like(time_ok))

    return pred


class MaskFields:
    """Device copies of the per-token fields for one layout."""

    def __init__(self, layout: TokenLayout, device: torch.device):
        self.sample_id = layout.sample_id().to(device)
        self.stream_id = layout.stream_id().to(device)
        self.tick = layout.tick().to(device)
        self.allow = stream_allow_tensor(device)
        if layout.cfg.task_mode == "action_full":
            from vtm_cosmos.streams import ALL_STREAMS, ALL_ACTION_STREAMS, MODALITY_ACTIONS, STREAM_ID
            allow = torch.zeros(len(ALL_STREAMS), len(ALL_STREAMS), dtype=torch.bool, device=device)
            allow[:5, :5] = self.allow[:5, :5]
            for action in ALL_ACTION_STREAMS:
                allow[STREAM_ID[action], STREAM_ID[action]] = True
            for modality, actions in MODALITY_ACTIONS.items():
                for action in actions:
                    i, j = STREAM_ID[modality], STREAM_ID[action]
                    allow[i, j] = allow[j, i] = True
            self.allow = allow
        if layout.cfg.attention_topology == "all_streams_causal":
            self.allow = torch.ones_like(self.allow)
        elif layout.cfg.attention_topology == "custom":
            from vtm_cosmos.streams import STREAM_ID
            self.allow = torch.zeros_like(self.allow)
            for query, keys in layout.cfg.attention_connections.items():
                for key in keys:
                    self.allow[STREAM_ID[query], STREAM_ID[key]] = True
        self.n = layout.n_pad

    def predicate(self, temporal_mode: str):
        return make_pair_predicate(self.sample_id, self.stream_id, self.tick, self.allow, temporal_mode)


def dense_mask(layout: TokenLayout, device: torch.device, temporal_mode: str) -> torch.Tensor:
    """bool [n_pad, n_pad], True = query may read key (SDPA semantics)."""
    if layout.cfg.attention_topology == "all_streams_causal" and temporal_mode != "causal":
        raise ValueError("all_streams_causal requires causal temporal attention")
    f = MaskFields(layout, device)
    pred = f.predicate(temporal_mode)
    idx = torch.arange(f.n, device=device)
    b = torch.zeros((), dtype=torch.long, device=device)
    return pred(b, b, idx[:, None], idx[None, :])


def build_block_mask(layout: TokenLayout, device: torch.device, temporal_mode: str, block_size: tuple[int, int]):
    """FlexAttention BlockMask over the GEN-only [n_pad, n_pad] stream via the official run builder."""
    from cosmos_framework.model.generator.mot.flex_attention_utils import (
        build_block_mask_from_metadata_runs,
        metadata_run_groups,
    )

    if layout.cfg.attention_topology == "all_streams_causal" and temporal_mode != "causal":
        raise ValueError("all_streams_causal requires causal temporal attention")
    f = MaskFields(layout, device)
    pred = f.predicate(temporal_mode)
    group_id, reps = metadata_run_groups((f.sample_id, f.stream_id, f.tick), device=device)
    return build_block_mask_from_metadata_runs(
        q_group_id=group_id,
        kv_group_id=group_id,
        q_representatives=reps,
        kv_representatives=reps,
        pair_allowed=pred,
        mask_mod=pred,
        q_len=f.n,
        kv_len=f.n,
        device=device,
        block_size=block_size,
    )


def block_mask_to_dense(block_mask, n: int) -> torch.Tensor:
    """Materialize a BlockMask (full + partial blocks refined by mask_mod) as bool [n, n]; tests only."""
    return block_mask.to_dense()[0, 0, :n, :n].bool()
