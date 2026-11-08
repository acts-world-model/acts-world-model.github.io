"""CosmosGeneratorAdapter: multimodal streams in, image/action velocities out, through the
official Cosmos3 Generator layers (spec §8.3).

Pipeline per forward:
  states -> patchify / raw 9-D -> projections (official vae2llm for RGB, new tactile2llm,
  new action2llm9) -> + stream embedding -> + official time_embedder(sigma) on target
  tokens -> GEN-only SequencePack (official build_packed_sequence) -> official
  language_model(...) with our flex/sdpa dispatch and float32 mRoPE positions ->
  official llm2vae / new heads -> unpatchify -> zero on non-target positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from cosmos_framework.data.generator.sequence_packing.runtime import get_gen_seq
from cosmos_framework.model.generator.mot.attention import build_packed_sequence

from vtm_cosmos.model.attention_dispatch import VtmAttentionDispatch, install_dispatch
from vtm_cosmos.model.build import init_linear_
from vtm_cosmos.model.layout import LayoutConfig, TokenLayout, build_layout, patchify, unpatchify
from vtm_cosmos.model.mask import build_block_mask, dense_mask
from vtm_cosmos.tasks import canonical_task_mode, validate_prediction_config
from vtm_cosmos.model.topology import normalize_connections
from vtm_cosmos.streams import (
    ACTION_DIM,
    ACTION_STREAMS, ALL_ACTION_STREAMS, ALL_STREAMS, VIEW_ACTION_STREAMS,
    IMAGE_STREAMS,
    LATENT_CHANNELS,
    NUM_STREAMS,
    RGB_STREAMS,
    STREAM_ID,
    STREAMS,
    TACTILE_STREAMS,
    active_pattern,
    normalize_active_streams,
)


@dataclass
class AdapterConfig:
    attention_backend: str = "flex"  # flex | sdpa
    flex_preference: str = "flex_triton"  # flex_triton | auto | flex_flash
    temporal_mode: str = "causal"
    task_mode: str = "action_causal"
    attention_topology: str = "legacy"
    attention_connections: dict[str, tuple[str, ...]] | None = None
    prediction_strategy: str = "joint"
    init_tactile_from_rgb: bool = True
    train_time_embedder: bool = False
    train_rgb_projections: bool = False
    active_streams: tuple[str, ...] | None = None  # None = every stream of the task

    def __post_init__(self):
        self.task_mode = canonical_task_mode(self.task_mode)
        self.attention_connections = normalize_connections(
            self.task_mode, self.attention_topology, self.attention_connections)
        self.active_streams = normalize_active_streams(self.task_mode, self.active_streams)

    def models(self, stream: str) -> bool:
        return self.active_streams is None or stream in self.active_streams


class CosmosGeneratorAdapter(nn.Module):
    def __init__(self, net, cfg: AdapterConfig):
        super().__init__()
        self.net = net
        self.cfg = cfg
        validate_prediction_config({"model": vars(cfg),
                                    "training": {"prediction_strategy": cfg.prediction_strategy}})
        D = net.hidden_size
        patch_dim = net.patch_latent_dim
        self.tactile2llm = nn.Linear(patch_dim, D, bias=True)
        self.llm2tactile = nn.Linear(D, patch_dim, bias=True)
        self.action2llm9 = nn.Linear(ACTION_DIM, D, bias=True)
        self.llm2action9 = nn.Linear(D, ACTION_DIM, bias=True)
        self.view_action2llm9 = nn.Linear(ACTION_DIM, D, bias=True) if cfg.task_mode == "action_full" else None
        self.stream_embed = nn.Parameter(torch.zeros(len(ALL_STREAMS) if cfg.task_mode == "action_full" else NUM_STREAMS, D))
        self.reset_new_parameters()
        if cfg.task_mode == "action_full":
            self.llm2action9.requires_grad_(False)

        self.layout_cfg = LayoutConfig(task_mode=cfg.task_mode, attention_topology=cfg.attention_topology,
                                       attention_connections=cfg.attention_connections,
                                       active_streams=cfg.active_streams)
        self.dispatch = VtmAttentionDispatch(cfg.attention_backend)
        self.num_layers = install_dispatch(net.language_model, self.dispatch)
        self.flex_backend = None
        self._mask_cache: dict = {}

    # ------------------------------------------------------------------ init
    @torch.no_grad()
    def reset_new_parameters(self) -> None:
        D = self.net.hidden_size
        if self.cfg.init_tactile_from_rgb and not any(p.device.type == "meta" for p in self.net.vae2llm.parameters()):
            self.tactile2llm.weight.copy_(self.net.vae2llm.weight.to(self.tactile2llm.weight.dtype))
            self.tactile2llm.bias.copy_(self.net.vae2llm.bias.to(self.tactile2llm.bias.dtype))
            self.llm2tactile.weight.copy_(self.net.llm2vae.weight.to(self.llm2tactile.weight.dtype))
            self.llm2tactile.bias.copy_(self.net.llm2vae.bias.to(self.llm2tactile.bias.dtype))
        else:
            init_linear_(self.tactile2llm, self.tactile2llm.in_features)
            init_linear_(self.llm2tactile, D)
        init_linear_(self.action2llm9, ACTION_DIM)
        if self.view_action2llm9 is not None:
            init_linear_(self.view_action2llm9, ACTION_DIM)
        init_linear_(self.llm2action9, D)
        std = 1.0 / (D**0.5)
        nn.init.trunc_normal_(self.stream_embed, std=std, a=-3 * std, b=3 * std)

    @torch.no_grad()
    def init_action_from_nano(self, head: dict) -> None:
        """Initialize all action inputs and their stream embeddings from a Nano domain.

        Sensor and view projections receive independent copies of the same 9-D
        input weights, then fine-tune separately. Each projection is shared by
        its two hands; view coordinates retain their own representation/normalizer.
        """
        if head["hidden"] != self.net.hidden_size:
            raise ValueError(f"Nano action head width {head['hidden']} != model hidden {self.net.hidden_size}")
        inputs = [self.action2llm9]
        action_streams = ACTION_STREAMS
        if self.view_action2llm9 is not None:
            inputs.append(self.view_action2llm9)
            action_streams += VIEW_ACTION_STREAMS
        for projection in inputs:
            projection.weight.copy_(head["in_weight"].to(projection.weight))
            projection.bias.copy_(head["in_bias"].to(projection.bias))
        self.llm2action9.weight.copy_(head["out_weight"].to(self.llm2action9.weight))
        self.llm2action9.bias.copy_(head["out_bias"].to(self.llm2action9.bias))
        emb = head["modality_embed"].to(self.stream_embed)
        for name in action_streams:  # retain random stream identity offsets around the pretrained modality vector
            self.stream_embed[STREAM_ID[name]] += emb

    def set_trainable(self) -> None:
        """New modules always train; official projections/time embedder per config;
        the Generator body is whatever apply_lora left (LoRA params on, base frozen)."""
        for m in (self.tactile2llm, self.llm2tactile, self.action2llm9, self.llm2action9):
            for p in m.parameters():
                p.requires_grad_(True)
        if self.cfg.task_mode == "action_full":
            self.llm2action9.requires_grad_(False)
            self.view_action2llm9.requires_grad_(True)
        # Heads of streams this run never models get no gradient; leave them frozen so DDP is happy
        # and they stay out of the trainable checkpoint state.
        if not any(self.cfg.models(s) for s in TACTILE_STREAMS):
            self.tactile2llm.requires_grad_(False)
            self.llm2tactile.requires_grad_(False)
        if not any(self.cfg.models(s) for s in ACTION_STREAMS):
            self.action2llm9.requires_grad_(False)
            self.llm2action9.requires_grad_(False)
        self.stream_embed.requires_grad_(True)
        for p in self.net.time_embedder.parameters():
            p.requires_grad_(self.cfg.train_time_embedder)
        for m in (self.net.vae2llm, self.net.llm2vae):
            for p in m.parameters():
                p.requires_grad_(self.cfg.train_rgb_projections)

    # ---------------------------------------------------------------- helpers
    @property
    def device(self) -> torch.device:
        return self.net.vae2llm.weight.device

    def _backend(self, device):
        if self.cfg.attention_backend != "flex":
            return None
        if self.flex_backend is None:
            from cosmos_framework.model.generator.mot.flex_attention import resolve_flex_backend

            self.flex_backend = resolve_flex_backend(device, self.cfg.flex_preference)
        return self.flex_backend

    def layout_for(self, batch_size: int, device, active=None, target_index: int | None = None) -> TokenLayout:
        """Full window or causal prefix for samples sharing one active-stream pattern."""
        be = self._backend(device)
        align = be.full_seq_alignment if be is not None else 1
        if active is None:
            active = self.cfg.active_streams or (ALL_STREAMS if self.cfg.task_mode == "action_full" else STREAMS)
        pat = active_pattern(active)
        if target_index is not None and self.cfg.prediction_strategy != "causal_teacher_forcing":
            raise ValueError("Prefix layouts require causal_teacher_forcing")
        key = ("layout", batch_size, align, pat, target_index)
        if key not in self._mask_cache:
            cfg = self.layout_cfg if target_index is None else self.layout_cfg.causal_prefix(target_index)
            self._mask_cache[key] = build_layout(cfg, batch_size, align=align, active=active)
        return self._mask_cache[key]

    def _masks(self, layout: TokenLayout, device):
        key = ("mask", layout.batch_size, layout.n_pad, layout.pattern, layout.cfg.target_index,
               str(device), self.cfg.temporal_mode,
               self.cfg.attention_backend)
        if key not in self._mask_cache:
            if self.cfg.attention_backend == "flex":
                be = self._backend(device)
                self._mask_cache[key] = build_block_mask(layout, device, self.cfg.temporal_mode, be.block_size)
            else:
                self._mask_cache[key] = dense_mask(layout, device, self.cfg.temporal_mode)
        return self._mask_cache[key]

    def _in_proj(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if name in RGB_STREAMS:
            return self.net.vae2llm(x)
        if name in TACTILE_STREAMS:
            return self.tactile2llm(x)
        if name in VIEW_ACTION_STREAMS:
            return self.view_action2llm9(x)
        return self.action2llm9(x)

    def _out_proj(self, name: str, h: torch.Tensor) -> torch.Tensor:
        if name in RGB_STREAMS:
            return self.net.llm2vae(h)
        if name in TACTILE_STREAMS:
            return self.llm2tactile(h)
        return self.llm2action9(h)

    # ---------------------------------------------------------------- forward
    def forward(self, states: dict[str, torch.Tensor], sigma_b: torch.Tensor, layout: TokenLayout | None = None):
        """states: image [B,48,4,16,16] (history clean, future noisy), action [B,15,9] (normalized),
        one entry per ACTIVE stream (absent wrists are simply not in the dict). sigma_b: [B] in [0,1].
        Returns {active stream: velocity with the state's shape, zero off-target}."""
        B = states[STREAMS[0]].shape[0]
        device = states[STREAMS[0]].device
        if layout is None:
            layout = self.layout_for(B, device, active=tuple(states.keys()))
        if self.cfg.prediction_strategy == "causal_teacher_forcing":
            expected = self.layout_cfg.causal_prefix(layout.cfg.target_index) if layout.cfg.target_index is not None else None
            if expected is None or layout.cfg != expected:
                raise ValueError("Causal forwards require an explicit current-target prefix layout")
        if set(layout.active) != set(states.keys()):
            raise ValueError(f"states {sorted(states)} do not match layout.active {layout.active}")
        D = self.net.hidden_size
        param_dtype = self.net.vae2llm.weight.dtype

        # 1. per-stream tokens (active streams only, in STREAMS order)
        toks = []
        for name in layout.active:
            x = states[name]
            feats = patchify(x) if name in IMAGE_STREAMS else x
            a, b = layout.slices[name]
            if feats.shape[1] != b - a:
                raise ValueError(f"{name}: {feats.shape[1]} tokens, layout expects {b - a}")
            h = self._in_proj(name, feats.to(param_dtype))
            toks.append(h)
        tokens = torch.cat(toks, dim=1)  # [B, n, D]

        # 2. stream identity + noise-level embedding on target tokens only
        sid = layout.stream_id_s.to(device)
        tokens = tokens + self.stream_embed[sid].to(tokens.dtype)[None]
        if device.type == "cuda":  # official code embeds timesteps in fp32 under autocast
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float32):
                emb = self.net.time_embedder(sigma_b.float().reshape(B))  # [B, D]
        else:
            emb = self.net.time_embedder(sigma_b.float().reshape(B))
        tgt = layout.is_target_s.to(device).to(tokens.dtype)  # [n]
        tokens = tokens + (tgt[None, :, None] * emb.to(tokens.dtype)[:, None, :])

        # 3. GEN-only pack (official packer) + our mask
        packed = tokens.reshape(B * layout.n_per_sample, D)
        be = self._backend(device)
        align_q = be.full_seq_alignment if be is not None else 1
        align_kv = be.causal_seq_alignment if be is not None else 1
        n = layout.n_per_sample
        input_pack, split_info, _ = build_packed_sequence(
            "two_way",
            packed_sequence=packed,
            attn_modes=["full"] * B,
            split_lens=[n] * B,
            sample_lens=[n] * B,
            packed_und_token_indexes=torch.empty(0, dtype=torch.long, device=device),
            packed_gen_token_indexes=torch.arange(B * n, device=device),
            num_heads=self.net.num_heads,
            head_dim=self.net.head_dim,
            num_layers=self.net.num_hidden_layers,
            full_seq_alignment=align_q,
            causal_seq_alignment=align_kv,
        )
        if get_gen_seq(input_pack).shape[0] != layout.n_pad:
            raise RuntimeError(f"pack length {get_gen_seq(input_pack).shape[0]} != layout n_pad {layout.n_pad}")
        mask = self._masks(layout, device)
        if self.cfg.attention_backend == "flex":
            split_info.flex_block_mask = mask
            split_info.flex_backend = be
            self.dispatch.set_mask(block_mask=mask, backend=be)
        else:
            self.dispatch.set_mask(dense_mask=mask)
        position_ids = layout.position_ids().to(device)  # [3, n_real] float32

        # 4. official Generator
        out_pack, _ = self.net.language_model(input_pack, attention_mask=split_info, position_ids=position_ids)
        h_all = get_gen_seq(out_pack)[: layout.n_real].reshape(B, n, D)

        # 5. per-stream decode, zero off-target
        result = {}
        g = layout.cfg.grid
        for name in layout.active:
            if layout.cfg.task_mode == "action_full" and name in ALL_ACTION_STREAMS:
                continue
            a, b = layout.slices[name]
            feats = self._out_proj(name, h_all[:, a:b])
            if name in IMAGE_STREAMS:
                pred = unpatchify(feats.float(), layout.cfg.latent_t, g, g, LATENT_CHANNELS)
            else:
                pred = feats.float()
            m = layout.target_mask(name).to(device)
            result[name] = torch.where(m, pred, torch.zeros_like(pred))
        return result

    # ------------------------------------------------------------ checkpoint
    def trainable_state_dict(self) -> dict:
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if self._is_trainable_key(k)}

    def _is_trainable_key(self, key: str) -> bool:
        params = dict(self.named_parameters())
        p = params.get(key)
        return p is not None and p.requires_grad

    def load_trainable_state_dict(self, sd: dict, strict: bool = True) -> None:
        missing, unexpected = self.load_state_dict(sd, strict=False)
        expected = set(k for k, p in self.named_parameters() if p.requires_grad)
        got = set(sd.keys())
        if strict and (expected - got):
            raise RuntimeError(f"trainable keys missing from checkpoint: {sorted(expected - got)[:10]}")
        if unexpected:
            raise RuntimeError(f"unexpected keys: {unexpected[:10]}")
