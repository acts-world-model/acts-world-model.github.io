"""Build the full training/inference model from a config (shared by train and infer)."""

from __future__ import annotations

import functools
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from vtm_cosmos.model.build import (
    LORA_TARGETS_ATTN,
    LORA_TARGETS_FFN,
    apply_lora,
    build_language_model,
    build_network,
    freeze_all,
    load_nano_weights,
    materialize,
    trainable_summary,
)
from vtm_cosmos.model.generator_adapter import AdapterConfig, CosmosGeneratorAdapter
from vtm_cosmos.tasks import normalize_task_config, validate_prediction_config

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def build_model(cfg: DictConfig, device: torch.device, log=print) -> tuple[CosmosGeneratorAdapter, dict]:
    normalize_task_config(cfg)
    validate_prediction_config(cfg)
    m = cfg.model
    dtype = DTYPES[str(m.get("dtype", "float32"))]
    to = m.get("text_overrides", None)
    text_overrides = OmegaConf.to_container(to, resolve=True) if isinstance(to, DictConfig) else dict(to or {})
    lm = build_language_model(str(m.language_model), text_overrides=text_overrides, nano_json=m.get("nano_json"))
    net = build_network(lm)
    materialize(net, device, dtype)
    report = {}
    if m.get("nano_checkpoint"):
        report = load_nano_weights(net, m.nano_checkpoint, strict_expected=bool(m.get("strict_load", True)))
        log(f"[build] loaded Nano weights: {report['loaded']} tensors, missing={len(report['missing'])}, "
            f"unexpected={len(report['unexpected'])}")
    freeze_all(net)
    lora = m.get("lora", {}) or {}
    if lora.get("enable", False):
        targets = lora.get("targets", "attn")
        if targets == "attn":
            targets = LORA_TARGETS_ATTN
        elif targets == "attn_ffn":
            targets = LORA_TARGETS_ATTN + "," + LORA_TARGETS_FFN
        n = apply_lora(net, rank=int(lora.get("rank", 16)), alpha=int(lora.get("alpha", 32)), targets=targets, device=device)
        log(f"[build] LoRA injected into {n} GEN-tower linears (targets={targets})")
    elif m.get("train_generator_full", False):
        # GEN-only packs never touch the UND tower (q_proj, mlp, embed_tokens, lm_head, norm), so
        # training those would leave DDP with unused parameters: unfreeze the GEN tower only.
        for n_, p in net.language_model.named_parameters():
            p.requires_grad_("moe_gen" in n_)
    acfg = AdapterConfig(
        attention_backend=str(m.get("attention_backend", "flex")),
        flex_preference=str(m.get("flex_preference", "flex_triton")),
        temporal_mode=str(m.get("temporal_mode", "causal")),
        task_mode=str(m.get("task_mode", "action_causal")),
        attention_topology=str(m.get("attention_topology", "legacy")),
        attention_connections=m.get("attention_connections"),
        prediction_strategy=str(cfg.training.get("prediction_strategy", "joint")),
        init_tactile_from_rgb=bool(m.get("init_tactile_from_rgb", True)),
        train_time_embedder=bool(m.get("train_time_embedder", False)),
        train_rgb_projections=bool(m.get("train_rgb_projections", False)),
        active_streams=tuple(m.active_streams) if m.get("active_streams") else None,
    )
    adapter = CosmosGeneratorAdapter(net, acfg)
    for name in ("tactile2llm", "llm2tactile", "action2llm9", "llm2action9"):
        getattr(adapter, name).to(device=device, dtype=dtype)
    if adapter.view_action2llm9 is not None:
        adapter.view_action2llm9.to(device=device, dtype=dtype)
    adapter.stream_embed.data = adapter.stream_embed.data.to(device=device, dtype=dtype)
    ia = m.get("init_action_from_nano", None)
    if ia and ia.get("domain") and m.get("nano_checkpoint"):
        from vtm_cosmos.model.build import load_nano_action_head

        head = load_nano_action_head(m.nano_checkpoint, str(ia.domain))
        adapter.init_action_from_nano(head)
        action_inputs = "action2llm9"
        if adapter.view_action2llm9 is not None:
            action_inputs += ", view_action2llm9"
        log(f"[build] action inputs [{action_inputs}] + action output + all action stream embeds "
            f"initialized from Nano domain '{head['domain']}' (id {head['domain_id']})")
    adapter.set_trainable()
    if bool(m.get("trainable_fp32", True)):
        # fp32 master copies for every trainable tensor (LoRA A/B, new heads, stream embed);
        # autocast still runs the matmuls in bf16. Pure-bf16 AdamW silently loses small updates.
        for p in adapter.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()
    if bool(cfg.training.get("activation_checkpointing", False)):
        enable_activation_checkpointing(adapter)
    summ = trainable_summary(adapter)
    log(f"[build] params total={summ['total']:,} trainable={summ['trainable']:,} groups={summ['groups']}")
    return adapter, {"load_report": report, "trainable": summ}


def enable_activation_checkpointing(adapter: CosmosGeneratorAdapter) -> None:
    """Wrap each official decoder layer's forward in torch.utils.checkpoint (non-reentrant),
    keeping module names / state_dict keys intact."""
    from torch.utils.checkpoint import checkpoint

    for layer in adapter.net.language_model.model.layers:
        orig = layer.forward

        def ckpt_forward(*args, _orig=orig, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(_orig, *args, use_reentrant=False, **kwargs)
            return _orig(*args, **kwargs)

        layer.forward = ckpt_forward


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return normalize_task_config(cfg)
