"""Construction of the official Cosmos3 Generator (tiny random-init or Nano),
checkpoint loading into the ``Cosmos3VFMNetwork`` subtree, and LoRA injection.

Everything here calls cosmos-framework code unmodified:
  Qwen3MoTConfig / Qwen3VLMoTConfig + Qwen3VLTextForCausalLM   (mot/unified_mot.py)
  Cosmos3VFMNetworkConfig + Cosmos3VFMNetwork                    (mot/cosmos3_vfm_network.py)
  inject_lora_pre_fsdp / init_lora_weights_post_materialization (utils/generator/lora.py)
  diffusers_to_net_key (vtm_cosmos/model/nano_keys.py, copied from inference/model.py)
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
from torch import nn

from vtm_cosmos.streams import LATENT_CHANNELS, LATENT_HW, LATENT_T, PATCH

# NVIDIA/cosmos-framework @ 3f94d25 (cloned by scripts/setup_env.sh); override with $COSMOS_FRAMEWORK_ROOT
COSMOS_ROOT = Path(os.environ.get("COSMOS_FRAMEWORK_ROOT",
                                  Path(__file__).resolve().parents[2] / "third_party" / "cosmos-framework"))
NANO_TEXT_JSON = (
    COSMOS_ROOT / "cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"
)

# Tiny Qwen3-style text config: same layer code as Nano, ~2M params. mrope_section must
# sum to head_dim // 2; Q:KV head ratio must be a power of two (FlexAttention GQA).
TINY_TEXT_CONFIG: dict = {
    "vocab_size": 64,
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "hidden_act": "silu",
    "max_position_embeddings": 32768,
    "rms_norm_eps": 1e-6,
    "rope_theta": 5000000.0,
    "rope_scaling": {"rope_type": "default", "mrope_section": [8, 4, 4], "mrope_interleaved": True},
    "attention_bias": False,
    "attention_dropout": 0.0,
    "tie_word_embeddings": True,
    "use_cache": False,
}

LORA_TARGETS_ATTN = "q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen"
LORA_TARGETS_FFN = "mlp_moe_gen.gate_proj,mlp_moe_gen.up_proj,mlp_moe_gen.down_proj"


def build_language_model(kind: str = "tiny", text_overrides: dict | None = None, nano_json: str | Path | None = None):
    """Returns the MoT LM (both towers). 'tiny' builds on the current device; 'nano' on meta."""
    from cosmos_framework.model.generator.mot.unified_mot import (
        Qwen3MoTConfig,
        Qwen3VLMoTConfig,
        Qwen3VLTextForCausalLM,
    )

    if kind == "tiny":
        cfg_dict = {**TINY_TEXT_CONFIG, **(text_overrides or {})}
        cfg = Qwen3MoTConfig(config_dict=cfg_dict, qk_norm_for_text=True, qk_norm_for_diffusion=True)
        return Qwen3VLTextForCausalLM(cfg)
    if kind == "nano":
        cfg = Qwen3VLMoTConfig.from_json_file(str(nano_json or NANO_TEXT_JSON))
        cfg.include_visual = False  # the ViT is never used by the generator adapter
        cfg.qk_norm_for_text = True
        cfg.qk_norm_for_diffusion = True
        if text_overrides:
            cfg.text_config_overrides = dict(text_overrides)
        with torch.device("meta"):
            return Qwen3VLTextForCausalLM(cfg)
    raise ValueError(f"unknown language model kind {kind!r}")


def build_network(language_model, *, latent_channels: int = LATENT_CHANNELS, patch: int = PATCH, timestep_scale: float = 1e-3):
    from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork, Cosmos3VFMNetworkConfig

    cfg = Cosmos3VFMNetworkConfig(
        vision_gen=True,
        action_gen=False,  # the 9-D action heads are new modules in the adapter (spec §5.2)
        sound_gen=False,
        vlm_config=language_model.config,
        latent_patch_size=patch,
        latent_downsample_factor=16,
        latent_channel_size=latent_channels,
        max_latent_h=LATENT_HW // patch,
        max_latent_w=LATENT_HW // patch,
        max_latent_t=LATENT_T,
        enable_fps_modulation=True,
        base_fps=24,
        timestep_shift=1.0,
        timestep_scale=timestep_scale,
        joint_attn_implementation="two_way",
        enable_input_bias=True,
    )
    device = torch.device("meta") if _is_meta(language_model) else None
    if device is not None:
        with torch.device("meta"):
            return Cosmos3VFMNetwork(language_model, cfg)
    return Cosmos3VFMNetwork(language_model, cfg)


def _is_meta(module: nn.Module) -> bool:
    return any(p.device.type == "meta" for p in module.parameters())


def materialize(net, device: torch.device, dtype: torch.dtype | None = None):
    """Allocate meta params on ``device`` and rebuild non-persistent buffers (RoPE inv_freq,
    timestep frequencies) via the official ``init_weights``. Also (re)initializes weights,
    so call it BEFORE loading a checkpoint."""
    if _is_meta(net):
        net.to_empty(device=device)
    else:
        net.to(device)
    net.init_weights(buffer_device=device)
    if dtype is not None:
        # cast parameters only: RoPE inv_freq / timestep frequencies must stay float32
        for p in net.parameters():
            p.data = p.data.to(dtype)
    return net


# ---------------------------------------------------------------------------
# checkpoint loading
# ---------------------------------------------------------------------------
def _iter_safetensors(ckpt: Path):
    """Yield (relative weight path, safetensors file) for a diffusers-style Cosmos3 checkpoint."""
    for index, prefix in ((ckpt / "model.safetensors.index.json", ""),
                          (ckpt / "transformer/diffusion_pytorch_model.safetensors.index.json", "transformer/")):
        if index.exists():
            wm = json.loads(index.read_text())["weight_map"]
            for rel in sorted(set(wm.values())):
                path = ckpt / prefix / rel
                if not path.exists():
                    if rel.startswith("vision_encoder/"):  # reasoner ViT: never used by the adapter
                        continue
                    raise FileNotFoundError(path)
                yield prefix + rel, path
            return
    found = sorted(ckpt.rglob("*.safetensors"))
    if not found:
        raise FileNotFoundError(f"no safetensors under {ckpt}")
    for f in found:
        yield str(f.relative_to(ckpt)), f


def load_nano_weights(net, checkpoint_path: str | Path, strict_expected: bool = True) -> dict:
    """Copy Cosmos3 generator weights (HF/diffusers layout) into ``net`` shard by shard.

    Returns a report {loaded, missing, unexpected, skipped}. Missing keys are checked
    against the set we know the adapter does not use (visual tower, action heads, lm_head)."""
    from safetensors import safe_open

    from vtm_cosmos.model.nano_keys import diffusers_to_net_key as _diffusers_to_net_key

    ckpt = Path(checkpoint_path)
    if not ckpt.is_dir():
        raise FileNotFoundError(ckpt)
    target = dict(net.named_parameters())
    target.update({k: v for k, v in net.named_buffers() if k in dict(net.state_dict())})
    state = net.state_dict()
    loaded, unexpected, skipped = [], [], []
    with torch.no_grad():
        for rel, f in _iter_safetensors(ckpt):
            with safe_open(str(f), framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    net_key = _diffusers_to_net_key(key, rel)
                    if net_key is None:
                        skipped.append(key)
                        continue
                    if net_key not in state:
                        unexpected.append(net_key)
                        continue
                    t = sf.get_tensor(key)
                    dst = state[net_key]
                    if tuple(t.shape) != tuple(dst.shape):
                        raise ValueError(f"{net_key}: checkpoint shape {tuple(t.shape)} != model {tuple(dst.shape)}")
                    dst.copy_(t.to(dst.dtype))
                    loaded.append(net_key)
    loaded_set = set(loaded)
    missing = [k for k in state.keys() if k not in loaded_set and not k.endswith("inv_freq")]
    allowed_missing_prefixes = ("language_model.visual.", "action2llm", "llm2action", "language_model.lm_head.")
    bad_missing = [k for k in missing if not k.startswith(allowed_missing_prefixes) and "lora_" not in k]
    report = {"loaded": len(loaded), "missing": missing, "unexpected": unexpected, "skipped": len(skipped)}
    if strict_expected and bad_missing:
        raise RuntimeError(f"unexpected missing generator keys ({len(bad_missing)}): {bad_missing[:20]}")
    return report


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
def apply_lora(net, *, rank: int, alpha: int, targets: str, device: torch.device) -> int:
    """Inject official LoRA into the GEN tower of an already materialized network.

    ``inject_lora_pre_fsdp`` freezes every non-LoRA parameter and creates lora_A/B on the
    meta device; we allocate those two sub-modules only (never the whole net) and init them."""
    from cosmos_framework.utils.generator.lora import (
        LoraInjectedLinear,
        init_lora_weights_post_materialization,
        inject_lora_pre_fsdp,
    )

    inject_lora_pre_fsdp(net, lora_rank=rank, lora_alpha=alpha, lora_target_modules=targets)
    n = 0
    for m in net.modules():
        if isinstance(m, LoraInjectedLinear):
            m.lora_A.to_empty(device=device)
            m.lora_B.to_empty(device=device)
            n += 1
    init_lora_weights_post_materialization(net)
    return n


def freeze_all(net) -> None:
    for p in net.parameters():
        p.requires_grad_(False)


def trainable_summary(module: nn.Module) -> dict:
    total = sum(p.numel() for p in module.parameters())
    trainable = [(n, p.numel()) for n, p in module.named_parameters() if p.requires_grad]
    groups: dict[str, int] = {}
    for n, k in trainable:
        head = n.split(".")[0] if "lora_" not in n else "lora"
        groups[head] = groups.get(head, 0) + k
    return {"total": total, "trainable": sum(k for _, k in trainable), "groups": groups, "names": [n for n, _ in trainable]}


def init_linear_(lin: nn.Linear, fan_in: int) -> None:
    std = 1.0 / math.sqrt(fan_in)
    nn.init.trunc_normal_(lin.weight, std=std, a=-3 * std, b=3 * std)
    if lin.bias is not None:
        nn.init.zeros_(lin.bias)


# ---------------------------------------------------------------------------
# Nano action head (DomainAwareLinear) -> 9-D Linear initialization
# ---------------------------------------------------------------------------
NANO_ACTION_DIM = 64  # Nano's padded max_action_dim (transformer/config.json: action_dim 64)


def load_nano_action_head(checkpoint_path: str | Path, domain: str = "umi", n_channels: int = 9) -> dict:
    """Slice one embodiment domain's Cosmos3-Nano action projections to our first ``n_channels``.

    Nano stores ``action_proj_in.fc.weight [num_domains, 64*4096]`` viewed as ``[in=64, out=4096]``
    (the official ``DomainAwareLinear.forward`` computes ``x @ weight``), ``action_proj_out.fc.weight``
    viewed as ``[in=4096, out=64]``, per-domain biases, and ``action_modality_embed [4096]``.
    Domains such as ``umi`` / ``droid_lerobot`` use ``[pos_delta(3), rot6d_delta(6), gripper(1)]`` with
    ``delta_T = T_i^{-1} T_{i+1}`` and column-based rot6d — the same convention as our 9-D actions, so
    channels 0..8 map one to one (gripper dropped)."""
    from safetensors import safe_open

    from cosmos_framework.data.generator.action.utils.domain_utils import EMBODIMENT_TO_DOMAIN_ID, EMBODIMENT_TO_RAW_ACTION_DIM

    if domain not in EMBODIMENT_TO_DOMAIN_ID:
        raise ValueError(f"unknown Nano embodiment domain {domain!r}")
    if EMBODIMENT_TO_RAW_ACTION_DIM[domain] < n_channels:
        raise ValueError(f"domain {domain} has only {EMBODIMENT_TO_RAW_ACTION_DIM[domain]} action channels")
    d = EMBODIMENT_TO_DOMAIN_ID[domain]
    ckpt = Path(checkpoint_path)
    want = {"action_proj_in.fc.weight", "action_proj_in.bias.weight", "action_proj_out.fc.weight",
            "action_proj_out.bias.weight", "action_modality_embed"}
    found = {}
    for rel, f in _iter_safetensors(ckpt):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for k in sf.keys():
                if k in want:
                    found[k] = sf.get_tensor(k)
    missing = want - set(found)
    if missing:
        raise KeyError(f"Nano checkpoint lacks {sorted(missing)}")
    w_in_all = found["action_proj_in.fc.weight"]  # [N, 64*D]
    w_out_all = found["action_proj_out.fc.weight"]  # [N, D*64]
    D = found["action_modality_embed"].numel()
    w_in = w_in_all[d].view(NANO_ACTION_DIM, D)[:n_channels, :]  # [9, D]  (x @ w)
    b_in = found["action_proj_in.bias.weight"][d]  # [D]
    w_out = w_out_all[d].view(D, NANO_ACTION_DIM)[:, :n_channels]  # [D, 9]
    b_out = found["action_proj_out.bias.weight"][d][:n_channels]  # [9]
    return {
        "domain": domain, "domain_id": d, "hidden": D,
        "in_weight": w_in.t().contiguous().float(),   # nn.Linear(9, D).weight is [D, 9]
        "in_bias": b_in.float(),
        "out_weight": w_out.t().contiguous().float(),  # nn.Linear(D, 9).weight is [9, D]
        "out_bias": b_out.float(),
        "modality_embed": found["action_modality_embed"].float(),
    }
