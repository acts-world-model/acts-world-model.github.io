"""Diffusers-checkpoint -> Cosmos3VFMNetwork key mapping.

Verbatim copy of the tables in cosmos-framework ``cosmos_framework/inference/model.py``
(``_DIFFUSERS_KEY_MAPPING_RES`` / ``_DIFFUSERS_NET_KEY_PREFIXES`` / ``_DIFFUSERS_NET_KEYS``,
commit 3f94d25). Copied because importing that module drags in the storage-client stack
(multistorageclient, easy_io) which this project does not need."""

from __future__ import annotations

import re

_DROP_WEIGHT_PATH_RES = (re.compile(r"^(?!transformer/|vision_encoder/)"),)
_DROP_KEY_RES = (
    re.compile(r"^(?:feature_extractor|image_processor|scheduler|sound_tokenizer|text_encoder|tokenizer|vae)\."),
)
_KEY_MAPPING_RES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^transformer\."), ""),
    (re.compile(r"^vision_encoder\."), ""),
    (re.compile(r"^model\.net\."), ""),
    (re.compile(r"^action_proj_in\."), "action2llm."),
    (re.compile(r"^action_proj_out\."), "llm2action."),
    (re.compile(r"^audio_proj_in\."), "sound2llm."),
    (re.compile(r"^audio_proj_out\."), "llm2sound."),
    (re.compile(r"^audio_modality_embed$"), "sound_modality_embed"),
    (re.compile(r"^proj_in\."), "vae2llm."),
    (re.compile(r"^proj_out\."), "llm2vae."),
    (re.compile(r"^time_embedder\.linear_1\."), "time_embedder.mlp.0."),
    (re.compile(r"^time_embedder\.linear_2\."), "time_embedder.mlp.2."),
    (re.compile(r"\.self_attn\.to_q\."), ".self_attn.q_proj."),
    (re.compile(r"\.self_attn\.to_k\."), ".self_attn.k_proj."),
    (re.compile(r"\.self_attn\.to_v\."), ".self_attn.v_proj."),
    (re.compile(r"\.self_attn\.to_out\."), ".self_attn.o_proj."),
    (re.compile(r"\.self_attn\.norm_q\."), ".self_attn.q_norm."),
    (re.compile(r"\.self_attn\.norm_k\."), ".self_attn.k_norm."),
    (re.compile(r"\.self_attn\.add_q_proj\."), ".self_attn.q_proj_moe_gen."),
    (re.compile(r"\.self_attn\.add_k_proj\."), ".self_attn.k_proj_moe_gen."),
    (re.compile(r"\.self_attn\.add_v_proj\."), ".self_attn.v_proj_moe_gen."),
    (re.compile(r"\.self_attn\.to_add_out\."), ".self_attn.o_proj_moe_gen."),
    (re.compile(r"\.self_attn\.norm_added_q\."), ".self_attn.q_norm_moe_gen."),
    (re.compile(r"\.self_attn\.norm_added_k\."), ".self_attn.k_norm_moe_gen."),
    (re.compile(r"^model\.lm_head\."), "language_model.lm_head."),
    (re.compile(r"^lm_head\."), "language_model.lm_head."),
    (re.compile(r"^model\.visual\."), "language_model.visual."),
    (re.compile(r"^visual\."), "language_model.visual."),
    (
        re.compile(r"^(blocks\.|deepstack_merger_list\.|merger\.|patch_embed\.|pos_embed\.)(.*)$"),
        r"language_model.visual.\1\2",
    ),
    (
        re.compile(
            r"^language_model\.(?!model\.|lm_head\.|visual\.)(embed_tokens\.|layers\.|norm(?:_moe_gen)?\.)(.*)$"
        ),
        r"language_model.model.\1\2",
    ),
    (
        re.compile(r"^model\.(embed_tokens\.|layers\.|norm(?:_moe_gen)?\.)(.*)$"),
        r"language_model.model.\1\2",
    ),
    (
        re.compile(r"^(embed_tokens\.|layers\.|norm(?:_moe_gen)?\.)(.*)$"),
        r"language_model.model.\1\2",
    ),
)
_NET_KEY_PREFIXES = (
    "action2llm.",
    "action_pos_embed.",
    "language_model.",
    "latent_pos_embed.",
    "llm2action.",
    "llm2sound.",
    "llm2vae.",
    "sound2llm.",
    "time_embedder.",
    "vae2llm.",
)
_NET_KEYS = frozenset({"action_modality_embed", "latent_pos_embed", "sound_modality_embed"})


def _drop_path(path: str) -> bool:
    path = path.replace("\\", "/")
    return bool(path) and any(p.search(path) is not None for p in _DROP_WEIGHT_PATH_RES)


def _drop_key(name: str) -> bool:
    return any(p.search(name) is not None for p in _DROP_KEY_RES)


def diffusers_to_net_key(name: str, weight_path: str = "") -> str | None:
    """Rename a diffusers checkpoint key to its Cosmos3VFMNetwork key; None = not a generator weight."""
    if _drop_path(weight_path) or _drop_key(name):
        return None
    for pattern, repl in _KEY_MAPPING_RES:
        name = pattern.sub(repl, name)
    if _drop_key(name) or not (name in _NET_KEYS or name.startswith(_NET_KEY_PREFIXES)):
        return None
    return name
