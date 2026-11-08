"""Stream registry shared by packing, masking, output splitting and evaluation.

Nothing downstream is allowed to hard-code the numbers here: layouts derive
offsets from this table (spec §5), masks read STREAM_ALLOW (spec §7), and the
time coordinates follow the 15 FPS extraction contract (spec §6).
"""

from __future__ import annotations

import torch
from vtm_cosmos.tasks import canonical_task_mode

STREAMS: tuple[str, ...] = (
    "video",
    "wrist_left",
    "wrist_right",
    "tactile_left",
    "tactile_right",
    "action_left",
    "action_right",
)
IMAGE_STREAMS: tuple[str, ...] = STREAMS[:5]
ACTION_STREAMS: tuple[str, ...] = STREAMS[5:]
RGB_STREAMS: tuple[str, ...] = ("video", "wrist_left", "wrist_right")
TACTILE_STREAMS: tuple[str, ...] = ("tactile_left", "tactile_right")
# Keep the legacy seven-stream schema available for old checkpoints.
VIEW_ACTION_STREAMS = ("view_action_left", "view_action_right")
ALL_ACTION_STREAMS = ACTION_STREAMS + VIEW_ACTION_STREAMS
ALL_STREAMS = STREAMS + VIEW_ACTION_STREAMS
STREAM_ID: dict[str, int] = {name: i for i, name in enumerate(ALL_STREAMS)}
# Main view and wrists share calibrated main-view actions; tactile uses sensor deltas.
MODALITY_ACTIONS = {
    "video": VIEW_ACTION_STREAMS,
    "wrist_left": ("view_action_left",), "wrist_right": ("view_action_right",),
    "tactile_left": ("action_left",), "tactile_right": ("action_right",),
}
NUM_STREAMS = len(STREAMS)

# Missing-modality support (spec-MM): only the wrist cameras may be absent from a recording or
# dropped as augmentation; every other stream is required. Stream IDs never get renumbered.
WRIST_STREAMS: tuple[str, ...] = ("wrist_left", "wrist_right")
OPTIONAL_STREAMS: tuple[str, ...] = WRIST_STREAMS
REQUIRED_STREAMS: tuple[str, ...] = tuple(s for s in STREAMS if s not in OPTIONAL_STREAMS)


def active_pattern(active) -> tuple[bool, ...]:
    """Legacy 7-slot or action_full 9-slot pattern, with stable existing stream IDs."""
    act = set(active)
    unknown = act - set(ALL_STREAMS)
    if unknown:
        raise ValueError(f"unknown streams {sorted(unknown)}")
    return tuple(s in act for s in (ALL_STREAMS if act.intersection(VIEW_ACTION_STREAMS) else STREAMS))


def streams_from_pattern(pattern) -> tuple[str, ...]:
    return tuple(s for s, on in zip(ALL_STREAMS, pattern) if on)


def check_required(active, allowed=None) -> None:
    """``allowed`` (model.active_streams) narrows the required set to the streams a run models at all."""
    required = REQUIRED_STREAMS if allowed is None else tuple(s for s in REQUIRED_STREAMS if s in set(allowed))
    missing = [s for s in required if s not in set(active)]
    if missing:
        raise ValueError(f"required streams missing: {missing} (only {OPTIONAL_STREAMS} may be absent)")


def normalize_active_streams(task_mode: str, streams) -> tuple[str, ...] | None:
    """Validate ``model.active_streams`` (None = every stream of the task, the historical behavior).
    A run may model a subset of streams (e.g. main view + view actions only); the main view is
    always required, and action_full keeps both view action streams. Canonical order."""
    if streams is None:
        return None
    task_mode = canonical_task_mode(task_mode)
    act = set(streams)
    unknown = act - set(ALL_STREAMS)
    if unknown:
        raise ValueError(f"active_streams: unknown streams {sorted(unknown)}")
    if "video" not in act:
        raise ValueError("active_streams must include 'video'")
    if task_mode == "action_full" and not set(VIEW_ACTION_STREAMS).issubset(act):
        raise ValueError("active_streams: action_full requires both view action streams")
    if task_mode != "action_full" and act.intersection(VIEW_ACTION_STREAMS):
        raise ValueError("active_streams: view action streams require action_full")
    return tuple(s for s in ALL_STREAMS if s in act)

# Query stream (row) may read key stream (column). Spec §7 table.
#            V  WL WR TL TR AL AR
STREAM_ALLOW: tuple[tuple[int, ...], ...] = (
    (1, 1, 1, 1, 1, 1, 1),  # video
    (1, 1, 0, 1, 0, 1, 0),  # wrist_left
    (1, 0, 1, 0, 1, 0, 1),  # wrist_right
    (1, 1, 0, 1, 0, 1, 0),  # tactile_left
    (1, 0, 1, 0, 1, 0, 1),  # tactile_right
    (1, 1, 0, 1, 0, 1, 0),  # action_left
    (1, 0, 1, 0, 1, 0, 1),  # action_right
)


def stream_allow_tensor(device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.tensor(STREAM_ALLOW, dtype=torch.bool, device=device)


# ---- latent cache contract (spec §3) -------------------------------------
LATENT_CHANNELS = 48
LATENT_T = 4  # stored latents per window (anchor dropped)
LATENT_HW = 16
PATCH = 2
PATCH_DIM = PATCH * PATCH * LATENT_CHANNELS  # 192
TOKENS_PER_LATENT_FRAME = (LATENT_HW // PATCH) ** 2  # 64
IMAGE_TOKENS = LATENT_T * TOKENS_PER_LATENT_FRAME  # 256
HISTORY_LATENTS = 2
FUTURE_LATENTS = 2
FRAMES_PER_LATENT = 4
WINDOW_FRAMES = 16

# ---- action contract (spec §4) -------------------------------------------
ACTION_DIM = 9
N_ACTIONS = WINDOW_FRAMES - 1  # a1..a15
HISTORY_ACTIONS = 7  # a1..a7 ; a8..a15 are future
FUTURE_ACTIONS = N_ACTIONS - HISTORY_ACTIONS

# ---- physical time (spec §6) ---------------------------------------------
EXTRACTION_FPS = 15.0
COSMOS_BASE_FPS = 24.0
COSMOS_TEMPORAL_COMPRESSION = 4
BASE_TPS = COSMOS_BASE_FPS / COSMOS_TEMPORAL_COMPRESSION  # 6.0, do NOT change with extraction fps

# Integer frame ticks used by the causal mask: latent i represents raw frames
# [4i, 4i+3] and carries the END frame index; action k (0-based row) is the
# transition frame k -> k+1 and carries its end frame k+1 (= a_{k+1} in the spec).
IMAGE_TICKS: tuple[int, ...] = tuple(FRAMES_PER_LATENT * i + (FRAMES_PER_LATENT - 1) for i in range(LATENT_T))
ACTION_TICKS: tuple[int, ...] = tuple(k + 1 for k in range(N_ACTIONS))


def tick_to_seconds(tick: int | torch.Tensor, fps: float = EXTRACTION_FPS):
    return tick / fps


def seconds_to_rope_time(seconds, base_tps: float = BASE_TPS, common_offset: float = 0.0):
    return common_offset + base_tps * seconds


# ---- on-disk key mapping for the Wan2.2 latent trees ----------------------
# stream -> (subdir under the latent root, tensor key inside the episode .pt)
LATENT_FILE_KEYS: dict[str, tuple[str, str]] = {
    "video": ("visual", "z_visual_middle"),
    "wrist_left": ("wrist", "z_wrist_left"),
    "wrist_right": ("wrist", "z_wrist_right"),
    "tactile_left": ("tac", "z_tactile_left"),
    "tactile_right": ("tac", "z_tactile_right"),
}
# stream -> key used by wan22-vae/pad_latent.py (PadLatents) and decode helpers
PAD_STREAM_KEY: dict[str, str] = {
    "video": "view",
    "wrist_left": "wl",
    "wrist_right": "wr",
    "tactile_left": "tl",
    "tactile_right": "tr",
}
# action stream -> key inside sensor_actions_*.pt
ACTION_FILE_KEYS: dict[str, tuple[str, str]] = {
    "action_left": ("action_left", "valid_left"),
    "action_right": ("action_right", "valid_right"),
}
# legacy 36-D view+tactile action rows: hand slices (dataset.py asserts this layout)
LEGACY36_SLICES: dict[str, tuple[int, int]] = {"action_left": (18, 27), "action_right": (27, 36)}


def baseline_token_count(task_mode: str = "action_causal") -> dict[str, int]:
    """Reference numbers from spec §5.3 (full-modality, no AR, no padding)."""
    task_mode = canonical_task_mode(task_mode)
    total = len(IMAGE_STREAMS) * IMAGE_TOKENS + len(ACTION_STREAMS) * N_ACTIONS
    if task_mode == "action_full":
        total += len(VIEW_ACTION_STREAMS) * WINDOW_FRAMES
    hist_img = len(IMAGE_STREAMS) * HISTORY_LATENTS * TOKENS_PER_LATENT_FRAME
    fut_img = len(IMAGE_STREAMS) * FUTURE_LATENTS * TOKENS_PER_LATENT_FRAME
    if task_mode == "action_causal":
        hist_act = len(ACTION_STREAMS) * HISTORY_ACTIONS
        fut_act = len(ACTION_STREAMS) * FUTURE_ACTIONS
    elif task_mode in ("forward_dynamics", "action_full"):
        hist_act = len(ACTION_STREAMS) * N_ACTIONS
        if task_mode == "action_full":
            hist_act += len(VIEW_ACTION_STREAMS) * WINDOW_FRAMES
        fut_act = 0
    else:
        raise ValueError(task_mode)
    return {"total": total, "history": hist_img + hist_act, "future": fut_img + fut_act}
