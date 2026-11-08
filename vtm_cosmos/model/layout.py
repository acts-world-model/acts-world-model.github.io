"""Token layout: per-token metadata (sample / stream / time / space / target) and
the stream slices, computed from the stream registry instead of hard-coded.

Per-sample token order is stream-major following STREAMS; inside an image stream
tokens run t-major then h then w (the order of the official patchify
``cthpwq->thwpqc``); an action stream has one token per transition a_1..a_15.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch

from vtm_cosmos.streams import (
    ACTION_STREAMS, ALL_ACTION_STREAMS, ALL_STREAMS, VIEW_ACTION_STREAMS, WINDOW_FRAMES,
    ACTION_TICKS,
    BASE_TPS,
    EXTRACTION_FPS,
    FRAMES_PER_LATENT,
    HISTORY_ACTIONS,
    HISTORY_LATENTS,
    IMAGE_STREAMS,
    IMAGE_TICKS,
    LATENT_HW,
    LATENT_T,
    N_ACTIONS,
    PATCH,
    STREAM_ID,
    STREAMS,
    active_pattern,
    check_required,
    normalize_active_streams,
)

from vtm_cosmos.tasks import TASK_MODES, canonical_task_mode
from vtm_cosmos.model.topology import normalize_connections
TEMPORAL_MODES = ("causal", "chunk_bidirectional")


@dataclass
class LayoutConfig:
    task_mode: str = "action_causal"
    history_latents: int = HISTORY_LATENTS
    history_actions: int = HISTORY_ACTIONS
    latent_t: int = LATENT_T
    n_actions: int = N_ACTIONS
    grid: int = LATENT_HW // PATCH  # 8
    fps: float = EXTRACTION_FPS
    base_tps: float = BASE_TPS
    common_offset: float = 0.0
    image_ticks: tuple[int, ...] = IMAGE_TICKS
    action_ticks: tuple[int, ...] = ACTION_TICKS
    view_ticks: tuple[int, ...] = tuple(range(WINDOW_FRAMES))
    attention_topology: str = "legacy"
    attention_connections: dict[str, tuple[str, ...]] | None = None
    active_streams: tuple[str, ...] | None = None  # streams this run models at all (None = every stream)
    target_index: int | None = None

    def __post_init__(self):
        self.task_mode = canonical_task_mode(self.task_mode)
        if self.task_mode not in TASK_MODES:
            raise ValueError(f"task_mode must be one of {TASK_MODES}, got {self.task_mode}")
        self.active_streams = normalize_active_streams(self.task_mode, self.active_streams)
        if len(self.image_ticks) != self.latent_t or len(self.action_ticks) != self.n_actions:
            raise ValueError("tick tables must match latent_t / n_actions")
        self.attention_connections = normalize_connections(
            self.task_mode, self.attention_topology, self.attention_connections)

    def causal_prefix(self, target_index: int) -> "LayoutConfig":
        if self.task_mode != "action_full" or target_index not in (HISTORY_LATENTS, HISTORY_LATENTS + 1):
            raise ValueError("Causal prefixes require action_full and target_index 2 or 3")
        cutoff = self.image_ticks[target_index]
        actions = tuple(t for t in self.action_ticks if t <= cutoff)
        return replace(self, history_latents=target_index, latent_t=target_index + 1,
                       image_ticks=self.image_ticks[:target_index + 1], n_actions=len(actions),
                       action_ticks=actions, view_ticks=tuple(t for t in self.view_ticks if t <= cutoff),
                       target_index=target_index)


@dataclass
class TokenLayout:
    """Metadata for a batch of B identical samples packed sample-major, then padded."""

    cfg: LayoutConfig
    batch_size: int
    n_per_sample: int
    active: tuple[str, ...]
    n_real: int
    n_pad: int
    # per-sample tables (length n_per_sample), CPU long/float tensors
    stream_id_s: torch.Tensor
    tick_s: torch.Tensor
    time_s_s: torch.Tensor
    h_s: torch.Tensor
    w_s: torch.Tensor
    is_target_s: torch.Tensor
    slices: dict[str, tuple[int, int]] = field(default_factory=dict)

    # ---- derived batch tensors --------------------------------------------
    def _tile(self, x: torch.Tensor, pad_value) -> torch.Tensor:
        out = x.repeat(self.batch_size)
        if self.n_pad > self.n_real:
            pad = torch.full((self.n_pad - self.n_real,), pad_value, dtype=x.dtype)
            out = torch.cat([out, pad])
        return out

    def sample_id(self) -> torch.Tensor:  # (n_pad,) long, -1 on padding
        sid = torch.arange(self.batch_size).repeat_interleave(self.n_per_sample)
        if self.n_pad > self.n_real:
            sid = torch.cat([sid, torch.full((self.n_pad - self.n_real,), -1, dtype=torch.long)])
        return sid

    def stream_id(self) -> torch.Tensor:
        return self._tile(self.stream_id_s, 0)

    def tick(self) -> torch.Tensor:
        return self._tile(self.tick_s, 0)

    def is_target(self) -> torch.Tensor:  # (n_pad,) bool
        return self._tile(self.is_target_s, False)

    def valid(self) -> torch.Tensor:
        return self._tile(torch.ones(self.n_per_sample, dtype=torch.bool), False)

    def position_ids(self) -> torch.Tensor:
        """[3, n_real] float32 mRoPE coordinates (t = offset + base_tps * seconds, h, w)."""
        t = self.cfg.common_offset + self.cfg.base_tps * self.time_s_s
        pos = torch.stack([t, self.h_s.float(), self.w_s.float()], 0)  # [3, n_per_sample]
        return pos.repeat(1, self.batch_size).contiguous()

    def stream_index(self, name: str) -> torch.Tensor:
        """(B, n_s) long indices into the packed [n_real, D] tensor."""
        a, b = self.slices[name]
        base = torch.arange(self.batch_size) * self.n_per_sample
        return base[:, None] + torch.arange(a, b)[None, :]

    def target_mask(self, name: str) -> torch.Tensor:
        """Broadcastable bool mask over the stream's STATE tensor (image [1,1,T,1,1], action [1,T,1])."""
        if name in IMAGE_STREAMS:
            m = torch.zeros(1, 1, self.cfg.latent_t, 1, 1, dtype=torch.bool)
            m[:, :, self.cfg.history_latents :] = True
            return m
        m = torch.zeros(1, len(self.cfg.view_ticks) if name in VIEW_ACTION_STREAMS else self.cfg.n_actions, 1, dtype=torch.bool)
        if self.cfg.task_mode == "action_causal":
            m[:, self.cfg.history_actions :] = True
        return m

    def counts(self) -> dict[str, int]:
        t = int(self.is_target_s.sum())
        return {"total": self.n_per_sample, "history": self.n_per_sample - t, "future": t}

    def slice_states(self, states: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Select only the prefix represented by this layout, before any projection/noising."""
        return {s: states[s][:, :, :self.cfg.latent_t] if s in IMAGE_STREAMS else
                states[s][:, :self.slices[s][1] - self.slices[s][0]] for s in self.active}

    @property
    def pattern(self) -> tuple[bool, ...]:
        return active_pattern(self.active)


def build_layout(cfg: LayoutConfig, batch_size: int, align: int = 1, active=None) -> TokenLayout:
    """Token layout for the ACTIVE streams only (spec-MM §3-5): absent/dropped streams contribute
    no tokens, every remaining stream keeps its fixed stream ID, time ticks and spatial coords."""
    allowed = cfg.active_streams
    if active is None:
        active = allowed or (ALL_STREAMS if cfg.task_mode == "action_full" else STREAMS)
    active = tuple(s for s in ALL_STREAMS if s in set(active))  # canonical order
    if allowed is not None and not set(active).issubset(allowed):
        raise ValueError(f"streams {sorted(set(active) - set(allowed))} are not modeled by this run (active_streams)")
    if cfg.task_mode == "action_full" and not set(VIEW_ACTION_STREAMS).issubset(active):
        raise ValueError("action_full requires both view action streams")
    if cfg.task_mode != "action_full" and set(active).intersection(VIEW_ACTION_STREAMS):
        raise ValueError("view action streams require action_full")
    check_required(active, allowed)
    stream_id, tick, time_s, hh, ww, tgt = [], [], [], [], [], []
    slices = {}
    cursor = 0
    g = cfg.grid
    for name in active:
        sid = STREAM_ID[name]
        if name in IMAGE_STREAMS:
            n = cfg.latent_t * g * g
            for t in range(cfg.latent_t):
                tk = cfg.image_ticks[t]
                for h in range(g):
                    for w in range(g):
                        stream_id.append(sid)
                        tick.append(tk)
                        time_s.append(tk / cfg.fps)
                        hh.append(h)
                        ww.append(w)
                        tgt.append(t >= cfg.history_latents)
        elif name in ALL_ACTION_STREAMS:
            ticks = cfg.view_ticks if name in VIEW_ACTION_STREAMS else cfg.action_ticks
            n = len(ticks)
            for k, tk in enumerate(ticks):
                stream_id.append(sid)
                tick.append(tk)
                time_s.append(tk / cfg.fps)
                hh.append(0)
                ww.append(0)
                tgt.append(cfg.task_mode == "action_causal" and k >= cfg.history_actions)
        else:
            raise AssertionError(name)
        slices[name] = (cursor, cursor + n)
        cursor += n
    n_per = cursor
    n_real = n_per * batch_size
    n_pad = ((n_real + align - 1) // align) * align if align > 1 else n_real
    return TokenLayout(
        cfg=cfg,
        batch_size=batch_size,
        n_per_sample=n_per,
        active=active,
        n_real=n_real,
        n_pad=n_pad,
        stream_id_s=torch.tensor(stream_id, dtype=torch.long),
        tick_s=torch.tensor(tick, dtype=torch.long),
        time_s_s=torch.tensor(time_s, dtype=torch.float32),
        h_s=torch.tensor(hh, dtype=torch.long),
        w_s=torch.tensor(ww, dtype=torch.long),
        is_target_s=torch.tensor(tgt, dtype=torch.bool),
        slices=slices,
    )


def patchify(z: torch.Tensor, p: int = PATCH) -> torch.Tensor:
    """[B,C,T,H,W] -> [B, T*(H/p)*(W/p), p*p*C] with feature order (p q c), token order (t h w)."""
    B, C, T, H, W = z.shape
    z = z.reshape(B, C, T, H // p, p, W // p, p)  # b c t h p w q
    z = z.permute(0, 2, 3, 5, 4, 6, 1)  # b t h w p q c
    return z.reshape(B, T * (H // p) * (W // p), p * p * C)


def unpatchify(x: torch.Tensor, t: int, h: int, w: int, c: int, p: int = PATCH) -> torch.Tensor:
    """Inverse of patchify: [B, t*h*w, p*p*c] -> [B, c, t, h*p, w*p]."""
    B = x.shape[0]
    x = x.reshape(B, t, h, w, p, p, c)  # b t h w p q c
    x = x.permute(0, 6, 1, 2, 4, 3, 5)  # b c t h p w q
    return x.reshape(B, c, t, h * p, w * p)
