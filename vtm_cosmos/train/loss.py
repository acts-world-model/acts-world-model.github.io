"""Per-stream future-only flow-matching loss (spec §8.4) with missing-stream-aware
normalization (spec-MM §6).

For each stream: MSE over valid target elements, normalized per sample, then averaged over
the samples that have at least one valid target *globally* (across DDP ranks and across the
active-pattern groups of a batch). Absent/dropped streams contribute no term at all, so the
weights of the base modalities are unchanged by how many samples carry a wrist."""

from __future__ import annotations

import torch

from vtm_cosmos.streams import ACTION_STREAMS, IMAGE_STREAMS, STREAMS


def per_sample_stream_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """pred/target/mask: [B,...]. Returns (per_sample [B] normalized SE, has_valid [B] bool)."""
    B = pred.shape[0]
    m = mask.bool().expand_as(pred)
    rows = m.reshape(B, -1).nonzero(as_tuple=True)[0]
    # Select valid elements before arithmetic, including for an empty selection.
    error = (pred[m].float() - target[m].float()).square()
    se = error.new_zeros(B).scatter_add(0, rows, error)
    cnt = m.reshape(B, -1).sum(1)
    has = cnt > 0
    per_sample = torch.where(has, se / cnt.clamp_min(1.0), torch.zeros_like(se))
    return per_sample, has


def target_counts(masks: dict[str, torch.Tensor], action_valid: dict[str, torch.Tensor] | None = None) -> dict[str, int]:
    """#samples with at least one valid target per stream (only streams present in masks)."""
    out = {}
    for name, m in masks.items():
        if action_valid is not None and name in action_valid:
            m = m & action_valid[name]
        B = m.shape[0]
        out[name] = int((m.reshape(B, -1).sum(1) > 0).sum())
    return out


def stream_loss_sums(
    pred: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    action_valid: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, int], dict[str, float]]:
    """Per-stream (sum over valid samples of the per-sample loss, #valid samples) for the streams
    present in ``pred``; plus detached translation/rotation splits for the action streams."""
    sums, counts, extra = {}, {}, {}
    for name in STREAMS:
        if name not in pred:
            continue
        m = masks[name]
        if action_valid is not None and name in action_valid:
            m = m & action_valid[name]
        ps, has = per_sample_stream_loss(pred[name], targets[name], m)
        sums[name] = ps.sum()
        counts[name] = int(has.sum())
        if name in ACTION_STREAMS and counts[name] > 0:
            mt = m.clone()
            mt[..., 3:] = False
            mr = m.clone()
            mr[..., :3] = False
            extra[f"loss/{name}_trans"] = float(per_sample_stream_loss(pred[name], targets[name], mt)[0].detach().sum() / counts[name])
            extra[f"loss/{name}_rot"] = float(per_sample_stream_loss(pred[name], targets[name], mr)[0].detach().sum() / counts[name])
    return sums, counts, extra


def combine_sums(
    sums: dict[str, torch.Tensor],
    counts: dict[str, int],
    global_counts: dict[str, int],
    weights: dict[str, float] | None,
    world_size: int,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """spec-MM §6: loss_local = Σ_m w_m · world · local_sum_m / max(global_count_m, 1).

    With the standard DDP gradient mean this equals the per-stream global mean over valid samples."""
    weights = weights or {}
    total = None
    logs = {}
    for name in STREAMS:
        if name not in sums:
            continue
        gc = max(int(global_counts.get(name, 0)), 1)
        term = sums[name] * (float(weights.get(name, 1.0)) * world_size / gc)
        total = term if total is None else total + term
        if counts.get(name, 0) > 0:
            logs[f"loss/{name}"] = float(sums[name].detach()) / counts[name]  # local per-sample mean, for logging
        logs[f"n/{name}"] = float(counts.get(name, 0))
    return total, logs


def flow_losses(
    pred: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
    action_valid: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Single-group, single-process convenience: per-stream mean over valid samples, weighted sum."""
    sums, counts, extra = stream_loss_sums(pred, targets, masks, action_valid)
    total, logs = combine_sums(sums, counts, counts, weights, 1)
    if total is None:
        total = next(iter(pred.values())).sum() * 0.0
    logs.update(extra)
    logs["loss/total"] = float(total.detach())
    img = [logs[f"loss/{s}"] for s in IMAGE_STREAMS if f"loss/{s}" in logs]
    logs["loss/image_mean"] = sum(img) / len(img) if img else float("nan")
    return total, logs
