"""Flow-matching corruption of the FUTURE only (spec §8.2).

    x_sigma = (1 - sigma) * x0 + sigma * eps,   v_target = eps - x0
Identical to the upstream ``rectified_flow.get_interpolation(x_0=noise, x_1=clean, t)``.
Applied once, after reading the clean cache and before patchify / projection; history
positions are left untouched (effective sigma 0). One sigma per sample shared by all
streams; Gaussian noise drawn independently per stream.
"""

from __future__ import annotations

import torch

from vtm_cosmos.model.layout import TokenLayout
from vtm_cosmos.streams import ACTION_STREAMS, ALL_ACTION_STREAMS, IMAGE_STREAMS, STREAMS


def sample_sigma(
    batch_size: int,
    device: torch.device,
    distribution: str = "waver",
    shift: float = 3.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Cosmos3 training recipe via the official TrainTimeSampler ('waver' for video, shift 3 at 256px)."""
    import cosmos_framework.model.generator.algorithm.loss  # noqa: F401  (breaks the upstream circular import)
    from cosmos_framework.model.generator.diffusion.rectified_flow import TrainTimeSampler

    sampler = TrainTimeSampler(distribution=distribution)
    shifts = torch.full((batch_size,), float(shift), dtype=torch.float32)
    return sampler(batch_size, device=device, dtype=torch.float32, generator=generator, shifts=shifts).clamp(0.0, 1.0)


def make_noisy_state(x0: torch.Tensor, target_mask: torch.Tensor, sigma_b: torch.Tensor, noise: torch.Tensor):
    """x0: [B,...]; target_mask broadcastable bool; sigma_b: [B]. Returns (state, velocity_target)."""
    sigma = sigma_b.reshape((-1,) + (1,) * (x0.ndim - 1)).to(x0)
    mixed = (1.0 - sigma) * x0 + sigma * noise
    state = torch.where(target_mask, mixed, x0)
    return state, noise - x0


def build_training_states(
    latents: dict[str, torch.Tensor],
    actions_norm: dict[str, torch.Tensor],
    sigma_b: torch.Tensor,
    layout: TokenLayout,
    generator: torch.Generator | None = None,
):
    """Clean latents [B,48,4,16,16] and normalized actions [B,15,9] -> (states, targets, masks).

    A causal prefix slices the cache first and targets only its final image block;
    all earlier blocks are clean teacher-forced context. Otherwise masks select the
    usual image/action future positions. Validity is intersected by the loss."""
    states, targets, masks = {}, {}, {}
    if layout.cfg.target_index is not None:
        prefix = layout.slice_states({**latents, **actions_norm})
        latents = {s: x for s, x in prefix.items() if s in IMAGE_STREAMS}
        actions_norm = {s: x for s, x in prefix.items() if s not in IMAGE_STREAMS}
    for name in layout.active:  # absent/dropped streams: no GT read, no noise drawn, no tokens
        x0 = latents[name].float() if name in IMAGE_STREAMS else actions_norm[name].float()
        if layout.cfg.task_mode == "action_full" and name in ALL_ACTION_STREAMS:
            if not torch.isfinite(x0).all():
                raise ValueError("action_full requires finite action conditions")
            states[name] = x0.clone()
            continue
        m = layout.target_mask(name).to(x0.device)
        if bool(m.any()):
            noise = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype)
            state, vt = make_noisy_state(x0, m, sigma_b, noise)
        else:  # forward-dynamics actions: known clean condition, no target
            state, vt = x0.clone(), torch.zeros_like(x0)
        states[name], targets[name], masks[name] = state, vt, m.expand_as(x0)
    return states, targets, masks


def build_sampling_states(
    conditions: dict[str, torch.Tensor],
    layout: TokenLayout,
    generator: torch.Generator | None = None,
):
    """Initial sampler states: Gaussian noise on targets, clean conditions elsewhere (spec §9.1).
    ``conditions`` may hold zeros at unknown future positions; they are never read."""
    states, masks = {}, {}
    for name in layout.active:
        c = conditions[name].float()
        if layout.cfg.task_mode == "action_full" and name in ALL_ACTION_STREAMS:
            if not torch.isfinite(c).all():
                raise ValueError("action_full requires finite action conditions")
            states[name] = c.clone()
            continue
        m = layout.target_mask(name).to(c.device).expand_as(c)
        eps = torch.randn(c.shape, generator=generator, device=c.device, dtype=c.dtype)
        states[name] = torch.where(m, eps, c)
        masks[name] = m
    return states, masks


def action_valid_mask(valid: dict[str, torch.Tensor], like: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """[B,15] bool per hand -> broadcast to [B,15,9]."""
    return {s: valid[s].to(like[s].device)[:, :, None].expand_as(like[s]) for s in ACTION_STREAMS if s in like}
