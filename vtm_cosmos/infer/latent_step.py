"""Experimental direct-latent rollout: keep two history latents, commit one per step."""

import torch

from vtm_cosmos.streams import IMAGE_STREAMS, HISTORY_LATENTS, FRAMES_PER_LATENT


@torch.no_grad()
def generate_latent_rollout(sampler, initial_conditions, actions_at_step, num_steps, seed=0, progress=None):
    """Return initial history plus generated latents, without any VAE round trip.

    actions_at_step(k) supplies physical recorded actions for the window starting
    at t0 + 4*k. It must not supply observation tensors. Local RoPE ticks restart
    with each window, matching the trained F0 prefix layout.

    Seeds reproduce the old per-latent schedule: 0,1,1,2,2,3,... for seed=0.
    Thus each absolute generated block uses the same initial noise as the old
    two-latent chunk rollout, although its observation context now differs.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    present = tuple(s for s in IMAGE_STREAMS if s in initial_conditions)
    histories = {s: initial_conditions[s][:, :, :HISTORY_LATENTS].clone() for s in present}
    chunks = {s: [histories[s].cpu()] for s in present}
    records = []
    for k in range(num_steps):
        actions = actions_at_step(k)
        if set(actions).intersection(IMAGE_STREAMS):
            raise ValueError("The action provider must not supply observations")
        conditions = dict(actions)
        for s in present:
            # Preserve the full-window input shape; only F0 is integrated.
            z = torch.zeros_like(initial_conditions[s])
            z[:, :, :HISTORY_LATENTS] = histories[s]
            conditions[s] = z
        step_seed = int(seed) + k // 2 + k % 2
        out = sampler.sample_next(conditions, seed=step_seed)
        for s in present:
            new = out[s][:, :, HISTORY_LATENTS:HISTORY_LATENTS + 1]
            if new.shape[2] != 1 or not torch.isfinite(new).all():
                raise ValueError(f"Invalid generated latent: step={k}, stream={s}")
            if not torch.equal(out[s][:, :, :HISTORY_LATENTS], histories[s]):
                raise ValueError("Sampler changed fixed history")
            chunks[s].append(new.cpu())
            histories[s] = torch.cat([histories[s][:, :, 1:], new], dim=2)
        record = dict(step=k, window_offset_frames=FRAMES_PER_LATENT*k, seed=step_seed,
                      generated_frame_start=FRAMES_PER_LATENT*(HISTORY_LATENTS+k),
                      committed_latents=1, history_latents=HISTORY_LATENTS)
        records.append(record)
        if progress:
            progress(record)
    return {s: torch.cat(chunks[s], dim=2) for s in present}, records
