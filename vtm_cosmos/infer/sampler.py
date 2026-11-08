"""Window sampling with fixed history and known actions.

Legacy mode integrates all future targets together. The causal mode completes
one latent-time block before advancing to the next, with all current streams
integrated jointly. Both use official UniPC (default, 35 steps per solve) or Euler.

The sampler never sees future ground truth: ``conditions`` may carry zeros at the
target positions; they are replaced by Gaussian noise before the first network call."""

from __future__ import annotations

import torch

from vtm_cosmos.data.actions import ActionNormalizer, ActionNormalizers
from vtm_cosmos.model.generator_adapter import CosmosGeneratorAdapter
from vtm_cosmos.model.layout import TokenLayout
from vtm_cosmos.streams import ALL_STREAMS, ALL_ACTION_STREAMS, ACTION_STREAMS, IMAGE_STREAMS, STREAMS
from vtm_cosmos.train.noising import build_sampling_states


class FlatTargets:
    """Bijection between the per-stream target elements and one flat vector."""

    def __init__(self, masks: dict[str, torch.Tensor]):
        self.masks = masks
        self.streams = tuple(s for s in STREAMS if s in masks)  # active streams only
        self.sizes = {s: int(masks[s].sum()) for s in self.streams}
        self.total = sum(self.sizes.values())

    def flatten(self, states: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([states[s][self.masks[s]] for s in self.streams if self.sizes[s] > 0])

    def unflatten(self, flat: torch.Tensor, base: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out, i = dict(base), 0
        for s in self.streams:
            x = base[s].clone()
            n = self.sizes[s]
            if n > 0:
                x[self.masks[s]] = flat[i : i + n].to(x.dtype)
                i += n
            out[s] = x
        return out


class WindowSampler:
    def __init__(
        self,
        adapter: CosmosGeneratorAdapter,
        normalizer: ActionNormalizer,
        num_steps: int = 35,
        solver: str = "unipc",
        shift: float = 1.0,
        amp_bf16: bool = True,
    ):
        self.adapter = adapter
        self.normalizer = ActionNormalizers.wrap(normalizer)  # sensor + (action_full) view normalizers
        self.num_steps = int(num_steps)
        self.solver = solver
        self.shift = float(shift)
        self.amp_bf16 = amp_bf16

    def _velocity(self, states, sigma_b, layout):
        device = sigma_b.device
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(self.amp_bf16 and device.type == "cuda")):
            return self.adapter(states, sigma_b, layout)

    @torch.no_grad()
    def sample_next(self, conditions: dict[str, torch.Tensor], seed: int = 0) -> dict:
        """Generate only F0 from two history latents; return the three-latent prefix.

        This is exactly the first solve of ``sample``, with the same normalization,
        causal action cutoff, and random seed. Future placeholders are never read.
        The caller decides how to advance history after committing this latent.
        """
        adapter = self.adapter
        if adapter.cfg.prediction_strategy != "causal_teacher_forcing" or adapter.cfg.task_mode != "action_full":
            raise ValueError("sample_next requires a causal action_full checkpoint")
        active = tuple(s for s in ALL_STREAMS if s in conditions)
        batch = conditions[IMAGE_STREAMS[0]].shape[0]
        layout = adapter.layout_for(batch, adapter.device, active=active, target_index=2)
        cond = {s: conditions[s].to(adapter.device).float() for s in IMAGE_STREAMS if s in active}
        cond.update({s: self.normalizer.normalize(conditions[s].to(adapter.device).float(), s)
                     for s in ALL_ACTION_STREAMS if s in active})
        return self._integrate(layout.slice_states(cond), layout, int(seed))

    @torch.no_grad()
    def sample(self, conditions: dict[str, torch.Tensor], seed: int = 0, layout: TokenLayout | None = None) -> dict:
        """Full-window conditions: images [B,48,4,16,16], physical actions (15/16 rows).

        action_full requires all actions; the causal mode reveals only prefixes to
        each solve. Returns full images and actions de-normalized to physical units.
        """
        adapter = self.adapter
        device = adapter.device
        B = conditions[IMAGE_STREAMS[0]].shape[0]
        # active streams = what the history actually provides (spec-MM §7); never inferred from future GT
        active = tuple(s for s in ALL_STREAMS if s in conditions)
        layout = layout or adapter.layout_for(B, device, active=active)
        if set(layout.active) != set(active):
            raise ValueError(f"layout.active {layout.active} != conditions {active}")
        if layout.cfg != adapter.layout_cfg:
            raise ValueError("WindowSampler requires the full-window layout, not a prefix or modified contract")
        cond = {s: conditions[s].to(device).float() for s in IMAGE_STREAMS if s in active}
        cond.update({s: self.normalizer.normalize(conditions[s].to(device).float(), s)
                     for s in ALL_ACTION_STREAMS if s in active})  # every action stream is normalized
        if adapter.cfg.prediction_strategy == "causal_teacher_forcing":
            out = {s: x.clone() for s, x in cond.items()}
            for target_index in (2, 3):
                prefix_layout = adapter.layout_for(B, device, active=active, target_index=target_index)
                prefix = prefix_layout.slice_states(out)
                stage = self._integrate(prefix, prefix_layout, int(seed) + target_index - 2)
                for s in IMAGE_STREAMS:
                    if s in out:
                        out[s][:, :, target_index:target_index + 1] = stage[s][:, :, target_index:target_index + 1]
        else:
            out = self._integrate(cond, layout, seed)
        result = {s: out[s] for s in IMAGE_STREAMS if s in out}
        for s in ALL_ACTION_STREAMS:
            if s in out:
                result[s] = self.normalizer.denormalize(out[s], s)
        result["_masks"] = {s: layout.target_mask(s).to(device).expand_as(out[s]) for s in out
                            if not (layout.cfg.task_mode == "action_full" and s in ALL_ACTION_STREAMS)}
        result["_active"] = active
        return result

    def _integrate(self, cond: dict[str, torch.Tensor], layout: TokenLayout, seed: int):
        """Solve one target set while reconstructing all non-target inputs from fixed conditions."""
        device, B = self.adapter.device, layout.batch_size
        gen = torch.Generator(device=device).manual_seed(int(seed))
        states, masks = build_sampling_states(cond, layout, generator=gen)
        flat = FlatTargets(masks)
        x = flat.flatten(states)

        def velocity_fn(x_flat: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            # UniPC passes int64 timesteps in [0, 1000]; Euler passes float sigma directly
            t = timestep.float().reshape(-1)[0]
            sigma = (t / 1000.0) if timestep.dtype in (torch.int64, torch.int32) else t
            sigma_b = sigma.to(device).expand(B).clone()
            st = flat.unflatten(x_flat, cond)
            vel = self._velocity(st, sigma_b, layout)
            return flat.flatten(vel).to(x_flat.dtype)

        if self.solver == "unipc":
            x_out = unipc_sample(velocity_fn, x, self.num_steps, self.shift, seed=int(seed))
        elif self.solver == "euler":
            x_out = euler_sample(velocity_fn, x, self.num_steps, self.shift)
        else:
            raise ValueError(self.solver)
        out = flat.unflatten(x_out, cond)
        return out


def unipc_sample(velocity_fn, x: torch.Tensor, num_steps: int, shift: float = 1.0, seed: int = 0) -> torch.Tensor:
    """The loop of the official ``UniPCSampler.forward`` (samplers/unipc.py) on the official
    ``FlowUniPCMultistepScheduler``; inlined because the wrapper module imports the whole
    cosmos_framework.utils.config stack (storage clients) that this project does not need.
    Timesteps are int64 in [0, 1000]; the scheduler's flow prediction expects v = noise - x0."""
    from cosmos_framework.model.generator.diffusion.samplers.fm_solvers_unipc import FlowUniPCMultistepScheduler

    device = x.device
    gen = torch.Generator(device=device).manual_seed(int(seed))
    sched = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=shift, use_dynamic_shifting=False)
    sched.set_timesteps(num_steps, device=device, shift=shift)
    latent = x
    for timestep in sched.timesteps:
        v = velocity_fn(latent, timestep.reshape(1, 1))
        latent = sched.step(model_output=v, timestep=timestep, sample=latent.unsqueeze(0), return_dict=False, generator=gen)[0].squeeze(0)
    return latent


def euler_sample(velocity_fn, x: torch.Tensor, num_steps: int, shift: float = 1.0) -> torch.Tensor:
    """Reference integrator: sigma 1 -> 0 on a shifted grid, x <- x + (s_next - s_now) * v."""
    t = torch.linspace(1.0, 0.0, num_steps + 1, device=x.device)
    sig = shift * t / (1.0 + (shift - 1.0) * t)
    for s_now, s_next in zip(sig[:-1], sig[1:]):
        v = velocity_fn(x, s_now.reshape(1, 1))
        x = x + (s_next - s_now) * v
    return x
