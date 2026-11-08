"""Training entry point.

    python -m vtm_cosmos.train.train --config configs/tiny_debug.yaml [key=value ...]

Distributed: reads torchrun (RANK/WORLD_SIZE/LOCAL_RANK) or Slurm (SLURM_PROCID/NTASKS/LOCALID)
environment; MASTER_ADDR/MASTER_PORT must be set by the launcher for multi-rank runs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import contextlib
import functools

from vtm_cosmos.data.actions import ActionNormalizer, ActionNormalizers
from vtm_cosmos.tasks import canonical_task_mode, prediction_contract, validate_prediction_config
from vtm_cosmos.data.dataset import VtmWindowDataset, WristDropout, collate_by_active, collect_action_rows
from vtm_cosmos.data.splits import load_split
from vtm_cosmos.data.proportional_sampler import ProportionalWristBatchSampler, wrist_pools
from vtm_cosmos.streams import ALL_ACTION_STREAMS, ACTION_STREAMS, IMAGE_STREAMS, STREAM_ID, STREAMS, WRIST_STREAMS, VIEW_ACTION_STREAMS
from vtm_cosmos.train.loss import combine_sums, stream_loss_sums, target_counts
from vtm_cosmos.train.model_factory import build_model, load_config
from vtm_cosmos.train.noising import action_valid_mask, build_training_states, sample_sigma


# ------------------------------------------------------------------ distributed
def init_distributed() -> tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank, world, local = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ and int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        rank, world, local = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"]), int(os.environ.get("SLURM_LOCALID", 0))
        os.environ.setdefault("RANK", str(rank))
        os.environ.setdefault("WORLD_SIZE", str(world))
        os.environ.setdefault("LOCAL_RANK", str(local))
    else:
        return 0, 1, 0
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", rank=rank, world_size=world)
    return rank, world, local


def all_reduce_mean(x: float, world: int, device) -> float:
    if world == 1:
        return x
    t = torch.tensor([x], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / world)


# ------------------------------------------------------------------ data
def build_datasets(cfg, rank: int, log):
    d = cfg.data
    split = load_split(d.split_file)
    train_eps = list(split["train"])
    if d.get("max_train_episodes"):
        train_eps = train_eps[: int(d.max_train_episodes)]
    val_eps = list(split["val"])
    if d.get("max_val_episodes"):
        val_eps = val_eps[: int(d.max_val_episodes)]
    optional = tuple(d.get("optional_streams", []) or [])
    action_full = canonical_task_mode(cfg.model.task_mode) == "action_full"
    common = dict(latent_root=d.latent_root, action_root=d.action_root, action_file=d.action_file,
                  optional_streams=optional,
                  include_view_actions=action_full,
                  view_action_file=d.get("view_action_file", "view_actions_fps15.pt"),
                  require_valid_actions=action_full,
                  active_streams=tuple(cfg.model.active_streams) if cfg.model.get("active_streams") else None)
    train_ds = VtmWindowDataset(episodes=train_eps, window_stride=int(d.get("train_window_stride", 1)),
                                max_windows_per_episode=d.get("max_windows_per_episode"), **common)
    val_ds = VtmWindowDataset(episodes=val_eps, window_stride=int(d.get("val_window_stride", 50)),
                              max_windows_per_episode=d.get("max_val_windows_per_episode"), **common)
    log(f"[data] train episodes={len(train_eps)} windows={len(train_ds)} | val episodes={len(val_eps)} windows={len(val_ds)}")
    if action_full:
        log(f"[data] action_full: dropped windows with an invalid action transition: "
            f"train {train_ds.dropped_invalid_windows} | val {val_ds.dropped_invalid_windows}")
    if optional:
        log(f"[data] optional streams {optional}: train {train_ds.presence_summary()} | val {val_ds.presence_summary()}")
    return train_ds, val_ds


def _fit_one(train_ds, cfg, streams, log, label: str) -> ActionNormalizer:
    rows = collect_action_rows(train_ds, window_stride=int(cfg.data.get("stats_window_stride", 8)),
                               max_windows=cfg.data.get("stats_max_windows"), streams=streams)
    mode = str(cfg.data.get("action_norm", "meanstd"))
    norm = ActionNormalizer.fit(rows, mode=mode, meta={"rows": int(rows.shape[0]), "episodes": list(train_ds.episodes),
                                                       "streams": list(streams), "kind": label, "hands": "left+right stacked"})
    log(f"[data] fitted {label} action normalizer ({mode}) on {rows.shape[0]} rows")
    return norm


def build_train_loader(cfg, train_ds, rank, world, collate_fn):
    bs, nw = int(cfg.training.batch_size), int(cfg.data.get("num_workers", 4))
    sampling = cfg.training.get("sampling", {}) or {}
    mode = str(sampling.get("mode", "uniform"))
    common = dict(num_workers=nw, pin_memory=True, persistent_workers=nw > 0, collate_fn=collate_fn)
    if mode == "dataset_proportional":
        wd = cfg.training.get("wrist_dropout", {}) or {}
        if float(wd.get("probability", 0.0)) != 0:
            raise ValueError("dataset_proportional requires wrist_dropout.probability=0 to preserve rank-aligned groups")
        if not sampling.get("steps_per_epoch"):
            raise ValueError("dataset_proportional requires training.sampling.steps_per_epoch")
        sampler = ProportionalWristBatchSampler.from_dataset(
            train_ds, batch_size=bs, steps_per_epoch=int(sampling.steps_per_epoch),
            rank=rank, world_size=world, seed=int(cfg.get("seed", 0)))
        # Worker/iterator startup must not consume the model's diffusion RNG on resume.
        generator = torch.Generator().manual_seed(int(cfg.get("seed", 0)) + rank)
        return DataLoader(train_ds, batch_sampler=sampler, generator=generator, **common), sampler
    if mode != "uniform":
        raise ValueError(f"Unknown training.sampling.mode: {mode}")
    causal = prediction_contract(cfg)[3] == "causal_teacher_forcing"
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True,
                                 seed=int(cfg.get("seed", 0))) if world > 1 or causal else None
    return DataLoader(train_ds, batch_size=bs, shuffle=sampler is None, sampler=sampler,
                      drop_last=True, generator=torch.Generator().manual_seed(int(cfg.get("seed", 0)) + rank)
                      if causal else None, **common), sampler


def capture_rng_state(device):
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng_state(state, device):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def sampling_checkpoint_state(sampler, world, device):
    if not isinstance(sampler, ProportionalWristBatchSampler):
        return None
    rng = capture_rng_state(device)
    states = [None] * world
    if world > 1:
        dist.all_gather_object(states, rng)
    else:
        states[0] = rng
    return {"sampler": sampler.state_dict(), "rng_by_rank": states}


def causal_checkpoint_state(data_contract, world, device):
    if data_contract is None:
        return None
    states = [None] * world
    rng = capture_rng_state(device)
    if world > 1:
        dist.all_gather_object(states, rng)
    else:
        states[0] = rng
    return {"data_contract": data_contract, "rng_by_rank": states}


def initialization_checkpoint(cfg):
    """Weights-only initialization may change prediction behavior, not the frozen base architecture."""
    path = cfg.training.get("init_from_checkpoint")
    if not path:
        return None
    if (cfg.get("data", {}) or {}).get("action_stats"):
        raise ValueError("Weights-only initialization restores checkpoint normalizers; do not set action_stats")
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    source = ck["config"]["model"]
    from omegaconf import OmegaConf
    target = OmegaConf.to_container(cfg.model, resolve=True)
    for key in ("language_model", "nano_checkpoint", "nano_json", "text_overrides", "lora"):
        if source.get(key) != target.get(key):
            raise ValueError(f"Weights-only initialization requires matching model.{key}")
    if canonical_task_mode(source.get("task_mode", "action_causal")) != cfg.model.task_mode:
        raise ValueError("Weights-only initialization requires the same action task and normalizers")
    if cfg.model.task_mode == "action_full" and ActionNormalizers.from_state_dict(ck["normalizer"]).view is None:
        raise ValueError("action_full initialization requires checkpoint view normalization")
    return ck


def restore_sampling_checkpoint(checkpoint, sampler, rank, world, device):
    saved = checkpoint.get("sampling_state")
    proportional = isinstance(sampler, ProportionalWristBatchSampler)
    if proportional != (saved is not None):
        raise ValueError("Cannot change sampling mode on resume; use a new run_name")
    if not proportional:
        return
    sampler.load_state_dict(saved["sampler"])
    if sampler.completed_steps != int(checkpoint["step"]) or len(saved["rng_by_rank"]) != world:
        raise ValueError("Checkpoint step or RNG ranks do not match sampler state")
    restore_rng_state(saved["rng_by_rank"][rank], device)


def _norm_to_buf(norm: ActionNormalizer | None, buf: torch.Tensor) -> None:
    """[3,9] slot: row0/row1 = (q01,q99) or (mean,std); row2[0] = mode flag, row2[1] = present flag."""
    if norm is None:
        return
    a, b = (norm.q01, norm.q99) if norm.mode == "quantile" else (norm.mean, norm.std)
    buf[0], buf[1] = a.to(buf), b.to(buf)
    buf[2, 0] = 1.0 if norm.mode == "quantile" else 0.0
    buf[2, 1] = 1.0


def _buf_to_norm(buf: torch.Tensor, meta: dict) -> ActionNormalizer | None:
    if float(buf[2, 1]) < 0.5:
        return None
    mode = "quantile" if float(buf[2, 0]) > 0.5 else "meanstd"
    return ActionNormalizer(buf[0].cpu(), buf[1].cpu(), mode=mode, meta=meta)


def get_normalizer(cfg, train_ds, out_dir: Path, rank: int, world: int, log, device=None) -> ActionNormalizers:
    """Rank 0 fits (or loads) the sensor-action normalizer and, for action_full, the view-action
    normalizer; both are broadcast to every rank so a slow NFS view of the freshly written JSON on
    other nodes cannot break start-up. Every action stream the Generator sees is normalized."""
    path = cfg.data.get("action_stats")
    path = Path(path) if path else out_dir / "action_normalizer.json"
    need_view = canonical_task_mode(cfg.model.get("task_mode", "action_causal")) == "action_full"
    norms = None
    if rank == 0:
        resume_path = out_dir / "last.pt"
        if resume_path.exists():
            # Resume with the same transformation used to train the saved weights.
            ck = torch.load(resume_path, map_location="cpu", weights_only=False, mmap=True)
            norms = ActionNormalizers.from_state_dict(ck["normalizer"])
            if need_view and norms.view is None:
                raise ValueError("Cannot resume action_full checkpoint without view normalization; use a new run_name")
            log(f"[data] restored action normalizers from {resume_path}")
        elif (cfg.get("training", {}) or {}).get("init_from_checkpoint"):
            if cfg.data.get("action_stats"):
                raise ValueError("Weights-only initialization restores checkpoint normalizers; do not set action_stats")
            ck = torch.load(cfg.training.init_from_checkpoint, map_location="cpu", weights_only=False, mmap=True)
            norms = ActionNormalizers.from_state_dict(ck["normalizer"])
            if need_view and norms.view is None:
                raise ValueError("action_full initialization requires checkpoint view normalization")
            norms.save(out_dir / "action_normalizer.json")
            log(f"[data] restored initialization normalizers from {cfg.training.init_from_checkpoint}")
        elif path.exists():
            norms = ActionNormalizers.load(path)
            if need_view and norms.view is None:
                norms.view = _fit_one(train_ds, cfg, VIEW_ACTION_STREAMS, log, "view")
                # Preserve explicitly supplied legacy statistics; save the combined table with this run.
                combined_path = out_dir / "action_normalizer.json"
                norms.save(combined_path)
                log(f"[data] saved combined sensor/view normalizers to {combined_path}")
        else:
            sensor = _fit_one(train_ds, cfg, ACTION_STREAMS, log, "sensor")
            view = _fit_one(train_ds, cfg, VIEW_ACTION_STREAMS, log, "view") if need_view else None
            norms = ActionNormalizers(sensor, view)
            path.parent.mkdir(parents=True, exist_ok=True)
            norms.save(path)
            log(f"[data] saved action normalizers -> {path}")
    if world > 1:
        dev = device if (device is not None and device.type == "cuda") else torch.device("cpu")
        buf = torch.zeros(2, 3, 9, dtype=torch.float32, device=dev)
        if rank == 0:
            _norm_to_buf(norms.sensor, buf[0])
            _norm_to_buf(norms.view, buf[1])
        dist.broadcast(buf, src=0)
        meta = {"broadcast_from_rank0": str(path)}
        norms = ActionNormalizers(_buf_to_norm(buf[0], meta), _buf_to_norm(buf[1], meta))
    if need_view and norms.view is None:
        raise RuntimeError("action_full needs a view-action normalizer")
    return norms


def move_batch(batch, device):
    lat = {s: t.to(device, non_blocking=True) for s, t in batch["latents"].items()}
    act = {s: batch["actions"][s].to(device, non_blocking=True) for s in batch["actions"]}
    val = {s: batch["action_valid"][s].to(device, non_blocking=True) for s in batch["action_valid"]}
    return lat, act, val


def all_reduce_counts(counts: dict[str, int], world: int, device) -> dict[str, int]:
    """Global #valid-target samples per stream, reduced in the fixed STREAMS order on every rank."""
    vec = torch.tensor([float(counts.get(s, 0)) for s in STREAMS], dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(vec, op=dist.ReduceOp.SUM)
    return {s: int(vec[i].item()) for i, s in enumerate(STREAMS)}


def presence_logs(groups: list[dict]) -> dict[str, float]:
    present = torch.cat([g["stream_present"] for g in groups])
    dropped = torch.cat([g["stream_dropped"] for g in groups])
    wl = STREAM_ID["wrist_left"]
    n = present.shape[0]
    n_present = int(present[:, wl].sum())
    n_dropped = int(dropped[:, wl].sum())
    return {
        "present_rate/wrist": n_present / max(n, 1),
        "active_rate/wrist": (n_present - n_dropped) / max(n, 1),
        "drop_rate_given_present/wrist": n_dropped / max(n_present, 1),
        "groups": float(len(groups)),
    }


# ------------------------------------------------------------------ steps
def group_passes(core, group, cfg, device, sigma_generator=None):
    """Stage metadata only: forward AND backward must occur inside the caller's no_sync context."""
    if core.cfg.prediction_strategy == "causal_teacher_forcing":
        sigma = sample_sigma(len(group["latents"]["video"]), device,
                             str(cfg.training.get("sigma_distribution", "waver")),
                             float(cfg.training.get("sigma_shift", 3.0)), generator=sigma_generator)
        for target_index in (2, 3):
            yield target_index, 0.5, sigma
    else:
        yield None, 1.0, None


def group_forward(adapter, group, normalizer, cfg, device, generator=None, sigma_generator=None,
                  target_index=None, sigma_b=None):
    """One active-pattern group: noising -> adapter forward -> per-stream loss sums/counts."""
    lat, act, valid = move_batch(group, device)
    B = lat["video"].shape[0]
    act_n = ActionNormalizers.wrap(normalizer).normalize_dict(act)  # sensor AND view actions
    sigma = sigma_b if sigma_b is not None else sample_sigma(
        B, device, str(cfg.training.get("sigma_distribution", "waver")),
        float(cfg.training.get("sigma_shift", 3.0)), generator=sigma_generator)
    core = adapter.module if isinstance(adapter, DDP) else adapter
    if core.cfg.prediction_strategy == "causal_teacher_forcing" and target_index is None:
        raise ValueError("Use group_passes and an explicit target_index for causal training")
    layout = core.layout_for(B, device, active=group["active"], target_index=target_index)
    if layout.cfg.task_mode == "action_full" and not all(bool(v.all()) for v in valid.values()):
        raise ValueError("action_full requires all action transitions to be valid")
    states, targets, masks = build_training_states(lat, act_n, sigma, layout, generator=generator)
    amp = str(cfg.training.get("amp", "bf16"))
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(amp == "bf16" and device.type == "cuda")):
        pred = adapter(states, sigma, layout)
    av = action_valid_mask(valid, act_n)
    sums, counts, extra = stream_loss_sums(pred, targets, masks, action_valid=av)
    return sums, counts, extra


def group_counts(core, group, device, task_mode: str) -> dict[str, int]:
    """Valid-target sample counts per stream for a group, computable before any forward."""
    B = group["latents"]["video"].shape[0]
    lay = core.layout_for(1, device, active=group["active"])
    masks = {}
    for s in group["active"]:
        ref = group["latents"][s] if s in IMAGE_STREAMS else group["actions"][s]
        masks[s] = lay.target_mask(s).expand(B, *ref.shape[1:])
    if canonical_task_mode(task_mode) in ("forward_dynamics", "action_full"):
        masks = {s: m for s, m in masks.items() if s not in ALL_ACTION_STREAMS}
    av = {s: group["action_valid"][s][:, :, None].expand_as(masks[s]) for s in ACTION_STREAMS if s in masks}
    return target_counts(masks, av)


def train_step(model, groups, normalizer, cfg, device, world, params, grad_clip):
    """Forward/backward over all active-pattern groups of one batch with global-count normalization
    (spec-MM §6). Under DDP the first N-1 groups run inside no_sync(); the last one syncs, so every
    rank does one synchronized backward per step regardless of its group count."""
    core = model.module if isinstance(model, DDP) else model
    task_mode = str(cfg.model.get("task_mode", "action_causal"))
    local_counts: dict[str, int] = {}
    for g in groups:
        for s, c in group_counts(core, g, device, task_mode).items():
            local_counts[s] = local_counts.get(s, 0) + c
    global_counts = all_reduce_counts(local_counts, world, device)  # same collective order on all ranks
    weights = dict(cfg.training.get("loss_weights", {}) or {})
    logs_acc: dict[str, float] = {}
    sums_acc: dict[str, float] = {}
    for gi, g in enumerate(groups):
        for target_index, stage_weight, sigma in group_passes(core, g, cfg, device):
            last = gi == len(groups) - 1 and target_index in (None, 3)
            ctx = model.no_sync() if (isinstance(model, DDP) and not last) else contextlib.nullcontext()
            with ctx:
                sums, counts, extra = group_forward(model, g, normalizer, cfg, device,
                                                    target_index=target_index, sigma_b=sigma)
                sums = {s: v * stage_weight for s, v in sums.items()}
                total, logs = combine_sums(sums, counts, global_counts, weights, world)
                if total is None:
                    total = sum(p.sum() * 0.0 for p in params)
                total.backward()
            for k, v in logs.items():
                if k.startswith("n/"):
                    logs_acc[k] = logs_acc.get(k, 0.0) + v * stage_weight
            for k, v in sums.items():
                sums_acc[k] = sums_acc.get(k, 0.0) + float(v.detach())
            logs_acc.update(extra)
    gn = torch.nn.utils.clip_grad_norm_(params, grad_clip) if grad_clip > 0 else torch.tensor(0.0)
    # per-stream local mean over valid samples (logging only; the optimized quantity is the global mean)
    for s in STREAMS:
        n = logs_acc.get(f"n/{s}", 0.0)
        if n > 0:
            logs_acc[f"loss/{s}"] = sums_acc.get(s, 0.0) / n
    img = [logs_acc[f"loss/{s}"] for s in IMAGE_STREAMS if f"loss/{s}" in logs_acc]
    logs_acc["loss/image_mean"] = sum(img) / len(img) if img else float("nan")
    logs_acc["loss/total"] = sum(float(weights.get(s, 1.0)) * logs_acc[f"loss/{s}"] for s in STREAMS if f"loss/{s}" in logs_acc)
    logs_acc.update(presence_logs(groups))
    return float(gn), logs_acc


@torch.no_grad()
def validate(adapter, loader, normalizer, cfg, device, world, seed: int = 1234, arm: str = "") -> dict[str, float]:
    """Random dropout OFF. Losses are per-stream means over the globally valid samples; when the
    loader mixes patterns the log keys are additionally split by pattern ('wrist' / 'nowrist')."""
    adapter.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    sgen = torch.Generator().manual_seed(seed)
    sums: dict[str, float] = {}
    counts: dict[str, float] = {}
    for groups in loader:
        for g in groups:
            pat = "wrist" if g["latents"].get("wrist_left") is not None else "nowrist"
            core = adapter.module if isinstance(adapter, DDP) else adapter
            for target_index, stage_weight, sigma in group_passes(core, g, cfg, device, sgen):
                gs, gc, _ = group_forward(adapter, g, normalizer, cfg, device, generator=gen,
                                          sigma_generator=sgen, target_index=target_index, sigma_b=sigma)
                for s in gs:
                    for key in (f"{s}", f"{pat}/{s}"):
                        sums[key] = sums.get(key, 0.0) + float(gs[s]) * stage_weight
                        counts[key] = counts.get(key, 0.0) + gc[s] * stage_weight
    adapter.train()
    # All ranks must issue the same collectives even if their validation patterns differ.
    keys = sorted(f"{prefix}{s}" for prefix in ("", "wrist/", "nowrist/") for s in STREAMS)
    out = {}
    for k in keys:
        s_ = all_reduce_mean(sums.get(k, 0.0), world, device) * world
        c_ = all_reduce_mean(counts.get(k, 0.0), world, device) * world
        if c_ > 0:
            out[f"loss/{k}"] = s_ / c_
            out[f"n/{k}"] = c_
    img = [out[f"loss/{s}"] for s in IMAGE_STREAMS if f"loss/{s}" in out]
    out["loss/image_mean"] = sum(img) / len(img) if img else float("nan")
    weights = dict(cfg.training.get("loss_weights", {}) or {})
    out["loss/total"] = sum(float(weights.get(s, 1.0)) * out[f"loss/{s}"] for s in STREAMS if f"loss/{s}" in out)
    if arm:
        out = {(k.replace("loss/", f"loss/{arm}/") if k.startswith("loss/") else k.replace("n/", f"n/{arm}/")): v for k, v in out.items()}
    return out


def save_checkpoint(path: Path, adapter, optimizer, normalizer, cfg, epoch, step, best,
                    sampling_state=None, causal_state=None):
    core = adapter.module if isinstance(adapter, DDP) else adapter
    from omegaconf import OmegaConf

    torch.save({
        "trainable_state": core.trainable_state_dict(),
        "optimizer": optimizer.state_dict(),
        "normalizer": normalizer.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "epoch": epoch, "step": step, "best_val": best,
        "sampling_state": sampling_state,
        "causal_state": causal_state,
    }, path)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    validate_prediction_config(cfg)

    rank, world, local = init_distributed()
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    out_dir = Path(cfg.out_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    last = out_dir / "last.pt"
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=False, mmap=True)
        if prediction_contract(saved["config"]) != prediction_contract(cfg):
            raise ValueError("Cannot change prediction contract on resume; use a new run_name and init_from_checkpoint")
        del saved
    init_checkpoint = initialization_checkpoint(cfg) if not last.exists() else None
    log_f = open(out_dir / "train.log", "a") if rank == 0 else None

    def log(msg: str):
        if rank == 0:
            print(msg, flush=True)
            log_f.write(msg + "\n")
            log_f.flush()

    log(f"[init] rank={rank}/{world} device={device} run={cfg.run_name}")
    train_ds, val_ds = build_datasets(cfg, rank, log)
    normalizer = get_normalizer(cfg, train_ds, out_dir, rank, world, log, device)
    bs = int(cfg.training.batch_size)
    nw = int(cfg.data.get("num_workers", 4))
    val_sampler = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False) if world > 1 else None
    wd = cfg.training.get("wrist_dropout", {}) or {}
    p_drop = float(wd.get("probability", 0.0))
    if p_drop > 0 and str(wd.get("mode", "paired")) != "paired":
        raise ValueError("only wrist_dropout.mode=paired is implemented")
    train_collate = functools.partial(collate_by_active, dropout=WristDropout(p_drop))
    eval_collate = functools.partial(collate_by_active, dropout=WristDropout(0.0))
    hide_collate = functools.partial(collate_by_active, dropout=WristDropout(0.0, hide_all=True))
    log(f"[data] wrist_dropout.probability={p_drop} (train only); validation arms: natural" +
        (" + hidden-wrist" if bool(cfg.training.get("val_hide_wrist", False)) else ""))
    train_loader, train_sampler = build_train_loader(cfg, train_ds, rank, world, train_collate)
    proportional = isinstance(train_sampler, ProportionalWristBatchSampler)
    causal = prediction_contract(cfg)[3] == "causal_teacher_forcing"
    data_contract = None
    if causal:
        if not len(train_loader):
            raise ValueError("Causal training loader has no complete batches")
        with_wrist, without_wrist, fingerprint = wrist_pools(train_ds)
        data_contract = dict(fingerprint=fingerprint, windows=[len(with_wrist), len(without_wrist)],
                             batch_size=bs, world_size=world, steps_per_epoch=len(train_loader), seed=seed,
                             sampling_mode="dataset_proportional" if proportional else "uniform",
                             validation_fingerprint=wrist_pools(val_ds)[2] if hasattr(val_ds, "episode_info") else None)
        if world > 1:
            contracts = [None] * world
            dist.all_gather_object(contracts, data_contract)
            if any(c != contracts[0] for c in contracts):
                raise ValueError("Causal training data/configuration differs across ranks")
    if proportional:
        log(f"[data] dataset_proportional: valid W={train_sampler.n_w} N={train_sampler.n_n}, "
            f"local batch={bs}, reporting steps/epoch={train_sampler.steps_per_epoch}")
        if world > 1:
            signatures = [None] * world
            dist.all_gather_object(signatures, train_sampler.signature)
            if any(s != signatures[0] for s in signatures):
                raise ValueError("Training dataset/sampling configuration differs across ranks")
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, sampler=val_sampler, num_workers=min(nw, 2),
                            pin_memory=True, drop_last=False, collate_fn=eval_collate,
                            generator=torch.Generator().manual_seed(seed + rank + 10000) if proportional or causal else None)
    val_hide_loader = None
    if bool(cfg.training.get("val_hide_wrist", False)):
        val_hide_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, sampler=val_sampler, num_workers=min(nw, 2),
                                     pin_memory=True, drop_last=False, collate_fn=hide_collate,
                                     generator=torch.Generator().manual_seed(seed + rank + 20000) if proportional or causal else None)

    adapter, info = build_model(cfg, device, log)
    if init_checkpoint is not None:
        adapter.load_trainable_state_dict(init_checkpoint["trainable_state"])
        log(f"[init] weights only from {cfg.training.init_from_checkpoint}; fresh optimizer and step=0")
        del init_checkpoint
    if device.type == "cuda":
        # loading materializes weights transiently; reset so the logged 'mem' is the TRAINING peak
        log(f"[build] peak mem during build {torch.cuda.max_memory_allocated(device) / 2**30:.1f}G, "
            f"resident after build {torch.cuda.memory_allocated(device) / 2**30:.1f}G")
        torch.cuda.reset_peak_memory_stats(device)
    if rank == 0:
        (out_dir / "trainable_params.json").write_text(json.dumps(info["trainable"], indent=2))
        if info["load_report"]:
            (out_dir / "load_report.json").write_text(json.dumps(info["load_report"], indent=2))

    params = [p for p in adapter.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(cfg.training.lr), weight_decay=float(cfg.training.get("weight_decay", 0.0)),
                                  betas=tuple(cfg.training.get("betas", (0.9, 0.95))))
    start_epoch, step, best = 0, 0, math.inf
    resume_checkpoint = None
    last = out_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        resume_checkpoint = ck
        if causal:
            state = ck.get("causal_state")
            if state is None or state["data_contract"] != data_contract or len(state["rng_by_rank"]) != world:
                raise ValueError("Causal resume requires identical data, batch, world size and epoch length")
        adapter.load_trainable_state_dict(ck["trainable_state"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch, step, best = int(ck["epoch"]) + 1, int(ck["step"]), float(ck["best_val"])
        log(f"[resume] from {last}: epoch={start_epoch} step={step} best={best:.5f}")

    model = adapter
    if world > 1:
        find_unused = bool(cfg.training.get("find_unused_parameters", False)) or (
            str(cfg.model.get("task_mode", "action_causal")) in ("forward_dynamics", "action_full"))
        model = DDP(adapter, device_ids=[local] if device.type == "cuda" else None, broadcast_buffers=False,
                    find_unused_parameters=find_unused)
    if resume_checkpoint is not None:
        restore_sampling_checkpoint(resume_checkpoint, train_sampler, rank, world, device)
        if proportional:
            start_epoch = train_sampler.completed_steps // train_sampler.steps_per_epoch
        elif causal:
            start_epoch = step // len(train_loader)
        if causal:
            restore_rng_state(resume_checkpoint["causal_state"]["rng_by_rank"][rank], device)
        del resume_checkpoint, ck

    if rank == 0:
        from omegaconf import OmegaConf
        (out_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
        if causal:
            (out_dir / "data_manifest.json").write_text(json.dumps(data_contract, indent=2))

    epochs = int(cfg.training.epochs)
    max_steps = cfg.training.get("max_steps")
    warmup = int(cfg.training.get("warmup_steps", 0))
    base_lr = float(cfg.training.lr)
    grad_clip = float(cfg.training.get("grad_clip", 1.0))
    log_every = int(cfg.training.get("log_every", 50))
    metrics_f = open(out_dir / "metrics.jsonl", "a") if rank == 0 else None
    t0 = time.time()
    step0 = step  # steps done before this process started (resume): s/it is measured from here
    done = False
    model.train()
    for epoch in range(start_epoch, epochs):
        if max_steps is not None and step >= int(max_steps):
            break
        model.train()
        if train_sampler is not None and not proportional:
            train_sampler.set_epoch(epoch)
        skip_batches = step % len(train_loader) if causal and not proportional and epoch == start_epoch else 0
        for batch_index, groups in enumerate(train_loader):
            if batch_index < skip_batches:
                continue
            if warmup > 0 and step < warmup:
                for g in optimizer.param_groups:
                    g["lr"] = base_lr * (step + 1) / warmup
            optimizer.zero_grad(set_to_none=True)
            gn, logs = train_step(model, groups, normalizer, cfg, device, world, params, grad_clip)
            if step == step0:  # every trainable parameter must receive a gradient, or DDP fails at step 2
                unused = [n for n, p in adapter.named_parameters() if p.requires_grad and p.grad is None]
                if unused:
                    raise RuntimeError(f"{len(unused)} trainable parameters got no gradient (would break DDP): {unused[:8]}")
            optimizer.step()
            step += 1
            if proportional:
                train_sampler.commit()
                if train_sampler.completed_steps != step:
                    raise RuntimeError("Optimizer and committed sampling steps diverged")
                logs.update(train_sampler.metrics())
            if step % log_every == 0 or step == step0 + 1:
                logs.update({"step": step, "epoch": epoch, "grad_norm": float(gn), "lr": optimizer.param_groups[0]["lr"],
                             "elapsed_s": time.time() - t0,
                             "peak_mem_GiB": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0})
                wr = f" wr {logs['active_rate/wrist']:.2f}" if logs.get("present_rate/wrist", 1.0) < 1.0 or p_drop > 0 else ""
                action_log = "action=condition " if cfg.model.task_mode == "action_full" else (
                    f"aL {logs.get('loss/action_left', float('nan')):.5f} aR {logs.get('loss/action_right', float('nan')):.5f} ")
                log(f"[train] step {step} ep {epoch} loss {logs['loss/total']:.5f} img {logs['loss/image_mean']:.5f} "
                    f"{action_log}"
                    f"gn {float(gn):.3f} {logs['elapsed_s'] / max(step - step0, 1):.2f}s/it mem {logs['peak_mem_GiB']:.1f}G{wr}")
                if metrics_f:
                    metrics_f.write(json.dumps({"split": "train", **logs}) + "\n")
                    metrics_f.flush()
            if max_steps is not None and step >= int(max_steps):
                done = True
                break
        vlogs = validate(model, val_loader, normalizer, cfg, device, world)
        if val_hide_loader is not None:
            vlogs.update(validate(model, val_hide_loader, normalizer, cfg, device, world, arm="hidden"))
        v = vlogs.get("loss/total", math.inf)
        log(f"[val] epoch {epoch} step {step} " + " ".join(f"{k[5:]}={x:.5f}" for k, x in vlogs.items() if k.startswith("loss/")))
        sampling_state = sampling_checkpoint_state(train_sampler, world, device)
        causal_state = causal_checkpoint_state(data_contract, world, device)
        if rank == 0:
            metrics_f.write(json.dumps({"split": "val", "epoch": epoch, "step": step, **vlogs}) + "\n")
            metrics_f.flush()
            is_best = v < best
            best = min(best, v)
            save_checkpoint(last, adapter, optimizer, normalizer, cfg, epoch, step, best, sampling_state, causal_state)
            if is_best:
                save_checkpoint(out_dir / "best.pt", adapter, optimizer, normalizer, cfg, epoch, step, best, sampling_state, causal_state)
            if int(cfg.training.get("ckpt_every_epochs", 0)) and (epoch + 1) % int(cfg.training.ckpt_every_epochs) == 0:
                save_checkpoint(out_dir / f"epoch_{epoch:03d}.pt", adapter, optimizer, normalizer, cfg, epoch, step, best, sampling_state, causal_state)
        if world > 1:
            dist.barrier()
        if done:
            break
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
    log(f"[done] steps={step} best_val={best:.5f} elapsed={time.time() - t0:.0f}s peak_mem_GiB={peak:.1f}")
    if metrics_f:
        metrics_f.close()
    if log_f:
        log_f.close()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
