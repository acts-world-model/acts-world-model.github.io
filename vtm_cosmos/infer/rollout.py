"""Single-window prediction and the cache-consistent reference rollout (spec §10).

    python -m vtm_cosmos.infer.rollout --ckpt runs/<run>/best.pt --split val --num-windows 4 \
        [--windows 0 100 ...] [--rollout-windows 3] [--out-dir ...]

Teacher-forced window evaluation and free-running rollout are written to separate files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from vtm_cosmos.data.actions import ActionNormalizer, ActionNormalizers
from vtm_cosmos.data.dataset import VtmWindowDataset
from vtm_cosmos.data.splits import load_split
from vtm_cosmos.infer.decode import WanDecoder
from vtm_cosmos.infer.metrics import LPIPS, action_metrics, image_metrics, to_uint8
from vtm_cosmos.infer.sampler import WindowSampler
from vtm_cosmos.streams import (
    ACTION_STREAMS, VIEW_ACTION_STREAMS, streams_from_pattern,
    WRIST_STREAMS,
    FRAMES_PER_LATENT,
    FUTURE_ACTIONS,
    HISTORY_ACTIONS,
    HISTORY_LATENTS,
    IMAGE_STREAMS,
    N_ACTIONS,
    WINDOW_FRAMES,
)
from vtm_cosmos.train.model_factory import build_model
from vtm_cosmos.tasks import prediction_contract

FIRST_FUTURE_FRAME = FRAMES_PER_LATENT * HISTORY_LATENTS  # 8


def history_latent_error(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Exactness diagnostic for the two clean latent history blocks.

    A sampled window must retain this history verbatim.  In a chained rollout,
    however, later windows deliberately use the prior model output as history,
    so this value measures accumulated drift against the recording.
    """
    diff = pred[:, :HISTORY_LATENTS].float() - target[:, :HISTORY_LATENTS].float()
    return {
        "history_latent_max_abs_error": float(diff.abs().max()),
        "history_latent_mse": float(diff.square().mean()),
    }


def load_run(ckpt_path: str | Path, device, overrides: dict | None = None):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ck["config"])
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    if prediction_contract(cfg) != prediction_contract(ck["config"]):
        raise ValueError("Inference cannot change the checkpoint prediction contract; train a new run")
    adapter, _ = build_model(cfg, device)
    adapter.load_trainable_state_dict(ck["trainable_state"])
    adapter.eval()
    normalizer = ActionNormalizers.from_state_dict(ck["normalizer"])
    return cfg, adapter, normalizer


def present_actions(item: dict) -> list[str]:
    """Action streams the run models (dataset items always carry every action file; a stream-subset
    run marks the unmodeled ones absent in ``stream_present``)."""
    present = streams_from_pattern(tuple(bool(x) for x in item["stream_present"]))
    return [s for s in item["actions"] if s in present]


def conditions_from_item(item: dict, device, hide_wrist: bool = False) -> dict[str, torch.Tensor]:
    """History-only conditions for the streams the recording provides (absent wrists have no key;
    ``hide_wrist`` drops present wrists to emulate a wrist-less request). Future latents /
    action_causal future actions are discarded by the sampler; action_full keeps all actions."""
    cond = {}
    for s in IMAGE_STREAMS:
        if s not in item["latents"] or (hide_wrist and s in WRIST_STREAMS):
            continue
        z = item["latents"][s].clone()[None].to(device)
        z[:, :, HISTORY_LATENTS:] = 0.0
        cond[s] = z
    for s in present_actions(item):
        a = item["actions"][s].clone()[None].to(device)
        cond[s] = a  # the sampler only reads rows the layout marks as conditions
    return cond


def save_strip(path: Path, gt: torch.Tensor, pred: torch.Tensor, first_future: int = FIRST_FUTURE_FRAME) -> None:
    """Two rows (GT / PRED) x 16 frames; red bar marks the first predicted frame."""
    import imageio.v2 as imageio

    g, p = to_uint8(gt), to_uint8(pred)
    T, H, W, _ = g.shape
    canvas = np.zeros((2 * H + 4, T * (W + 2), 3), dtype=np.uint8)
    for t in range(T):
        x = t * (W + 2)
        canvas[:H, x : x + W] = g[t]
        canvas[H + 4 : 2 * H + 4, x : x + W] = p[t]
    x0 = first_future * (W + 2)
    canvas[:, max(x0 - 2, 0) : x0] = (255, 0, 0)
    imageio.imwrite(path, canvas)


def copy_last_prediction(item: dict, device, hide_wrist: bool = False) -> dict[str, torch.Tensor]:
    """Trivial baseline: repeat the last history latent for both future latents and the last
    history action a7 for a8..a15. Any model has to beat this on the same windows."""
    out = {}
    for s in IMAGE_STREAMS:
        if s not in item["latents"] or (hide_wrist and s in WRIST_STREAMS):
            continue
        z = item["latents"][s].clone()[None].to(device)
        z[:, :, HISTORY_LATENTS:] = z[:, :, HISTORY_LATENTS - 1 : HISTORY_LATENTS]
        out[s] = z
    for s in present_actions(item):
        a = item["actions"][s].clone()[None].to(device)
        if not set(item["actions"]).intersection(VIEW_ACTION_STREAMS):
            a[:, HISTORY_ACTIONS:] = a[:, HISTORY_ACTIONS - 1 : HISTORY_ACTIONS]
        out[s] = a
    return out


@torch.no_grad()
def predict_window(sampler: WindowSampler | None, decoder: WanDecoder | None, ds: VtmWindowDataset, idx: int, device, seed: int,
                   lpips_fn=None, out_dir: Path | None = None, baseline: str | None = None, hide_wrist: bool = False) -> dict:
    item = ds[idx]
    ep, t0 = ds.window_meta(idx)
    if baseline == "copy_last":
        out = copy_last_prediction(item, device, hide_wrist=hide_wrist)
    else:
        if sampler.adapter.cfg.task_mode == "action_full" and not all(bool(v.all()) for v in item["action_valid"].values()):
            raise ValueError("action_full requires all action transitions to be valid")
        cond = conditions_from_item(item, device, hide_wrist=hide_wrist)
        out = sampler.sample(cond, seed=seed)
    present = [s for s in IMAGE_STREAMS if s in out]
    rec = {"episode": ep, "t_start": t0, "seed": seed, "baseline": baseline, "hide_wrist": hide_wrist,
           "has_wrist": "wrist_left" in item["latents"], "streams": present, "image": {}, "action": {}}
    for s in present:
        gt = item["latents"][s]
        pred = out[s][0].cpu()
        rec["image"][s] = {
            **history_latent_error(pred, gt),
            "latent_future_mse": float((pred[:, HISTORY_LATENTS:] - gt[:, HISTORY_LATENTS:]).square().mean()),
        }
        if decoder is not None:
            px_gt = decoder.decode_window(s, ep, t0, gt)
            px_pred = decoder.decode_window(s, ep, t0, pred)
            rec["image"][s].update(image_metrics(px_pred, px_gt, FIRST_FUTURE_FRAME, lpips_fn))
            if out_dir is not None:
                save_strip(out_dir / f"{Path(ep).stem}_t{t0:05d}_{s}.png", px_gt, px_pred)
    for s in (() if set(item["actions"]).intersection(VIEW_ACTION_STREAMS) else ACTION_STREAMS):
        gt = item["actions"][s][HISTORY_ACTIONS:]
        pred = out[s][0, HISTORY_ACTIONS:].cpu()
        valid = item["action_valid"][s][HISTORY_ACTIONS:]
        rec["action"][s] = action_metrics(pred, gt, valid)
    if out_dir is not None:
        torch.save({"pred": {s: out[s][0].cpu() for s in present + present_actions(item)}, "item": item},
                   out_dir / f"{Path(ep).stem}_t{t0:05d}_pred.pt")
    return rec


@torch.no_grad()
def reference_rollout(sampler: WindowSampler, decoder: WanDecoder, ds: VtmWindowDataset, idx: int, device, seed: int,
                      num_windows: int, lpips_fn=None, out_dir: Path | None = None, hide_wrist: bool = False) -> dict:
    """Free-running rollout consistent with the per-window cache (spec §10.2).

    Each step: sample 8 future frames -> decode -> the 8 decoded frames become the next
    history (duplicate first frame -> 9 frames -> 3 latents: anchor + 2 history) -> next window
    time metadata restarts at 0 -> sample again. Actions: predicted a_8..a_15 -> next history
    a_1..a_7 = predicted rows 1..7 (row 0 is the incoming transition into the new frame 0).
    Ground truth for scoring is the dataset window at t0 + 8*k (never fed back).

    ``action_full`` predicts no actions, so the commanded action sequence is supplied per window
    from the recording (the actions the operator did execute); images stay free-running. The
    rollout stops early if a window is past the end of the episode or carries an invalid
    transition, since those actions cannot be conditioned on."""
    action_full = sampler.adapter.cfg.task_mode == "action_full"
    item = ds[idx]
    ep, t0 = ds.window_meta(idx)
    cond = conditions_from_item(item, device, hide_wrist=hide_wrist)
    present = [s for s in IMAGE_STREAMS if s in cond]
    anchors = {s: decoder.anchor(s, ep, t0) for s in present}
    records = []
    for k in range(num_windows):
        t_k = t0 + FUTURE_ACTIONS * k
        out = sampler.sample(cond, seed=seed + k)
        rec = {
            "window": k,
            "t_start": t_k,
            "history_source": "recording" if k == 0 else "previous_model_prediction",
            "image": {},
            "action": {},
        }
        gt_item = ds.item_at(ep, t_k)  # scoring only; never fed back
        next_cond = {}
        for s in present:
            z_full = out[s][0].cpu()  # history (given) + predicted future
            px = decoder.decode_with_anchor(s, anchors[s], z_full)  # (3,16,H,W)
            px_future = px[:, FIRST_FUTURE_FRAME:]  # 8 predicted frames
            if gt_item is not None:
                px_gt = decoder.decode_window(s, ep, t_k, gt_item["latents"][s])
                rec["image"][s] = {
                    **history_latent_error(z_full, gt_item["latents"][s]),
                    **image_metrics(px, px_gt, FIRST_FUTURE_FRAME, lpips_fn),
                }
                if out_dir is not None:
                    save_strip(out_dir / f"{Path(ep).stem}_t{t0:05d}_roll{k}_{s}.png", px_gt, px)
            # re-encode the 8 predicted frames as the next history (duplicate first frame -> 9)
            nine = torch.cat([px_future[:, :1], px_future], dim=1)
            z9 = decoder.encode_frames(s, nine)  # (48,3,g,g): anchor + 2 history
            anchors[s] = z9[:, :1]
            z_next = torch.zeros_like(z_full)
            z_next[:, :HISTORY_LATENTS] = z9[:, 1:3]
            next_cond[s] = z_next[None].to(device)
        if action_full:
            nxt = ds.item_at(ep, t0 + FUTURE_ACTIONS * (k + 1))
            stop = None
            if k + 1 < num_windows:
                if nxt is None:
                    stop = "no window at the next start (end of episode)"
                elif not all(bool(v.all()) for v in nxt["action_valid"].values()):
                    stop = "next window has an invalid action transition"
            records.append(rec)
            if stop is not None:
                rec["stopped"] = stop
                break
            if nxt is not None:
                for s in present_actions(nxt):
                    next_cond[s] = nxt["actions"][s].clone()[None].to(device)
            cond = next_cond
            continue
        for s in ACTION_STREAMS:
            pred_future = out[s][0, HISTORY_ACTIONS:].cpu()  # a_8..a_15
            if gt_item is not None:
                rec["action"][s] = action_metrics(pred_future, gt_item["actions"][s][HISTORY_ACTIONS:], gt_item["action_valid"][s][HISTORY_ACTIONS:])
            a_next = torch.zeros(N_ACTIONS, 9)
            a_next[:HISTORY_ACTIONS] = pred_future[1:1 + HISTORY_ACTIONS]
            next_cond[s] = a_next[None].to(device)
        records.append(rec)
        cond = next_cond
    return {"episode": ep, "t_start": t0, "streams": present, "hide_wrist": hide_wrist, "windows": records}


def summarize(results: list[dict]) -> dict:
    """Mean over windows of every per-stream metric (nan-safe)."""
    import math

    acc: dict[str, list[float]] = {}
    for r in results:
        pat = "wrist" if r.get("has_wrist", True) else "nowrist"
        for kind in ("image", "action"):
            for s, m in r[kind].items():
                for k, v in m.items():
                    if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                        acc.setdefault(f"{s}/{k}", []).append(float(v))
                        acc.setdefault(f"{pat}/{s}/{k}", []).append(float(v))
    out = {k: sum(v) / len(v) for k, v in acc.items() if v}
    out["num_windows"] = len(results)
    out["num_windows_with_wrist"] = sum(1 for r in results if r.get("has_wrist", True))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--num-windows", type=int, default=4)
    ap.add_argument("--windows", type=int, nargs="*", default=None, help="explicit dataset indices")
    ap.add_argument("--window-stride", type=int, default=200)
    ap.add_argument("--rollout-windows", type=int, default=0, help=">0 runs the reference rollout with this many windows")
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--solver", default="unipc", choices=["unipc", "euler"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-decode", action="store_true")
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--baseline", default=None, choices=[None, "copy_last"], help="score a trivial baseline instead of the model")
    ap.add_argument("--hide-wrist", action="store_true", help="drop present wrist history (artificial missing-wrist arm)")
    ap.add_argument("--optional-wrist", action="store_true", help="allow episodes without wrist latents (mixed sets)")
    ap.add_argument("--override", nargs="*", default=None, metavar="KEY=VALUE",
                    help="dotlist merged into the checkpoint config, e.g. data.split_file=... to score a model "
                         "on another run's windows (its own normalizer is kept)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overrides = OmegaConf.to_container(OmegaConf.from_dotlist(args.override)) if args.override else None
    cfg, adapter, normalizer = load_run(args.ckpt, device, overrides)
    if overrides:
        print(f"[override] {json.dumps(overrides)}", flush=True)
    split = load_split(cfg.data.split_file)
    optional = tuple(cfg.data.get("optional_streams", []) or [])
    if args.optional_wrist:
        optional = ("wrist_left", "wrist_right")
    ds = VtmWindowDataset(cfg.data.latent_root, cfg.data.action_root, split[args.split], action_file=cfg.data.action_file,
                          window_stride=args.window_stride, optional_streams=optional,
                          include_view_actions=cfg.model.task_mode == "action_full",
                          view_action_file=cfg.data.get("view_action_file", "view_actions_fps15.pt"),
                          require_valid_actions=cfg.model.task_mode == "action_full",
                          active_streams=tuple(cfg.model.active_streams) if cfg.model.get("active_streams") else None)
    tag = f"eval_{args.split}_{Path(args.ckpt).stem}" + (f"_{args.baseline}" if args.baseline else "") + ("_hidewrist" if args.hide_wrist else "")
    out_dir = Path(args.out_dir or Path(args.ckpt).parent / tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    sampler = None if args.baseline else WindowSampler(adapter, normalizer, num_steps=args.steps, solver=args.solver)
    decoder = None if args.no_decode else WanDecoder(cfg.data.latent_root, device)
    lpips_fn = None if (args.no_decode or args.no_lpips) else LPIPS(device)
    indices = args.windows if args.windows else list(range(min(args.num_windows, len(ds))))
    results = []
    for i in indices:
        rec = predict_window(sampler, decoder, ds, i, device, args.seed, lpips_fn, out_dir, baseline=args.baseline, hide_wrist=args.hide_wrist)
        results.append(rec)
        print(json.dumps(rec), flush=True)
    summary = summarize(results)
    print("[summary] " + json.dumps(summary), flush=True)
    (out_dir / "window_eval.json").write_text(json.dumps({"config": OmegaConf.to_container(cfg), "summary": summary, "results": results}, indent=2))
    if args.rollout_windows > 0 and decoder is not None and sampler is not None:
        rolls = [reference_rollout(sampler, decoder, ds, i, device, args.seed, args.rollout_windows, lpips_fn, out_dir, hide_wrist=args.hide_wrist) for i in indices]
        (out_dir / "reference_rollout.json").write_text(json.dumps(rolls, indent=2))
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
