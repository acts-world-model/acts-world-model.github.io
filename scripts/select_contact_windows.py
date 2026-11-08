"""Stratified contact-window selection for `vtm_cosmos.infer.rollout`.

Every window of a split is classed from the recorded GelSight normal force
(`max(force_left, force_right) > --thr N`) over its 16 fps15 frames, using the same rule as
`vm_diffusion/scripts/wrist_contact_eval.py`:

  none    : no frame in contact
  full    : >= 80 % of the 16 frames in contact
  onset   : last 3 context frames free, last 4 future frames in contact   (no-contact -> contact)
  release : last 3 context frames in contact, last 4 future frames free  (contact -> no-contact)
  mixed   : everything else (counted, never evaluated)

`--n-per-cat` windows are drawn per class with a fixed seed and written out as *dataset indices*
for `rollout.py --window-stride 1 --windows <idx ...>`. The long-horizon picks are a nested subset
of the short-horizon picks (same windows, filtered to those with room for the full rollout), so the
two horizons are a paired comparison.

    python scripts/select_contact_windows.py --config runs/nano_lora_mb_v1/config.yaml \
        --split val --out runs/nano_lora_mb_v1/contact/sel_val.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from vtm_cosmos.data.dataset import VtmWindowDataset
from vtm_cosmos.data.splits import load_split
from vtm_cosmos.streams import FUTURE_ACTIONS, WINDOW_FRAMES

CATS = ("none", "full", "onset", "release", "mixed")
EVAL_CATS = ("none", "full", "onset", "release")
FORCE_FILE = "sensor_pose7d_force_fps15.pt"


def classify(c: np.ndarray, thr: float) -> str:
    """Contact class of one 16-frame window. Order matters: none/full win over the transitions."""
    w = c > thr
    ctx, fut = w[:8], w[8:]
    if not w.any():
        return "none"
    if w.mean() >= 0.8:
        return "full"
    if not ctx[-3:].any() and fut[-4:].all():
        return "onset"
    if ctx[-3:].all() and not fut[-4:].any():
        return "release"
    return "mixed"


def episode_force(action_root: Path, episode: str) -> np.ndarray:
    """Per-frame max(F_left, F_right) in newtons, on the same fps15 grid as the latent tree."""
    p = Path(action_root) / Path(episode).stem / FORCE_FILE
    if not p.exists():
        raise FileNotFoundError(f"{episode}: force file {p} not found")
    d = torch.load(p, map_location="cpu", weights_only=False)
    return np.maximum(d["force_left"].numpy(), d["force_right"].numpy())


def classify_split(ds: VtmWindowDataset, action_root: Path, thr: float):
    """-> (cls array over dataset indices, [(episode, t_start)], {episode: num_frames})."""
    n_frames = {info["episode"]: int(info["num_frames"]) for info in ds.episode_info}
    force = {}
    for ep in n_frames:
        f = episode_force(action_root, ep)
        if len(f) != n_frames[ep]:
            raise ValueError(f"{ep}: force file has {len(f)} frames, latent tree has {n_frames[ep]}")
        force[ep] = f
    cls = np.empty(len(ds), dtype=object)
    meta = []
    for i in range(len(ds)):
        ep, t0 = ds.window_meta(i)
        c = force[ep][t0:t0 + WINDOW_FRAMES]
        cls[i] = classify(c, thr) if len(c) == WINDOW_FRAMES else "mixed"
        meta.append((ep, t0))
    return cls, meta, n_frames


def write_fixed(args, ds: VtmWindowDataset, episodes: list[str]) -> None:
    """Re-select the exact windows of an earlier eval as stride-1 indices, split into shards."""
    prev = json.loads(Path(args.fixed_from).read_text())["results"]
    want = [(r["episode"], int(r["t_start"])) for r in prev]
    lookup = {}
    for i in range(len(ds)):
        lookup.setdefault(ds.window_meta(i), i)
    missing = [k for k in want if k not in lookup]
    if missing:
        raise ValueError(f"{len(missing)} windows of {args.fixed_from} are not in this split, e.g. {missing[:3]}")
    rows = [{"index": lookup[k], "episode": k[0], "t_start": k[1]} for k in want]
    n = args.fixed_shards
    size = -(-len(rows) // n)
    short = {f"fixed_{chr(ord('a') + s)}": rows[s * size:(s + 1) * size] for s in range(n)}
    out = {"split": args.split, "config": str(args.config), "fixed_from": str(args.fixed_from),
           "thr": args.thr, "seed": None, "rollout_windows": 0, "window_stride": 1,
           "episodes": episodes, "num_windows": len(ds), "short": short, "long": {}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"[select] fixed: {len(rows)} windows from {args.fixed_from} -> "
          + ", ".join(f"{k} {len(v)}" for k, v in short.items()) + f" -> {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="a run's config.yaml (data.latent_root / action_root / split_file)")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.5, help="contact threshold in newtons")
    ap.add_argument("--n-per-cat", type=int, default=25, help="short-horizon windows per class")
    ap.add_argument("--n-long-per-cat", type=int, default=8, help="long-horizon windows per class (nested subset)")
    ap.add_argument("--rollout-windows", type=int, default=15, help="chained windows a long rollout needs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cats", default=",".join(EVAL_CATS))
    ap.add_argument("--fixed-from", default=None,
                    help="window_eval.json of an earlier run: re-select exactly its windows (as stride-1 indices) "
                         "instead of sampling by contact class")
    ap.add_argument("--fixed-shards", type=int, default=2, help="split the fixed windows into this many pseudo-classes")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    action_root = Path(cfg.data.action_root)
    episodes = load_split(cfg.data.split_file)[args.split]

    # window_stride=1 so the dataset index enumerates every window; rollout must use the same.
    # action_full also drops windows with an invalid action transition, which renumbers the index --
    # build the dataset exactly as rollout.py does or the picks point at different windows.
    action_full = str(cfg.model.get("task_mode", "action_causal")) == "action_full"
    ds = VtmWindowDataset(
        episodes=episodes,
        latent_root=cfg.data.latent_root,
        action_root=cfg.data.action_root,
        action_file=cfg.data.action_file,
        window_stride=1,
        optional_streams=tuple(cfg.data.get("optional_streams", ()) or ()),
        include_view_actions=action_full,
        view_action_file=cfg.data.get("view_action_file", "view_actions_fps15.pt"),
        require_valid_actions=action_full,
    )
    print(f"[select] split={args.split} episodes={len(episodes)} windows={len(ds)} "
          f"presence={ds.presence_summary()}", flush=True)

    if args.fixed_from:
        write_fixed(args, ds, episodes)
        return

    cls, meta, n_frames = classify_split(ds, action_root, args.thr)
    counts = {c: int((cls == c).sum()) for c in CATS}
    print(f"[select] windows by class: {counts}", flush=True)

    # a long rollout scores t0 + 8*k for k in 0..R-1, so the last GT window starts at t0 + 8*(R-1)
    long_span = FUTURE_ACTIONS * (args.rollout_windows - 1) + WINDOW_FRAMES

    # every window a long rollout touches must itself be conditionable (present in this index)
    usable = set(meta)
    rng = np.random.RandomState(args.seed)
    short: dict[str, list[dict]] = {}
    long_: dict[str, list[dict]] = {}
    for cat in args.cats.split(","):
        pool = np.flatnonzero(cls == cat)
        if len(pool) < args.n_per_cat:
            print(f"[select] WARNING class '{cat}' has only {len(pool)} windows, "
                  f"asked for {args.n_per_cat}", flush=True)
        pick = np.sort(rng.choice(pool, min(args.n_per_cat, len(pool)), replace=False))
        rows = [{"index": int(i), "episode": meta[i][0], "t_start": int(meta[i][1])} for i in pick]
        short[cat] = rows
        room = [r for r in rows if r["t_start"] + long_span <= n_frames[r["episode"]]
                and all((r["episode"], r["t_start"] + FUTURE_ACTIONS * k) in usable
                        for k in range(args.rollout_windows))]
        long_[cat] = room[: args.n_long_per_cat]
        print(f"[select] {cat:8s} pool={len(pool):6d} short={len(rows):3d} "
              f"long={len(long_[cat]):3d} (of {len(room)} with room)", flush=True)

    out = {
        "split": args.split, "config": str(args.config), "thr": args.thr, "seed": args.seed,
        "n_per_cat": args.n_per_cat, "n_long_per_cat": args.n_long_per_cat,
        "rollout_windows": args.rollout_windows, "long_span_frames": long_span,
        "window_stride": 1, "counts": counts,
        "episodes": episodes, "num_windows": len(ds),
        "short": short, "long": long_,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    n_s = sum(len(v) for v in short.values()); n_l = sum(len(v) for v in long_.values())
    print(f"[select] wrote {args.out}: {n_s} short + {n_l} long windows", flush=True)


if __name__ == "__main__":
    main()
