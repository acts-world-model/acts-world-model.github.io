#!/usr/bin/env python
"""Force-informed 9D actions from the mode1_v2 parquets (motherboard episodes only).

The parquets carry per-native-frame calibrated normal force (react_calib, ~0.5 s
latency-compensated upstream) as columns force_{left,right}_normal_n, row-aligned 1:1
with sensor_{left,right}_pose — no timestamp alignment is needed here. This script runs
vm_diffusion/model/pose7d_action.py over each episode and writes, per episode folder
under --videos-root (default: the mode1_v2_actions tree, one subfolder per episode):

    <root>/<episode>/sensor_actions_with_<F>.pt         (T-1, 9) @ 30 fps
    <root>/<episode>/sensor_actions_with_<F>_fps<N>.pt  (F-1, 9) @  N fps

where N = --target-fps (default 15 = stride 2, the default Wan2.2 tree since 2026-09-07;
--target-fps 6 --latent-dir .../mode1_v2_video_latents_wan2.2/tac gives the stride-5
files) and <F> names the force encoding (--force-term):

    absolute -> sensor_actions_with_force*.pt    dp = R_t^T (p_{t+1} - p_t) + k_f * f[t]          * n
    delta    -> sensor_actions_with_dforce*.pt   dp = R_t^T (p_{t+1} - p_t) + k_f * (f[t+1]-f[t]) * n

with n = [0, -1, 0] and k_f = 1/(2 N/mm) = 0.0005 m/N. Same keys/meta as
sensor_actions{,_fps6}.pt (action/valid/rot_deg per side, frame_idx, meta with fps_stride
for the subsampled file), so `data.action_file:` can point straight at them.

Force never touches rotation, so rot_deg/valid are identical to the pure-pose files.
Downsampling converts directly between the stride-N pose endpoints (bit-exact with the
SE(3) composition of native steps for the pose part). Under 'absolute' the force of each
coarse step is the one at its FIRST native frame (force_agg='start'), which makes the
applied force per action k exactly the force at native frame frame_idx[k] — same source,
same index; at stride 5 that is force_fps6.pt['force_<side>'][k] verbatim. The cost of
'start' vs the span mean at stride 5: displacement rms 0.255 mm, max 2.69 mm (29% of the
force term's own std). Under 'delta' the force is the endpoint difference and force_agg
does not apply. The absolute term is a per-frame state and does not depend on fps; the
delta term grows with the stride like dp does.

NOTE: valid_*/frame_idx are consumed by the Stage-1 builders only — the MoT world-model
dataset (`dataset.load_episode_actions`) reads action_*/meta and never applies the 30-deg
rotation dropout.

pusht episodes have no force columns and are skipped. Existing outputs are skipped
unless --overwrite.

Usage (needs pyarrow + torch; effecterase env):
    /group2/ct/weihanx/miniconda3/envs/effecterase/bin/python \
        vm_diffusion/data/build_force_actions.py [--force-term delta] [--episodes ...] \
        [--overwrite] [--dry-run]

fps6 tree (stride 5, explicit):
    ... build_force_actions.py --target-fps 6 \
        --latent-dir /group2/ct/weihanx/world_model/mode1_v2_video_latents_wan2.2/tac
"""
from __future__ import annotations

import argparse
import os
import math
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                        # build_episode_actions (same dir)
sys.path.insert(0, str(_HERE.parent / "model"))       # pose7d_action

from build_episode_actions import read_poses, rotation_deg  # noqa: E402
from pose7d_action import (  # noqa: E402
    DEFAULT_NORMAL_AXIS,
    DEFAULT_STIFFNESS_N_PER_MM,
    FORCE_TERMS,
    Pose7DActionConfig,
    convert_pose_sequence,
    stiffness_to_k_f,
)

DEFAULT_PARQUET_ROOT = Path(os.environ.get("ACTS_PARQUET_ROOT", "mode1_v2"))
DEFAULT_VIDEOS_ROOT = Path(os.environ.get("ACTS_ACTIONS_OUT", "mode1_v2_videos"))   # legacy tree (absolute files)
# New per-episode action tree (2026-09-07): default OUTPUT root for every builder here.
DEFAULT_ACTIONS_ROOT = Path(os.environ.get("ACTS_ACTIONS_OUT", "mode1_v2_actions"))
DEFAULT_LATENT_DIR = Path(os.environ.get("ACTS_TAC_LATENTS", "latents/tac"))
DEFAULT_TARGET_FPS = 15.0
# Output names, keyed by force_term. Rebound in main() from --force-term / --target-fps /
# --out-fps-name; build_episode reads the module globals at call time, so both stay in sync.
FORCE_TERM_TAG = {"absolute": "force", "delta": "dforce"}
OUT_NATIVE = "sensor_actions_with_force.pt"
OUT_SUB = f"sensor_actions_with_force_fps{int(DEFAULT_TARGET_FPS)}.pt"
FORCE_COLUMNS = ("force_left_normal_n", "force_right_normal_n")
FORCE_AGG = "start"  # ('absolute') force of coarse step k = force at native frame stride*k
MAX_ROT_DEG = 30.0   # same dropout rule as build_episode_actions.py


def read_forces(path: Path) -> dict[str, np.ndarray] | None:
    """Parquet -> {'left': (T,), 'right': (T,)} newtons, or None if absent (pusht)."""
    import pyarrow.parquet as pq

    have = set(pq.ParquetFile(str(path)).schema_arrow.names)
    if not set(FORCE_COLUMNS) <= have:
        return None
    table = pq.read_table(str(path), columns=list(FORCE_COLUMNS))
    out = {}
    for side, col in zip(("left", "right"), FORCE_COLUMNS):
        f = np.asarray(table.column(col).to_pylist(), dtype=np.float64)
        if not np.isfinite(f).all() or (f < 0).any():
            raise RuntimeError(f"{path.name}: bad values in {col}")
        out[side] = f
    return out


def base_meta(parquet: Path, cfg: Pose7DActionConfig) -> dict:
    return {
        "layout": "[dp_x, dp_y, dp_z, r1..r6]",
        "frame": ("sensor frame at time t; dp = R_t^T (p_{t+1} - p_t) "
                  "+ k_f * f_n * normal_axis, dR = R_t^T R_{t+1}"),
        "rot6d": ("raw Rot6D(dR), first two columns as [c1x,c1y,c1z, c2x,c2y,c2z]; "
                  "NOT identity-subtracted"),
        "units": "metres",
        "quat_order": "xyzw",
        "source": str(parquet),
        "max_rot_deg": MAX_ROT_DEG,
        "gel_frame": False,
        # force-specific
        "force_informed": True,
        "force_term": cfg.force_term,
        "stiffness_n_per_mm": cfg.stiffness_n_per_mm,
        "k_f": cfg.resolved_k_f(),
        "normal_axis": tuple(cfg.normal_axis),
        "force_source": "force_<side>_normal_n (twm.force_recovery / react_calib), "
                        "row-aligned with the poses in the same parquet",
        "force_note": (
            "force enters translation only, along normal_axis; rotation and rot_deg/valid are "
            "identical to sensor_actions*.pt. force_term='absolute': k_f * f[start] (per-frame "
            "indentation state, fps-independent); force_term='delta': k_f * (f[end] - f[start]) "
            "(change of indentation over the step, grows with the stride like dp)"
        ),
    }


def build_side(pose: torch.Tensor, force: np.ndarray, cfg: Pose7DActionConfig):
    """-> (action (n,9) f32, valid (n,) bool, rot_deg (n,) f32, frame_idx (F,) i64)."""
    actions, fidx = convert_pose_sequence(pose.numpy(), force, cfg)
    a = torch.from_numpy(actions).float()
    if not torch.isfinite(a).all():
        raise RuntimeError("non-finite action")
    rd = rotation_deg(a)                 # force does not touch the rotation block
    return a, rd <= MAX_ROT_DEG, rd.float(), fidx


def valid_and(valid_native: torch.Tensor, fidx: np.ndarray) -> torch.Tensor:
    """A coarse step is valid only if every native step it spans is valid."""
    return torch.tensor([bool(valid_native[fidx[k]:fidx[k + 1]].all())
                         for k in range(len(fidx) - 1)])


def build_episode(parquet: Path, out_dir: Path, latent_dir: Path,
                  cfg_native: Pose7DActionConfig, cfg_sub: Pose7DActionConfig,
                  overwrite: bool, dry_run: bool) -> str:
    todo = [n for n in (OUT_NATIVE, OUT_SUB) if overwrite or not (out_dir / n).exists()]
    if not todo:
        return f"[skip] {parquet.stem}: outputs exist"

    forces = read_forces(parquet)
    if forces is None:
        return f"[skip] {parquet.stem}: no force columns (pusht?)"
    pose_l, pose_r, _ = read_poses(parquet)
    T = int(pose_l.shape[0])
    if forces["left"].shape[0] != T:
        raise RuntimeError(f"{parquet.stem}: force rows {forces['left'].shape[0]} != poses {T}")

    lat = torch.load(latent_dir / f"{parquet.stem}.pt", map_location="cpu", weights_only=False)
    fs = int(lat["fps_stride"])
    if fs != cfg_sub.stride():
        raise RuntimeError(f"{parquet.stem}: latent fps_stride={fs} != config stride {cfg_sub.stride()}")
    n_frames = math.ceil(T / fs)
    if n_frames != int(lat["num_frames"]):
        raise RuntimeError(f"{parquet.stem}: derived {n_frames} frames but latents say {lat['num_frames']}")

    payloads: dict[str, dict] = {}
    valid_native: dict[str, torch.Tensor] = {}   # filled by the OUT_NATIVE pass first
    for out_name, cfg in ((OUT_NATIVE, cfg_native), (OUT_SUB, cfg_sub)):
        payload: dict[str, object] = {}
        for side, pose in (("left", pose_l), ("right", pose_r)):
            a, valid, rd, fidx = build_side(pose, forces[side], cfg)
            if out_name == OUT_SUB:
                # native validity is authoritative: AND over the spanned native steps
                valid = valid_and(valid_native[side], fidx)
            else:
                valid_native[side] = valid
            payload[f"action_{side}"] = a
            payload[f"valid_{side}"] = valid
            payload[f"rot_deg_{side}"] = rd
        payload["frame_idx"] = torch.from_numpy(fidx).to(torch.int32)
        payload["meta"] = base_meta(parquet, cfg)
        if out_name == OUT_SUB:
            payload["meta"].update({
                "fps": float(lat["fps"]),
                "src_fps": float(lat["src_fps"]),
                "fps_stride": fs,
                "num_frames": n_frames,
                "native_num_frames": T,
                "force_agg": cfg.force_agg if cfg.force_term == "absolute" else None,
                "downsampled_from": str(parquet),
                "downsample": (
                    "action[k] converts directly between poses frame_idx[k] -> frame_idx[k+1] "
                    "(== ordered SE(3) composition of the spanned native steps); "
                    + ("force taken at native frame frame_idx[k] ('start'); at stride 5 that is "
                       "force_fps6.pt[k]" if cfg.force_term == "absolute" else
                       "force = f[frame_idx[k+1]] - f[frame_idx[k]] (endpoint difference)")
                ),
            })
        payloads[out_name] = payload

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        for out_name in todo:
            out_path = out_dir / out_name
            tmp = out_path.with_name(out_path.name + ".tmp")
            torch.save(payloads[out_name], tmp)
            tmp.replace(out_path)

    n6 = payloads[OUT_SUB]["action_left"].shape[0]
    flagged = int((~payloads[OUT_NATIVE]["valid_left"]).sum()
                  + (~payloads[OUT_NATIVE]["valid_right"]).sum())
    return (f"[write] {parquet.stem}: T={T} native={T - 1} {OUT_SUB}={n6} actions x9 per side "
            f"({flagged} native steps flagged)" + (" [dry-run]" if dry_run else ""))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--parquet-root", type=Path, default=DEFAULT_PARQUET_ROOT)
    ap.add_argument("--videos-root", type=Path, default=DEFAULT_ACTIONS_ROOT,
                    help="OUTPUT root; one subfolder per episode (default: the mode1_v2_actions "
                         "tree; the legacy absolute files live under mode1_v2_videos)")
    ap.add_argument("--latent-dir", type=Path, default=DEFAULT_LATENT_DIR,
                    help="tactile latent dir; fps/fps_stride/num_frames are validated against it "
                         "(default: the fps15 tree)")
    ap.add_argument("--force-term", choices=FORCE_TERMS, default="absolute",
                    help="force encoding: 'absolute' -> *_with_force*, 'delta' -> *_with_dforce*")
    ap.add_argument("--episodes", nargs="*", default=None,
                    help="explicit episode stems (default: every motherboard parquet)")
    ap.add_argument("--stiffness-n-per-mm", type=float, default=DEFAULT_STIFFNESS_N_PER_MM,
                    help="gel normal stiffness k; k_f = 1/(k*1000) m/N")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--target-fps", type=float, default=DEFAULT_TARGET_FPS,
                    help="rate of the subsampled output; round(src_fps/target_fps) must equal "
                         "the fps_stride of the latents under --latent-dir (default 15)")
    ap.add_argument("--out-fps-name", default=None,
                    help="subsampled output filename "
                         "(default sensor_actions_with_<force|dforce>_fps<int(target-fps)>.pt)")
    ap.add_argument("--dry-run", action="store_true", help="build + report without writing")
    args = ap.parse_args()

    global OUT_NATIVE, OUT_SUB
    tag = FORCE_TERM_TAG[args.force_term]
    OUT_NATIVE = f"sensor_actions_with_{tag}.pt"
    OUT_SUB = args.out_fps_name or f"sensor_actions_with_{tag}_fps{int(args.target_fps)}.pt"

    if args.episodes:
        parquets = [args.parquet_root / f"{e}.parquet" for e in args.episodes]
        for p in parquets:
            if not p.exists():
                raise FileNotFoundError(p)
    else:
        parquets = sorted(args.parquet_root.glob("motherboard_*.parquet"))
    if not parquets:
        raise SystemExit(f"no motherboard parquet under {args.parquet_root}")

    common = dict(use_force=True, stiffness_n_per_mm=args.stiffness_n_per_mm,
                  normal_axis=DEFAULT_NORMAL_AXIS, src_fps=30.0, force_term=args.force_term)
    cfg_native = Pose7DActionConfig(**common, target_fps=None)
    cfg_sub = Pose7DActionConfig(**common, target_fps=args.target_fps, force_agg=FORCE_AGG)

    print(f"{len(parquets)} episode(s) -> {args.videos_root}/*/{{{OUT_NATIVE},{OUT_SUB}}}  "
          f"(target_fps={args.target_fps:g} stride={cfg_sub.stride()}, latents={args.latent_dir}, "
          f"k={args.stiffness_n_per_mm} N/mm, k_f={stiffness_to_k_f(args.stiffness_n_per_mm):g} m/N, "
          f"normal_axis={DEFAULT_NORMAL_AXIS}, force_term={args.force_term}, "
          f"force_agg={FORCE_AGG if args.force_term == 'absolute' else None})")
    n_written = 0
    for parquet in parquets:
        out_dir = args.videos_root / parquet.stem
        if not out_dir.is_dir() and args.dry_run:
            print(f"  [note] {parquet.stem}: destination folder {out_dir} does not exist yet")
        msg = build_episode(parquet, out_dir, args.latent_dir, cfg_native, cfg_sub,
                            args.overwrite, args.dry_run)
        print(f"  {msg}")
        n_written += msg.startswith("[write]") and not args.dry_run
    print(f"done ({n_written} episodes written)")


if __name__ == "__main__":
    main()
