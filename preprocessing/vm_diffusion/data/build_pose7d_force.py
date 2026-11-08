#!/usr/bin/env python
"""Per-frame 7D pose + normal-force streams for the `action_chunking` representation.

`action_chunking` (vm_diffusion/model/pose7d_action.py) expresses every frame of a
training window relative to the window's FIRST frame (the anchor):

    a_k = [ R_0^T (p_{k+1} - p_0) + k_f * F_k * [0,-1,0] ,  Rot6D(R_0^T R_{k+1}) ]

    F_k = f_{k+1}         (data.force_term: absolute -- the arrival frame's indentation)
        = f_{k+1} - f_0   (data.force_term: delta    -- change of indentation since the anchor)

Because the anchor depends on the window, and latent-cache windows overlap, the rows
cannot be precomputed per frame. This script therefore stores the INPUTS at the latent
rate -- one 7D pose and one force per fps-subsampled frame -- and the dataset builds
the rows per window with `raw_pose_chunk_to_force_informed_9d`. The stream is
encoding-agnostic: which force_term is applied is decided by the training config, not
by this file.

    <videos-root>/<episode>/sensor_pose7d_force_fps<N>.pt

        pose_left / pose_right    (F, 7) float64   [x,y,z,qx,qy,qz,qw], world frame
        force_left / force_right  (F,)   float32   normal force [N] at that frame
        valid_left / valid_right  (F-1,) bool      DELTA-step validity (rot <= 30 deg,
                                                   AND over the spanned native steps);
                                                   chunk row k is trustworthy iff
                                                   valid[:k+1].all()
        frame_idx                 (F,)   int32     native frame index of each row
        meta                      dict             fps_stride / num_frames / k_f / ...

Frame grid and force sampling match the delta_action files exactly: frame_idx =
arange(0, T, stride), force point-sampled at native frame frame_idx[k] (== the 'start'
force of sensor_actions_with_force_fps<N>.pt row k, == force_fps6.pt[k] at stride 5).
When a delta twin (sensor_actions_with_{force,dforce}_fps<N>.pt) exists in the output
folder its valid_*/frame_idx are asserted equal, as a consistency check.

pusht episodes have no force columns and are skipped. Existing outputs are skipped
unless --overwrite.

Usage (needs pyarrow + torch; effecterase env). Defaults: fps15 (stride 2) and the
mode1_v2_actions output tree:
    /group2/ct/weihanx/miniconda3/envs/effecterase/bin/python \
        vm_diffusion/data/build_pose7d_force.py [--episodes ...] [--overwrite] [--dry-run]

fps6 tree (stride 5, explicit):
    ... build_pose7d_force.py --target-fps 6 \
        --latent-dir /group2/ct/weihanx/world_model/mode1_v2_video_latents_wan2.2/tac
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                        # build_episode_actions / build_force_actions
sys.path.insert(0, str(_HERE.parent / "model"))       # pose7d_action

from build_episode_actions import read_poses, rotation_deg  # noqa: E402
from build_force_actions import (  # noqa: E402
    DEFAULT_ACTIONS_ROOT,
    DEFAULT_LATENT_DIR,
    DEFAULT_PARQUET_ROOT,
    DEFAULT_TARGET_FPS,
    MAX_ROT_DEG,
    read_forces,
    valid_and,
)
from pose7d_action import (  # noqa: E402
    DEFAULT_NORMAL_AXIS,
    DEFAULT_STIFFNESS_N_PER_MM,
    Pose7DActionConfig,
    raw_pose_sequence_to_force_informed_9d,
    stiffness_to_k_f,
    subsample_pose_sequence,
)

CHUNK_FORMULA = ("row k of a window with anchor frame 0: dp = R_0^T (p_{k+1} - p_0) + k_f * F_k "
                 "* normal_axis, dR = R_0^T R_{k+1}; F_k = f_{k+1} (force_term absolute) or "
                 "f_{k+1} - f_0 (force_term delta), chosen by the training config; normal not "
                 "rotated; built per window by pose7d_action.raw_pose_chunk_to_force_informed_9d")


def delta_files_for(out_name: str) -> list[str]:
    """sensor_pose7d_force_fps6.pt -> the delta twins sharing its grid (either force encoding)."""
    return [out_name.replace("sensor_pose7d_force", f"sensor_actions_with_{tag}")
            for tag in ("force", "dforce")]


def build_side(pose: torch.Tensor, force: np.ndarray, cfg: Pose7DActionConfig):
    """-> (pose_sub (F,7) f64, force_sub (F,) f32, valid (F-1,) bool, frame_idx (F,) i64)."""
    poses_np = pose.numpy()
    pose_sub, force_sub, fidx = subsample_pose_sequence(poses_np, force, cfg)
    # delta-step validity from the pure-pose native stream (force never touches rotation)
    native = torch.from_numpy(
        raw_pose_sequence_to_force_informed_9d(poses_np, np.zeros(poses_np.shape[0]), k_f=0.0)).float()
    valid_native = rotation_deg(native) <= MAX_ROT_DEG
    valid = valid_and(valid_native, fidx)
    if not (np.isfinite(pose_sub).all() and np.isfinite(force_sub).all()):
        raise RuntimeError("non-finite pose/force")
    return (torch.from_numpy(pose_sub), torch.from_numpy(force_sub).float(), valid,
            torch.from_numpy(fidx))


def build_episode(parquet: Path, out_dir: Path, latent_dir: Path, cfg: Pose7DActionConfig,
                  out_name: str, overwrite: bool, dry_run: bool) -> str:
    out_path = out_dir / out_name
    if out_path.exists() and not overwrite:
        return f"[skip] {parquet.stem}: {out_name} exists"

    forces = read_forces(parquet)
    if forces is None:
        return f"[skip] {parquet.stem}: no force columns (pusht?)"
    pose_l, pose_r, _ = read_poses(parquet)
    T = int(pose_l.shape[0])
    if forces["left"].shape[0] != T:
        raise RuntimeError(f"{parquet.stem}: force rows {forces['left'].shape[0]} != poses {T}")

    lat = torch.load(latent_dir / f"{parquet.stem}.pt", map_location="cpu", weights_only=False)
    fs = int(lat["fps_stride"])
    if fs != cfg.stride():
        raise RuntimeError(f"{parquet.stem}: latent fps_stride={fs} != config stride {cfg.stride()}")
    n_frames = math.ceil(T / fs)
    if n_frames != int(lat["num_frames"]):
        raise RuntimeError(f"{parquet.stem}: derived {n_frames} frames but latents say {lat['num_frames']}")

    payload: dict[str, object] = {}
    fidx_ref = None
    for side, pose in (("left", pose_l), ("right", pose_r)):
        p, f, valid, fidx = build_side(pose, forces[side], cfg)
        if p.shape[0] != n_frames:
            raise RuntimeError(f"{parquet.stem}/{side}: {p.shape[0]} subsampled frames != {n_frames}")
        payload[f"pose_{side}"] = p
        payload[f"force_{side}"] = f
        payload[f"valid_{side}"] = valid
        fidx_ref = fidx
    payload["frame_idx"] = fidx_ref.to(torch.int32)
    payload["meta"] = {
        "action_repr": "action_chunking",
        "layout": "pose7d [x,y,z,qx,qy,qz,qw] world frame per frame; normal force [N] per frame",
        "quat_order": "xyzw",
        "units": "metres / newtons",
        "chunk_formula": CHUNK_FORMULA,
        "force_sample": "frame",
        "force_note": ("force_<side>[k] = native force at frame_idx[k] (== delta file's 'start' "
                       "force); per-frame ABSOLUTE newtons -- the encoding (force_term) is "
                       "applied by the dataset, not stored here"),
        "valid_note": ("valid[k] is the DELTA step k->k+1 (AND over spanned native steps); "
                       "a chunk row k is trustworthy iff valid[:k+1].all()"),
        "max_rot_deg": MAX_ROT_DEG,
        "gel_frame": False,
        "stiffness_n_per_mm": cfg.stiffness_n_per_mm,
        "k_f": cfg.resolved_k_f(),
        "normal_axis": tuple(cfg.normal_axis),
        "force_source": "force_<side>_normal_n (twm.force_recovery / react_calib), "
                        "row-aligned with the poses in the same parquet",
        "source": str(parquet),
        "fps": float(lat["fps"]),
        "src_fps": float(lat["src_fps"]),
        "fps_stride": fs,
        "num_frames": n_frames,
        "native_num_frames": T,
        "downsampled_from": str(parquet),
    }

    # consistency check against whichever delta twins exist (same grid, same valid rule)
    checked = ""
    for delta_path in (out_dir / n for n in delta_files_for(out_name)):
        if not delta_path.is_file():
            continue
        dd = torch.load(delta_path, map_location="cpu", weights_only=False)
        if not torch.equal(dd["frame_idx"].to(torch.int32), payload["frame_idx"]):
            raise RuntimeError(f"{parquet.stem}: frame_idx differs from {delta_path.name}")
        for side in ("left", "right"):
            if not torch.equal(dd[f"valid_{side}"].bool(), payload[f"valid_{side}"].bool()):
                raise RuntimeError(f"{parquet.stem}: valid_{side} differs from {delta_path.name}")
        checked += f", valid/frame_idx == {delta_path.name}"
    if not checked:
        checked = ", no delta twin in the output folder (check skipped)"

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(out_path.name + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(out_path)

    flagged = int((~payload["valid_left"]).sum() + (~payload["valid_right"]).sum())
    return (f"[write] {parquet.stem}: T={T} -> {out_name} F={n_frames} frames x (7 pose + 1 force) "
            f"per side ({flagged} delta steps flagged{checked})" + (" [dry-run]" if dry_run else ""))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--parquet-root", type=Path, default=DEFAULT_PARQUET_ROOT)
    ap.add_argument("--videos-root", type=Path, default=DEFAULT_ACTIONS_ROOT,
                    help="OUTPUT root; one subfolder per episode (default: the mode1_v2_actions tree)")
    ap.add_argument("--latent-dir", type=Path, default=DEFAULT_LATENT_DIR,
                    help="tactile latent dir; fps/fps_stride/num_frames are validated against it "
                         "(default: the fps15 tree)")
    ap.add_argument("--episodes", nargs="*", default=None,
                    help="explicit episode stems (default: every motherboard parquet)")
    ap.add_argument("--stiffness-n-per-mm", type=float, default=DEFAULT_STIFFNESS_N_PER_MM,
                    help="gel normal stiffness k; k_f = 1/(k*1000) m/N (stored in meta)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--target-fps", type=float, default=DEFAULT_TARGET_FPS,
                    help="rate of the output; round(src_fps/target_fps) must equal the "
                         "fps_stride of the latents under --latent-dir (default 15)")
    ap.add_argument("--out-fps-name", default=None,
                    help="output filename (default sensor_pose7d_force_fps<int(target-fps)>.pt)")
    ap.add_argument("--dry-run", action="store_true", help="build + report without writing")
    args = ap.parse_args()

    out_name = args.out_fps_name or f"sensor_pose7d_force_fps{int(args.target_fps)}.pt"

    if args.episodes:
        parquets = [args.parquet_root / f"{e}.parquet" for e in args.episodes]
        for p in parquets:
            if not p.exists():
                raise FileNotFoundError(p)
    else:
        parquets = sorted(args.parquet_root.glob("motherboard_*.parquet"))
    if not parquets:
        raise SystemExit(f"no motherboard parquet under {args.parquet_root}")

    cfg = Pose7DActionConfig(use_force=True, stiffness_n_per_mm=args.stiffness_n_per_mm,
                             normal_axis=DEFAULT_NORMAL_AXIS, src_fps=30.0,
                             target_fps=args.target_fps, action_repr="action_chunking")

    print(f"{len(parquets)} episode(s) -> {args.videos_root}/*/{out_name}  "
          f"(target_fps={args.target_fps:g} stride={cfg.stride()}, latents={args.latent_dir}, "
          f"k={args.stiffness_n_per_mm} N/mm, k_f={stiffness_to_k_f(args.stiffness_n_per_mm):g} m/N, "
          f"normal_axis={DEFAULT_NORMAL_AXIS}, action_repr=action_chunking)")
    n_written = 0
    for parquet in parquets:
        out_dir = args.videos_root / parquet.stem
        if not out_dir.is_dir() and args.dry_run:
            print(f"  [note] {parquet.stem}: destination folder {out_dir} does not exist yet")
        msg = build_episode(parquet, out_dir, args.latent_dir, cfg, out_name,
                            args.overwrite, args.dry_run)
        print(f"  {msg}")
        n_written += msg.startswith("[write]") and not args.dry_run
    print(f"done ({n_written} episodes written)")


if __name__ == "__main__":
    main()
