#!/usr/bin/env python
"""Precompute per-episode sensor-frame 9D actions from the mode1_v2 parquets.

Reads the world-frame sensor poses

    <parquet-root>/<episode>.parquet   columns sensor_left_pose / sensor_right_pose,
                                       each (T, 7) as [x, y, z, qx, qy, qz, qw]

and writes, next to that episode's videos,

    <videos-root>/<episode>/sensor_actions.pt
        action_left  : (T-1, 9) float32   [dp(3, metres), Rot6D(dR)(6)]
        action_right : (T-1, 9) float32
        valid_left   : (T-1,)   bool      False where the step looks like tracking dropout
        valid_right  : (T-1,)   bool
        rot_deg_left : (T-1,)   float32   per-step rotation magnitude, degrees
        rot_deg_right: (T-1,)   float32
        frame_idx    : (T,)     int32     action[t] maps frame t -> t+1
        meta         : dict     conventions, so consumers need not guess

Both actions are expressed in the sensor frame at time t (see common/actions.py);
the rotation is stored as raw Rot6D(dR), NOT identity-subtracted — that transform
belongs at the model input, alongside standardization, so this file stays a
faithful encoding of the pose stream.

Left and right stay as two separate (T-1, 9) tensors rather than one (T-1, 18):
consumers that want the concatenated form build it at the model boundary, and
single-sided uses do not have to slice.

With --gel-frame the poses are first composed with T_rigid_gel (see common/gel_frame.py)
and written to sensor_actions_gel.pt instead. The action is then expressed in the GEL
frame: +z is the real gel normal and the origin sits on the gel surface, so dp_z is press
depth and dp_x, dp_y are in-plane shear. sensor_action_9d itself is unchanged — only its
input poses are. rot_deg and valid are identical either way, because dR_gel = R_g^T dR R_g
is a conjugation and so preserves the geodesic angle.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.actions import rot6d_to_matrix, sensor_action_9d  # noqa: E402
from common.gel_frame import calib_dir_for, gel_frame, poses_to_gel  # noqa: E402

DEFAULT_PARQUET_ROOT = Path(os.environ.get("ACTS_PARQUET_ROOT", "mode1_v2"))
DEFAULT_VIDEOS_ROOT = Path(os.environ.get("ACTS_ACTIONS_OUT", "mode1_v2_videos"))
OUT_NAME = "sensor_actions.pt"
GEL_OUT_NAME = "sensor_actions_gel.pt"

POSE_COLUMNS = ("sensor_left_pose", "sensor_right_pose")


def read_poses(path: Path) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Parquet -> (pose_left, pose_right) as (T, 7) float64 tensors, plus frame_idx."""
    import pyarrow.parquet as pq

    cols = list(POSE_COLUMNS)
    have = set(pq.ParquetFile(str(path)).schema_arrow.names)
    if "frame_idx" in have:
        cols.append("frame_idx")
    missing = set(POSE_COLUMNS) - have
    if missing:
        raise RuntimeError(f"{path.name}: missing pose column(s) {sorted(missing)}")

    table = pq.read_table(str(path), columns=cols)

    def _pose(name: str) -> torch.Tensor:
        arr = np.asarray(table.column(name).to_pylist(), dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 7:
            raise RuntimeError(f"{path.name}: expected (T,7) for {name}, got {arr.shape}")
        return torch.from_numpy(arr)

    pl, pr = _pose(POSE_COLUMNS[0]), _pose(POSE_COLUMNS[1])
    if pl.shape != pr.shape:
        raise RuntimeError(f"{path.name}: left/right pose length mismatch {pl.shape} vs {pr.shape}")

    if "frame_idx" in cols:
        frame_idx = np.asarray(table.column("frame_idx").to_pylist(), dtype=np.int32)
    else:
        frame_idx = np.arange(pl.shape[0], dtype=np.int32)

    return pl, pr, frame_idx


def rotation_deg(action9: torch.Tensor) -> torch.Tensor:
    """Per-step geodesic rotation magnitude, degrees, recovered from the 6D block."""
    R = rot6d_to_matrix(action9[:, 3:])
    cos = (R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0
    return torch.rad2deg(torch.arccos(cos.clamp(-1.0, 1.0)))


def build_episode(parquet: Path, out_dir: Path, max_rot_deg: float,
                  out_name: str = OUT_NAME, gel: bool = False) -> dict:
    pose_l, pose_r, frame_idx = read_poses(parquet)

    payload: dict[str, object] = {}
    gel_meta: dict[str, object] = {}
    n_flagged = 0
    for side, pose in (("left", pose_l), ("right", pose_r)):
        if gel:
            # Re-express the poses in the gel frame BEFORE the (unchanged) converter, so
            # the 9D action comes out with +z along the real gel normal and its origin on
            # the gel surface. Every downstream raster inherits the fix for free.
            R_g, c_gel, info = gel_frame(side, calib_dir_for(parquet.stem))
            pose = poses_to_gel(pose, R_g, c_gel)
            gel_meta[side] = info
        action = sensor_action_9d(pose)
        rot_deg = rotation_deg(action)
        # A >30 deg/frame step at 30 fps is tracking dropout or a quaternion sign
        # flip, not real motion. Flag rather than drop, so indices stay aligned.
        valid = rot_deg <= max_rot_deg
        n_flagged += int((~valid).sum())

        if not torch.isfinite(action).all():
            raise RuntimeError(f"{parquet.name}: non-finite action for {side}")

        payload[f"action_{side}"] = action.float()
        payload[f"valid_{side}"] = valid
        payload[f"rot_deg_{side}"] = rot_deg.float()

    payload["frame_idx"] = torch.from_numpy(frame_idx)
    payload["meta"] = {
        "layout": "[dp_x, dp_y, dp_z, r1..r6]",
        "frame": "sensor frame at time t; dp = R_t^T (p_{t+1} - p_t), dR = R_t^T R_{t+1}",
        "rot6d": "raw Rot6D(dR), first two columns as [c1x,c1y,c1z, c2x,c2y,c2z]; NOT identity-subtracted",
        "units": "metres",
        "quat_order": "xyzw",
        "source": str(parquet),
        "max_rot_deg": max_rot_deg,
        "gel_frame": bool(gel),
    }
    if gel:
        # Record the full basis: the in-plane axes are a free choice, so Stage 2 has to
        # be able to reproduce this exact one.
        payload["meta"]["gel_calibration"] = gel_meta
        payload["meta"]["frame"] = (
            "GEL frame at time t (+z = gel_axis_in_rigid, origin = gel surface centre); "
            "poses composed with T_rigid_gel, then the standard sensor_action_9d"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    tmp = out_path.with_name(out_path.name + ".tmp")   # NOT with_suffix: it eats ".pt"
    torch.save(payload, tmp)
    tmp.replace(out_path)

    return {
        "path": out_path,
        "T": int(pose_l.shape[0]),
        "n_actions": int(payload["action_left"].shape[0]),
        "flagged": n_flagged,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--parquet-root", type=Path, default=DEFAULT_PARQUET_ROOT)
    ap.add_argument("--videos-root", type=Path, default=DEFAULT_VIDEOS_ROOT)
    ap.add_argument("--episodes", nargs="*", default=None,
                    help="explicit episode stems (default: every parquet in --parquet-root)")
    ap.add_argument("--max-rot-deg", type=float, default=30.0,
                    help="per-step rotation above this is flagged invalid (not dropped)")
    ap.add_argument("--gel-frame", action="store_true",
                    help="express the action in the gel frame (+z = gel normal, origin on "
                         "the gel surface) instead of the raw OptiTrack rigid frame")
    ap.add_argument("--out-name", default=None,
                    help=f"output filename (default: {OUT_NAME}, or {GEL_OUT_NAME} with --gel-frame)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_name = args.out_name or (GEL_OUT_NAME if args.gel_frame else OUT_NAME)

    if args.episodes:
        parquets = [args.parquet_root / f"{e}.parquet" for e in args.episodes]
        for p in parquets:
            if not p.exists():
                raise FileNotFoundError(p)
    else:
        parquets = sorted(args.parquet_root.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"no parquet found under {args.parquet_root}")

    print(f"building {'GEL-frame ' if args.gel_frame else ''}actions for {len(parquets)} "
          f"episode(s) -> {args.videos_root}/*/{out_name}")
    n_written = n_skipped = 0
    total_flagged = 0
    for parquet in parquets:
        out_dir = args.videos_root / parquet.stem
        if not out_dir.is_dir():
            # A parquet with no video folder means the two trees disagree; that is a
            # data problem worth stopping for, not something to paper over.
            raise FileNotFoundError(
                f"{parquet.name}: no destination folder {out_dir} — parquet and video trees disagree"
            )
        if (out_dir / out_name).exists() and not args.overwrite:
            print(f"  [skip] {parquet.stem}: {out_name} exists")
            n_skipped += 1
            continue

        info = build_episode(parquet, out_dir, args.max_rot_deg, out_name, args.gel_frame)
        total_flagged += info["flagged"]
        print(f"  [write] {parquet.stem}: T={info['T']} actions={info['n_actions']}x9 "
              f"per side ({info['flagged']} flagged)")
        n_written += 1

    print(f"done ({n_written} written, {n_skipped} skipped, {total_flagged} flagged steps)")


if __name__ == "__main__":
    main()
