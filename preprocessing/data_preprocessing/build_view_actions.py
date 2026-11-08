#!/usr/bin/env python
"""Batch the sensor-pose -> camera projection over every episode -> <episode>/view_actions.pt.

For each episode this projects both tactile sensors' absolute pose into all three camera views
(see project_pose_to_camera.py for the geometry and its departures from arXiv 2604.06168) and
stores the result on the **6 fps grid** the rest of the pipeline uses.

What is stored is the projected *points*, not the RGB Gaussian "action image". The image is a
deterministic rendering of these three points, so keeping the points (~250 kB/episode instead of
~130 MB) leaves sigma, resolution and the render itself changeable in the dataloader without
rebuilding anything. Use build_action_images.py to materialize images when a run needs them.

These are per-FRAME quantities (an absolute pose), not per-transition, so N = ceil(n_native/5) --
the same length as the tactile latents' num_frames, one more than the fps6 action count.

    python data_preprocessing/build_view_actions.py            # all episodes
    python data_preprocessing/build_view_actions.py --episodes motherboard_0510_episode_004
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_pose_to_camera import (  # noqa: E402
    PARQUET_ROOT, ROOT, SENSORS, VIEWS, AXIS_LEN_MM, load_calibration, load_episode_poses,
    project, sensor_points, task_for_episode, world_frame_offset,
)

VID = ROOT / "mode1_v2_videos"
OUT_NAME = "view_actions.pt"
POINT_ORDER = ("pos", "normal", "up")


def build_episode(episode: str, stride: int, apply_world_offset: bool = False) -> dict:
    task = task_for_episode(episode)
    cal = load_calibration(task)
    poses = load_episode_poses(episode, apply_world_offset=apply_world_offset)

    n_native = len(poses["left"])
    idx = np.arange(0, n_native, stride)          # 0, 5, 10, ... -- same grid as build_fps6_actions

    pts3d = {}
    for s_ in SENSORS:
        q = np.stack(sensor_points(poses[s_], cal["gel"][s_], AXIS_LEN_MM), axis=1)  # (n,3,3) mocap mm
        pts3d[s_] = q[idx]

    out = {
        "frame_idx": torch.from_numpy(idx.astype(np.int32)),
        "fps": 30.0 / stride, "src_fps": 30.0, "fps_stride": stride,
        "num_frames": len(idx), "native_num_frames": n_native,
        "cameras": {},
        "points_mm": {s_: torch.from_numpy(pts3d[s_].astype(np.float32)) for s_ in SENSORS},
        "pose": {s_: torch.from_numpy(poses[s_][idx].astype(np.float32)) for s_ in SENSORS},
    }

    for v in VIEWS:
        view = cal["views"][v]
        intr = view["intr"]
        W, H = float(intr["width"]), float(intr["height"])

        # (N, sensor, point, 3) with the last axis (u_px, v_px, z_mm).
        # sensor axis = SENSORS order = [left, right]; point axis = [pos, normal, up].
        uvz = np.zeros((len(idx), len(SENSORS), 3, 3), np.float32)
        valid = np.zeros((len(idx), len(SENSORS), 3), bool)
        for si, s_ in enumerate(SENSORS):
            for k in range(3):
                u_, v_, z_, ok_ = project(pts3d[s_][:, k, :], view)
                uvz[:, si, k, 0] = u_
                uvz[:, si, k, 1] = v_
                uvz[:, si, k, 2] = z_
                valid[:, si, k] = ok_

        uvz_norm = uvz.copy()
        uvz_norm[..., 0] /= W          # z is left in mm: there is no natural depth normalizer,
        uvz_norm[..., 1] /= H          # so the two tensors differ only in the u,v channels.

        # `view_` prefix: a bare "left" would collide with the *sensor* key "left" in
        # points_mm/pose, which mean different things. This matches the view_*.mp4 names.
        out[f"view_{v}"] = torch.from_numpy(uvz)
        out[f"view_{v}_norm"] = torch.from_numpy(uvz_norm)
        out[f"view_{v}_valid"] = torch.from_numpy(valid)
        out["cameras"][v] = {"intrinsics": dict(intr),
                             "T_mocap_to_cam": torch.from_numpy(view["T"].copy()),
                             "rmse_mm": view["rmse"],
                             "camera_serial": view["serial"]}

    out["meta"] = {
        "source": "arXiv 2604.06168 action-image points, projected per camera view",
        "view_tensor": "view_{left,middle,right}: (N, sensor, point, 3) = (u_px, v_px, z_mm); "
                       "view_*_norm has u/W, v/H and the SAME z in mm",
        "sensor_axis": list(SENSORS),
        "point_axis": list(POINT_ORDER),
        "valid": "view_*_valid (N, sensor, point) bool = z>0 AND inside the image",
        "axis_len_mm": AXIS_LEN_MM,
        "q_pos": "measured gel centre (T_gel_to_rigid.gel_center_in_rigid_mm), NOT the "
                 "tracked rigid-body origin",
        "q_normal": "measured gel axis (gel_axis_in_rigid) -- a deliberate departure from "
                    "the paper, which rotates a canonical -z because it has no gel calibration",
        "q_up": "rigid-body x, Gram-Schmidt-orthogonalized against the gel normal",
        "chain": "pose(m) x1000 -> +R@gel_center -> T_mocap_to_cam -> fx*X/Z+ppx",
        "task": task, "calibration": cal["meta"].get("calibration_id"),
        "world_frame_offset_applied": bool(apply_world_offset),
        "world_frame_offset_m": list(world_frame_offset(task, episode)),
        "units": "u,v px (or [0,1] in *_norm); z mm; points_mm mm; pose [xyz m, quat xyzw]",
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos-root", type=Path, default=VID)
    ap.add_argument("--episodes", nargs="*", default=None)
    ap.add_argument("--stride", type=int, default=5, help="5 -> 6 fps, matching the rest of the pipeline")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--apply-world-offset", action="store_true",
                    help="treat episodes.jsonl:world_frame_offset as a PENDING correction and "
                         "add it to the translation (default: off -- it is already applied)")
    ap.add_argument("--out-name", default=OUT_NAME,
                    help="output filename, so both hypotheses can coexist for comparison")
    a = ap.parse_args()

    eps = a.episodes or sorted(
        p.name for p in a.videos_root.iterdir()
        if p.is_dir() and (PARQUET_ROOT / f"{p.name}.parquet").is_file()
    )
    print(f"{len(eps)} episode(s), stride={a.stride} -> {30/a.stride:g} fps\n")

    n_written = 0
    for ep in eps:
        out = a.videos_root / ep / a.out_name
        if out.exists() and not a.overwrite:
            print(f"  [skip ] {ep}: {a.out_name} exists")
            continue
        d = build_episode(ep, a.stride, a.apply_world_offset)
        tmp = out.with_suffix(".pt.tmp")
        torch.save(d, tmp)
        tmp.replace(out)                                   # atomic; a killed run leaves no partial file
        n_written += 1
        frac = {v: float(d[f"view_{v}_valid"][:, :, 0].float().mean()) for v in VIEWS}
        print(f"  [write] {ep:<34} N={d['num_frames']:>5}  in-frame "
              + "  ".join(f"{v}={100*frac[v]:5.1f}%" for v in VIEWS)
              + f"  {out.stat().st_size/1e3:6.0f} kB")
    print(f"\ndone ({n_written} written)")


if __name__ == "__main__":
    main()
