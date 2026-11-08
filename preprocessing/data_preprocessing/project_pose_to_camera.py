#!/usr/bin/env python
"""Project the absolute sensor pose into the camera views.

Implements the geometry behind "action images" (arXiv 2604.06168, *Action Images:
End-to-End Policy Learning via Multiview Video Generation*): rather than feeding a policy a
low-dimensional action vector, the end-effector pose is projected into each camera view so the
action lives in the same 2D pixel space as the observation.

Three 3D points per sensor per timestep, following the paper:

    q_pos     the contact point itself
    q_normal  q_pos + l * (surface normal)
    q_up      q_pos + l * (an axis orthogonal to the normal)

with ``l = 100 mm`` (the paper's 0.1 m). The paper obtains its two axes by rotating the
canonical ``x`` and ``-z`` by the end-effector orientation, because it has no knowledge of the
gripper's geometry. **We deliberately depart from that**: this rig is calibrated
(``T_gel_to_rigid_*.json`` gives a measured ``gel_axis_in_rigid``), so the normal is the real
gel normal and ``q_pos`` is the real gel centre rather than the tracked rigid-body origin. That
is better grounded for this hardware, but it is not the paper's literal recipe.

Projection chain, exactly as ``calibration.json`` documents it:

    sensor_pose (metres, xyzw quat)
      -> x1000                          -> mocap mm
      -> + R_rigid @ gel_center_in_rigid_mm   -> gel centre, mocap mm
      -> T_mocap_to_cam                 -> camera mm
      -> fx*X/Z + ppx, fy*Y/Z + ppy     -> pixels

Verified over all 2406 frames of motherboard_0510_episode_004: 100% of points land in-frame
for all three views and both sensors, with Z > 0 throughout.

**``world_frame_offset`` is deliberately NOT applied here.** ``<task>/episodes.jsonl`` records
``world_frame_offset = [0.23, 0, 0.175]`` m for the five 2026-05-19 episodes (zero for 05-10,
05-11 and pushT), and nothing in this repo reads it — which looks exactly like a missed step.
It is not. The offset is already baked into the released poses; the field is provenance, not a
pending correction. Evidence, reproducible via ``vm_diffusion/scripts/verify_view_projection.py``:

  * The projected gel centre lands in that frame's top-10% optical-flow cells (AllTracker,
    24x32) on 69-90% of moving 05-19 frames, against a 10% chance level, and against 56-65%
    for the 05-10/05-11 episodes whose offset is zero. Adding the offset drops 05-19 to
    24-35%; subtracting it drops every date to 0%.
  * Overlaid on the middle-view mp4, the as-is projection sits on the operator's contact
    point in 05-19 episodes; both shifted variants sit on empty table.
  * Mean sensor position is 0.33 m (05-19) vs 0.29 m (05-10) in x — the same workspace.
    A pending 0.23 m offset would put them 0.23 m apart, and neither correction closes it.
  * ``mode1_v2/*.parquet`` poses are bit-identical to ``data/<task>/meta/<date>/*.parquet``,
    so nothing was silently re-framed during export either.

``load_episode_poses`` raises on any NEW date with a nonzero offset, because a genuinely
unapplied offset would project 23 cm into empty table and nothing downstream would notice.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.actions import quat_to_rotmat  # noqa: E402  (xyzw, renormalizes)

# release layout: preprocessing/ holds data/<task>/calibration; parquets via $ACTS_PARQUET_ROOT
ROOT = Path(os.environ.get("ACTS_PREPROC_ROOT", Path(__file__).resolve().parents[1]))
CAL_ROOT = ROOT / "data"
PARQUET_ROOT = Path(os.environ.get("ACTS_PARQUET_ROOT", ROOT / "mode1_v2"))
VIEWS = ("left", "middle", "right")
SENSORS = ("left", "right")
AXIS_LEN_MM = 100.0          # the paper's l = 0.1 m
# (date, offset) pairs whose NON-ZERO world_frame_offset has been verified to be ALREADY
# APPLIED in the released poses -- see the module docstring and
# vm_diffusion/scripts/verify_view_projection.py. Anything else nonzero raises: an offset
# that really were pending would shift the projection 23 cm into empty table, and no
# downstream stage checks that.
VERIFIED_PREAPPLIED_OFFSETS = {"05-19": (0.23, 0.0, 0.175)}
_OFFSET_CACHE: dict[str, dict[str, tuple]] = {}


def episode_world_offsets(task: str) -> dict[str, tuple]:
    """-> {'MM-DD': (x, y, z)} from <task>/episodes.jsonl, or {} if the manifest is absent."""
    if task in _OFFSET_CACHE:
        return _OFFSET_CACHE[task]
    p = CAL_ROOT / task / "episodes.jsonl"
    out: dict[str, tuple] = {}
    if p.is_file():
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            off = r.get("world_frame_offset")
            if off is not None and r.get("date"):
                out[str(r["date"])[-5:]] = tuple(float(x) for x in off)
    _OFFSET_CACHE[task] = out
    return out


def world_frame_offset(task: str, episode: str) -> tuple:
    """-> the (x, y, z) metres recorded for this episode's date, or zeros."""
    m = re.search(r"_(\d{2})(\d{2})_", episode)
    mmdd = f"{m.group(1)}-{m.group(2)}" if m else ""
    return episode_world_offsets(task).get(mmdd, (0.0, 0.0, 0.0))


def check_world_frame_offset(task: str, episode: str) -> tuple:
    """Raise if this episode's date carries an unverified nonzero world_frame_offset."""
    m = re.search(r"_(\d{2})(\d{2})_", episode)
    mmdd = f"{m.group(1)}-{m.group(2)}" if m else ""
    off = episode_world_offsets(task).get(mmdd, (0.0, 0.0, 0.0))
    if any(abs(x) > 1e-9 for x in off) and \
            not np.allclose(off, VERIFIED_PREAPPLIED_OFFSETS.get(mmdd, (np.nan,) * 3)):
        raise ValueError(
            f"{episode}: {task}/episodes.jsonl records world_frame_offset={off} for {mmdd}, "
            f"which is not in VERIFIED_PREAPPLIED_OFFSETS. On 05-19 that field turned out to "
            f"be provenance (already applied in the released poses), but that was ESTABLISHED, "
            f"not assumed — run vm_diffusion/scripts/verify_view_projection.py on this date "
            f"before adding it, or the projection may be off by {np.linalg.norm(off)*100:.0f} cm.")
    return off


def task_for_episode(episode: str) -> str:
    """Episode name -> calibration task dir, resolved via calibration.json:applies_to_dates.

    The date is embedded in the episode name (motherboard_0510_..., pusht_0618_...). Matching on
    the recorded dates rather than the task prefix means a future recalibration that declares new
    dates is picked up without editing this file.
    """
    m = re.search(r"_(\d{2})(\d{2})_", episode)
    if not m:
        raise ValueError(f"cannot parse a MMDD date out of episode name {episode!r}")
    mmdd = f"{m.group(1)}-{m.group(2)}"
    for task_dir in sorted(p for p in CAL_ROOT.iterdir() if (p / "calibration" / "calibration.json").is_file()):
        meta = json.loads((task_dir / "calibration" / "calibration.json").read_text())
        if any(d.endswith(mmdd) for d in meta.get("applies_to_dates", [])):
            return task_dir.name
    raise FileNotFoundError(
        f"no calibration covers date {mmdd} (from {episode!r}); "
        f"checked {[p.name for p in CAL_ROOT.iterdir() if p.is_dir()]}"
    )


def load_calibration(task: str) -> dict:
    """-> {'views': {view: {T, intr}}, 'gel': {sensor: {...}}, 'meta': calibration.json}."""
    d = CAL_ROOT / task / "calibration"
    out = {"views": {}, "gel": {}, "meta": json.loads((d / "calibration.json").read_text())}
    for v in VIEWS:
        c = json.loads((d / f"T_mocap_to_cam_{v}.json").read_text())
        T = np.asarray(c["T_mocap_to_cam"], dtype=np.float64)
        R = T[:3, :3]
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-5) or abs(np.linalg.det(R) - 1) > 1e-5:
            raise ValueError(f"{task}/{v}: T_mocap_to_cam rotation is not a proper rotation")
        out["views"][v] = {"T": T, "intr": c["intrinsics"], "rmse": c.get("rmse_mm"),
                           "serial": c.get("camera_serial")}
    for s in SENSORS:
        g = json.loads((d / f"T_gel_to_rigid_{s}.json").read_text())
        n = np.asarray(g["gel_axis_in_rigid"], dtype=np.float64)
        out["gel"][s] = {"center_mm": np.asarray(g["gel_center_in_rigid_mm"], dtype=np.float64),
                         "axis": n / np.linalg.norm(n),
                         "scale": float(g.get("mocap_scale", 1000.0))}
    return out


def sensor_points(pose: np.ndarray, gel: dict, axis_len_mm: float = AXIS_LEN_MM):
    """(T,7) pose [xyz m, quat xyzw] + gel calib -> (q_pos, q_normal, q_up), each (T,3) mocap mm.

    ``q_up`` uses the rigid body's x-axis Gram-Schmidt-orthogonalized against the gel normal, so
    the triad is well defined even when the two are nearly parallel (a raw cross product would be
    numerically unstable there).
    """
    pose = np.asarray(pose, dtype=np.float64)
    R = quat_to_rotmat(torch.from_numpy(pose[:, 3:])).numpy()        # (T,3,3), xyzw
    p_mm = pose[:, :3] * gel["scale"]                                # metres -> mocap mm

    q_pos = p_mm + np.einsum("nij,j->ni", R, gel["center_mm"])       # gel centre
    n_hat = np.einsum("nij,j->ni", R, gel["axis"])                   # measured gel normal
    n_hat /= np.linalg.norm(n_hat, axis=1, keepdims=True)

    x_hat = R[:, :, 0]                                               # rigid-body x
    u_hat = x_hat - (x_hat * n_hat).sum(1, keepdims=True) * n_hat    # orthogonalize vs normal
    bad = np.linalg.norm(u_hat, axis=1) < 1e-6                       # x nearly || normal
    if bad.any():                                                    # fall back to rigid-body y
        y_hat = R[bad, :, 1]
        u_hat[bad] = y_hat - (y_hat * n_hat[bad]).sum(1, keepdims=True) * n_hat[bad]
    u_hat /= np.linalg.norm(u_hat, axis=1, keepdims=True)

    return q_pos, q_pos + axis_len_mm * n_hat, q_pos + axis_len_mm * u_hat


def project(points_mm: np.ndarray, view: dict):
    """(N,3) mocap mm -> (u, v, z, valid). `valid` = in front of the camera AND inside the image."""
    T, intr = view["T"], view["intr"]
    Xc = points_mm @ T[:3, :3].T + T[:3, 3]                          # -> camera mm
    z = Xc[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)                       # never divide by ~0
    u = intr["fx"] * Xc[:, 0] / safe + intr["ppx"]
    v = intr["fy"] * Xc[:, 1] / safe + intr["ppy"]
    valid = (z > 0) & (u >= 0) & (u < intr["width"]) & (v >= 0) & (v < intr["height"])
    return u, v, z, valid


def load_episode_poses(episode: str, apply_world_offset: bool = False):
    """-> {'left': (T,7), 'right': (T,7)} absolute poses from the mode1_v2 parquet.

    ``apply_world_offset`` adds ``episodes.jsonl:world_frame_offset`` to the translation, i.e.
    treats that field as a PENDING correction. It defaults to False because every measurement
    on this copy of the data says the offset is already applied (see the module docstring);
    the flag exists so the two hypotheses can be built side by side and compared rather than
    argued about. Rotation is untouched either way -- a world translation does not rotate the
    body -- and the tactile action is unaffected in both cases, since dp = R_t^T (p_{t+1}-p_t)
    cancels any constant translation.
    """
    import pyarrow.parquet as pq
    task = task_for_episode(episode)
    if apply_world_offset:
        off = np.asarray(world_frame_offset(task, episode), dtype=np.float64)
    else:
        check_world_frame_offset(task, episode)
        off = np.zeros(3)
    f = PARQUET_ROOT / f"{episode}.parquet"
    t = pq.read_table(str(f), columns=[f"sensor_{s}_pose" for s in SENSORS])
    out = {s: np.asarray(t.column(f"sensor_{s}_pose").to_pylist(), dtype=np.float64)
           for s in SENSORS}
    if off.any():
        for s in SENSORS:
            out[s] = out[s].copy()
            out[s][:, :3] += off                     # translation only; quaternion unchanged
    return out


def project_episode(episode: str, axis_len_mm: float = AXIS_LEN_MM) -> dict:
    """Everything a renderer needs: per view, per sensor, the three projected points."""
    cal = load_calibration(task_for_episode(episode))
    poses = load_episode_poses(episode)
    out = {"episode": episode, "calibration": cal["meta"].get("calibration_id"), "views": {}}
    for v in VIEWS:
        out["views"][v] = {}
        for s in SENSORS:
            pts = sensor_points(poses[s], cal["gel"][s], axis_len_mm)
            proj = {}
            for name, q in zip(("pos", "normal", "up"), pts):
                u, vv, z, ok = project(q, cal["views"][v])
                proj[name] = {"u": u, "v": vv, "z": z, "valid": ok}
            out["views"][v][s] = proj
        out["views"][v]["intr"] = cal["views"][v]["intr"]
        out["views"][v]["rmse_mm"] = cal["views"][v]["rmse"]
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", default="motherboard_0510_episode_004")
    a = ap.parse_args()

    r = project_episode(a.episode)
    print(f"{a.episode}  calibration={r['calibration']}")
    print(f"{'view':<8}{'sensor':<7}{'in-frame':>10}{'u range':>20}{'v range':>20}{'depth mm':>16}")
    for v in VIEWS:
        for s in SENSORS:
            d = r["views"][v][s]["pos"]
            print(f"{v:<8}{s:<7}{d['valid'].mean()*100:>9.1f}%"
                  f"  [{d['u'].min():7.1f},{d['u'].max():7.1f}]"
                  f"  [{d['v'].min():7.1f},{d['v'].max():7.1f}]"
                  f"  [{d['z'].min():6.0f},{d['z'].max():6.0f}]")
