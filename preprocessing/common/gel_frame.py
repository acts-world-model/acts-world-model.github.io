"""The gel frame: the sensor frame re-expressed so +z is the real gel normal.

The OptiTrack rigid-body frame attached to each sensor is arbitrary — its axes come
from how the markers were placed, not from the gel. The dual-ball calibration in
``data/<task>/calibration/T_gel_to_rigid_<side>.json`` records where the gel actually
is relative to that rigid body:

    gel_axis_in_rigid       unit normal of the gel surface, in RIGID coordinates
    gel_center_in_rigid_mm  gel surface centre, in RIGID coordinates, millimetres

Both are far from the rigid axes: the normal is roughly -y (left
``(-0.174, -0.932, -0.316)``, right ``(+0.350, -0.925, +0.149)``) and the surface sits
66 mm (left) / 72 mm (right) from the rigid origin. So a "sensor-frame" action has a
``dp_z`` that is not press depth, a ``dp_x, dp_y`` that are not shear, and a rotation
whose lever arm is taken about the wrong point.

This module builds the rigid <- gel transform

    T_rigid_gel = [R_g | c_gel],   R_g = [e1 e2 gel_axis],   c_gel = gel_center / 1000

so that composing it onto the world poses BEFORE ``common.actions.sensor_action_9d``
yields a 9D action already in the gel frame:

    T_world_gel = T_world_rigid @ T_rigid_gel
    action_gel  = sensor_action_9d(pose7(T_world_gel))

Doing it at the pose level rather than rotating downstream rasters means every
consumer becomes correct for free — in particular ``build_inverse_action_flow.py``
puts its grid on the gel frame's ``z = 0`` plane, which is now the actual gel surface
in its actual location.

The in-plane axes e1, e2 are NOT pinned by the calibration (only the normal is). We
fix them deterministically from a reference vector so every rebuild agrees; the choice
is arbitrary and a learned encoder absorbs it, but it must be recorded so Stage 2 can
reproduce the identical basis.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]

# Both tasks were calibrated with the same rig and the files are byte-identical, but
# resolve per task anyway so a future re-calibration of one does not silently apply
# to the other.
GEL_CALIB_DIRS = {
    "motherboard": _REPO / "data" / "motherboard" / "calibration",
    "pusht": _REPO / "data" / "pushT" / "calibration",
}


def calib_dir_for(episode: str) -> Path:
    """``motherboard_0510_episode_000`` -> that task's calibration directory."""
    task = episode.split("_")[0]
    try:
        return GEL_CALIB_DIRS[task]
    except KeyError:
        raise KeyError(
            f"no gel calibration registered for task {task!r} (episode {episode!r}); "
            f"known tasks: {sorted(GEL_CALIB_DIRS)}"
        ) from None


def gel_frame(side: str, calib_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """-> (R_g (3,3), c_gel (3,) metres, info) for ``rigid <- gel``.

    ``R_g[:, 2]`` is ``gel_axis_in_rigid``: the press / normal axis. The returned
    matrix is orthonormal with det +1.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    path = Path(calib_dir) / f"T_gel_to_rigid_{side}.json"
    if not path.is_file():
        raise FileNotFoundError(f"gel calibration missing: {path}")
    c = json.loads(path.read_text())

    n = np.asarray(c["gel_axis_in_rigid"], dtype=np.float64)
    n = n / np.linalg.norm(n)
    c_gel = np.asarray(c["gel_center_in_rigid_mm"], dtype=np.float64) / 1000.0

    # Deterministic in-plane basis: project a fixed reference onto the tangent plane.
    # Swap the reference when it is near-parallel to n, or the cross product degenerates.
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(ref, n)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    R_g = np.stack([e1, e2, n], axis=1)          # columns

    if np.linalg.det(R_g) < 0:                   # keep it a rotation, never a reflection
        R_g[:, 1] *= -1.0

    info = {
        "gel_axis_in_rigid": n.tolist(),
        "gel_center_in_rigid_m": c_gel.tolist(),
        "R_rigid_from_gel": R_g.tolist(),
        "in_plane_reference": ref.tolist(),
        "in_plane_rule": "e1 = normalize(ref x n); e2 = n x e1; ref = +z unless |n_z| >= 0.9, then +x",
        "gel_distance_mm": float(c.get("gel_distance_mm", float("nan"))),
        "depth_offset_mm": float(c.get("depth_offset_mm", float("nan"))),
        "calib_file": str(path),
        "calib_method": c.get("method"),
    }
    return R_g, c_gel, info


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation -> (..., 4) quaternion, scipy xyzw order.

    The inverse of ``common.actions.quat_to_rotmat``. Written out rather than taken
    from scipy because scipy is not importable under ``PYTHONNOUSERSITE=1``, which is
    how the training scripts launch. Uses Shepperd's method: pick the branch whose
    denominator is largest, so the division never approaches 0.
    """
    R = np.asarray(R, dtype=np.float64)
    shape = R.shape[:-2]
    m = R.reshape(-1, 3, 3)
    n = m.shape[0]

    d = np.stack([
        1.0 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2],
        1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2],
        1.0 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2],
        1.0 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2],
    ], axis=1)
    k = d.argmax(axis=1)
    s = 2.0 * np.sqrt(np.maximum(d[np.arange(n), k], 1e-20))

    q = np.zeros((n, 4), dtype=np.float64)           # x, y, z, w
    for case in range(4):
        i = k == case
        if not i.any():
            continue
        si, mi = s[i], m[i]
        if case == 0:
            q[i, 3] = si / 4.0
            q[i, 0] = (mi[:, 2, 1] - mi[:, 1, 2]) / si
            q[i, 1] = (mi[:, 0, 2] - mi[:, 2, 0]) / si
            q[i, 2] = (mi[:, 1, 0] - mi[:, 0, 1]) / si
        elif case == 1:
            q[i, 0] = si / 4.0
            q[i, 3] = (mi[:, 2, 1] - mi[:, 1, 2]) / si
            q[i, 1] = (mi[:, 0, 1] + mi[:, 1, 0]) / si
            q[i, 2] = (mi[:, 0, 2] + mi[:, 2, 0]) / si
        elif case == 2:
            q[i, 1] = si / 4.0
            q[i, 3] = (mi[:, 0, 2] - mi[:, 2, 0]) / si
            q[i, 0] = (mi[:, 0, 1] + mi[:, 1, 0]) / si
            q[i, 2] = (mi[:, 1, 2] + mi[:, 2, 1]) / si
        else:
            q[i, 2] = si / 4.0
            q[i, 3] = (mi[:, 1, 0] - mi[:, 0, 1]) / si
            q[i, 0] = (mi[:, 0, 2] + mi[:, 2, 0]) / si
            q[i, 1] = (mi[:, 1, 2] + mi[:, 2, 1]) / si

    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.reshape(*shape, 4)


def poses_to_gel(pose7: torch.Tensor, R_g: np.ndarray, c_gel: np.ndarray) -> torch.Tensor:
    """(T, 7) world <- rigid poses -> (T, 7) world <- gel poses, same [xyz, qxyzw] layout.

    Quaternion sign is not made continuous: ``sensor_action_9d`` goes through
    ``quat_to_rotmat``, which is invariant to q -> -q.
    """
    from common.actions import quat_to_rotmat

    if pose7.ndim != 2 or pose7.shape[-1] != 7:
        raise ValueError(f"expected (T,7) poses, got {tuple(pose7.shape)}")

    p = pose7[:, :3].double().numpy()
    R = quat_to_rotmat(pose7[:, 3:].double()).numpy()  # (T,3,3) world <- rigid

    R_wg = R @ R_g                                     # world <- gel
    p_wg = p + np.einsum("tij,j->ti", R, c_gel)        # gel origin in world

    return torch.from_numpy(np.concatenate([p_wg, rotmat_to_quat(R_wg)], axis=1))


def gel_action_closed_form(a9_rigid: torch.Tensor, R_g: np.ndarray,
                           c_gel: np.ndarray) -> torch.Tensor:
    """Independent reference for the pose-composition path, used by the verify gate.

    ``A_gel = T_rg^-1 A_rigid T_rg`` expands to a conjugation of the rotation plus a
    lever-arm term on the translation:

        dR_gel = R_g^T dR R_g
        dp_gel = R_g^T ((dR - I) c_gel + dp)
    """
    from common.actions import matrix_to_rot6d, rot6d_to_matrix

    dp = a9_rigid[:, :3].double().numpy()
    dR = rot6d_to_matrix(a9_rigid[:, 3:].double()).numpy()

    dp_gel = np.einsum("ij,tj->ti", R_g.T,
                       np.einsum("tij,j->ti", dR - np.eye(3), c_gel) + dp)
    dR_gel = np.einsum("ij,tjk,kl->til", R_g.T, dR, R_g)

    r6 = matrix_to_rot6d(torch.from_numpy(dR_gel))
    return torch.cat([torch.from_numpy(dp_gel), r6], dim=1)
