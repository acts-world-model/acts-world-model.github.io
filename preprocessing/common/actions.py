"""Canonical sensor-frame action encoding.

One definition, shared by every consumer. This used to be duplicated in
``cfm/data/encode_episodes.py``, ``jepa/model/anchors.py`` and
``vm_diffusion/data/dataset.py``; all three computed the rotation in the sensor's
local frame but the translation in the *world* frame, and two of them disagreed
about the 6D rotation layout.

The action from frame t to t+1 is expressed entirely in the sensor frame at time t:

    dp   = R_t^T (p_{t+1} - p_t)          (3,)  metres
    dR   = R_t^T R_{t+1}                  (3,3)
    r6d  = Rot6D(dR)                      (6,)  first two columns, [c1; c2]
    a_t  = [dp, r6d]                      (9,)

Being body-frame matters because the tactile image is rigidly attached to the
sensor: a given slide across the gel must map to the same action vector no matter
how the hand is oriented in the world. Equivalently, the encoding is invariant to
any global transform of the whole pose trajectory — which is exactly what
``tests`` check, and what the old world-frame translation failed.

The 6D rotation follows Zhou et al. 2019 (the first two columns of the matrix,
concatenated), which is continuous in SO(3) and so friendlier to regression than
quaternions or Euler angles.
"""
from __future__ import annotations

import torch

# Rot6D of the identity rotation, in [c1; c2] layout. Subtracting this makes a
# near-stationary action sit near zero instead of on a large constant offset;
# consumers that standardize their conditioning want it, the stored artifact does not.
ROT6D_IDENTITY: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def identity_action(action_dim: int = 9) -> torch.Tensor:
    """The action of standing still, as an (action_dim,) vector.

    Zero translation plus the identity rotation, repeated per sensor. Useful as the
    default conditioning offset: near-stationary steps then sit near zero instead of
    on the constant 1.0s that Rot6D(I) puts in two of every nine dims.
    """
    if action_dim % 9:
        raise ValueError(f"action_dim must be a multiple of 9, got {action_dim}")
    one = torch.tensor((0.0, 0.0, 0.0) + ROT6D_IDENTITY)
    return one.repeat(action_dim // 9)


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) quaternion in scipy xyzw order -> (..., 3, 3) rotation matrix.

    The quaternion is renormalized first: the stored poses are only unit to
    float32 precision, and the drift compounds through the relative products.
    """
    if q.shape[-1] != 4:
        raise ValueError(f"expected (...,4) xyzw quaternion, got {tuple(q.shape)}")

    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    qx, qy, qz, qw = q.unbind(-1)

    r00 = 1 - 2 * (qy * qy + qz * qz)
    r01 = 2 * (qx * qy - qz * qw)
    r02 = 2 * (qx * qz + qy * qw)
    r10 = 2 * (qx * qy + qz * qw)
    r11 = 1 - 2 * (qx * qx + qz * qz)
    r12 = 2 * (qy * qz - qx * qw)
    r20 = 2 * (qx * qz - qy * qw)
    r21 = 2 * (qy * qz + qx * qw)
    r22 = 1 - 2 * (qx * qx + qy * qy)

    return torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)


def matrix_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 6), the first two columns as [c1x,c1y,c1z, c2x,c2y,c2z]."""
    cols = R[..., :, :2]                                    # (..., 3, 2)
    return cols.transpose(-1, -2).reshape(*R.shape[:-2], 6)


def rot6d_to_matrix(r6: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 3, 3), inverting matrix_to_rot6d by Gram-Schmidt.

    The third column is recovered as c1 x c2, so the result is orthonormal even if
    the input drifted (e.g. came out of a network rather than a real rotation).
    """
    if r6.shape[-1] != 6:
        raise ValueError(f"expected (...,6) rot6d, got {tuple(r6.shape)}")

    a1, a2 = r6[..., :3], r6[..., 3:]
    c1 = a1 / a1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    a2 = a2 - (c1 * a2).sum(-1, keepdim=True) * c1
    c2 = a2 / a2.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    c3 = torch.cross(c1, c2, dim=-1)

    return torch.stack([c1, c2, c3], dim=-1)                # columns


def sensor_action_9d(pose7: torch.Tensor) -> torch.Tensor:
    """(..., T, 7) poses [x,y,z, qx,qy,qz,qw] -> (..., T-1, 9) sensor-frame actions.

    Poses are world <- sensor. Output row t is the motion from frame t to t+1 as
    seen from the sensor frame at t: ``[dp(3), Rot6D(dR)(6)]``.
    """
    if pose7.shape[-1] != 7:
        raise ValueError(f"expected (...,T,7) poses, got {tuple(pose7.shape)}")
    if pose7.shape[-2] < 2:
        raise ValueError(f"need at least 2 frames to form an action, got {pose7.shape[-2]}")

    pos = pose7[..., :3]
    R = quat_to_rotmat(pose7[..., 3:])                      # (..., T, 3, 3)

    R_t = R[..., :-1, :, :]
    R_t1 = R[..., 1:, :, :]

    # Contract the *first* index of R_t, i.e. apply R_t^T to the world-frame delta.
    dp = torch.einsum("...ji,...j->...i", R_t, pos[..., 1:, :] - pos[..., :-1, :])
    dR = R_t.transpose(-1, -2) @ R_t1

    return torch.cat([dp, matrix_to_rot6d(dR)], dim=-1)


def pose_action_delta(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Left+right pose streams -> (..., T-1, 18), the two 9D actions concatenated."""
    if left.shape != right.shape:
        raise ValueError(f"pose shape mismatch: left={tuple(left.shape)} right={tuple(right.shape)}")
    return torch.cat([sensor_action_9d(left), sensor_action_9d(right)], dim=-1)


# ── SE(3) composition ─────────────────────────────────────────────────────────
# Actions are stored per consecutive frame pair. Consumers that sample frames at a
# stride (fps-subsampled latents) need the action *across* the stride, which is the
# ordered product of the steps it spans, not the first step of the run.

def action9_to_matrix(a9: torch.Tensor) -> torch.Tensor:
    """(..., 9) -> (..., 4, 4) homogeneous, with T_{t+1} = T_t @ action."""
    if a9.shape[-1] != 9:
        raise ValueError(f"expected (...,9) action, got {tuple(a9.shape)}")

    T = torch.eye(4, dtype=a9.dtype, device=a9.device).expand(*a9.shape[:-1], 4, 4).clone()
    T[..., :3, :3] = rot6d_to_matrix(a9[..., 3:])
    T[..., :3, 3] = a9[..., :3]
    return T


def matrix_to_action9(T: torch.Tensor) -> torch.Tensor:
    """(..., 4, 4) -> (..., 9), inverse of action9_to_matrix."""
    return torch.cat([T[..., :3, 3], matrix_to_rot6d(T[..., :3, :3])], dim=-1)


def compose_action9(a9: torch.Tensor) -> torch.Tensor:
    """(M, 9) consecutive actions -> (9,) their ordered composition.

    An empty input yields the identity action (no motion), which is what a clamped
    boundary window should see.
    """
    if a9.ndim != 2 or a9.shape[-1] != 9:
        raise ValueError(f"expected (M,9) actions, got {tuple(a9.shape)}")

    if a9.shape[0] == 0:
        eye = torch.eye(4, dtype=a9.dtype, device=a9.device)
        return matrix_to_action9(eye)
    if a9.shape[0] == 1:
        return a9[0]

    T = action9_to_matrix(a9)
    out = T[0]
    for i in range(1, T.shape[0]):
        out = out @ T[i]
    return matrix_to_action9(out)


def window_action9(a9: torch.Tensor, frame_idx: torch.Tensor) -> torch.Tensor:
    """(N, 9) episode actions + (W,) frame indices -> (W-1, 9) per-window actions.

    ``frame_idx`` are indices into the *pose/frame* stream (so valid actions are
    ``a9[i]`` for i in ``[frame_idx[k], frame_idx[k+1])``). Repeated indices — which
    is how a clamped boundary window ends — give the identity action.
    """
    steps = []
    for k in range(len(frame_idx) - 1):
        i0, i1 = int(frame_idx[k]), int(frame_idx[k + 1])
        steps.append(compose_action9(a9[i0:max(i0, i1)]))
    return torch.stack(steps, dim=0)


__all__ = [
    "ROT6D_IDENTITY",
    "quat_to_rotmat",
    "matrix_to_rot6d",
    "rot6d_to_matrix",
    "identity_action",
    "sensor_action_9d",
    "pose_action_delta",
    "action9_to_matrix",
    "matrix_to_action9",
    "compose_action9",
    "window_action9",
]
