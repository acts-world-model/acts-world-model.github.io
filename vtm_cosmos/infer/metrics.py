"""Evaluation metrics: future-only image PSNR/SSIM/LPIPS and action translation /
rotation-angle errors (spec §10.3)."""

from __future__ import annotations

import numpy as np
import torch

from vtm_cosmos.data.actions import rot6d_to_matrix, rotation_angle_deg


def to_uint8(frames: torch.Tensor) -> np.ndarray:
    """(3,T,H,W) in [-1,1] -> (T,H,W,3) uint8."""
    x = ((frames.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return x.permute(1, 2, 3, 0).cpu().numpy()


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float(10 * np.log10(255.0**2 / max(mse, 1e-10)))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    vals = [structural_similarity(a[i], b[i], channel_axis=-1, data_range=255) for i in range(a.shape[0])]
    return float(np.mean(vals))


class LPIPS:
    def __init__(self, device):
        import lpips

        self.net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        ta = torch.from_numpy(a).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(self.device)
        tb = torch.from_numpy(b).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(self.device)
        return float(self.net(ta, tb).mean())


def image_metrics(pred: torch.Tensor, gt: torch.Tensor, first_future_frame: int = 8, lpips_fn: LPIPS | None = None) -> dict:
    """pred/gt (3,16,H,W) in [-1,1]; metrics over frames [first_future_frame:] only."""
    a = to_uint8(pred[:, first_future_frame:])
    b = to_uint8(gt[:, first_future_frame:])
    out = {"psnr": psnr(a, b), "ssim": ssim(a, b)}
    if lpips_fn is not None:
        out["lpips"] = lpips_fn(a, b)
    return out


def action_metrics(pred9: torch.Tensor, gt9: torch.Tensor, valid: torch.Tensor | None = None) -> dict:
    """pred/gt (K,9) physical. Translation error in metres (L2 per step) and geodesic rotation
    angle in degrees, averaged over valid steps; rot6d Euclidean error reported separately."""
    if valid is None:
        valid = torch.ones(pred9.shape[0], dtype=torch.bool)
    p, g = pred9[valid].float(), gt9[valid].float()
    if p.numel() == 0:
        return {"trans_err_m": float("nan"), "rot_err_deg": float("nan"), "rot6d_l2": float("nan"), "n": 0}
    trans = (p[:, :3] - g[:, :3]).norm(dim=-1)
    ang = rotation_angle_deg(rot6d_to_matrix(p[:, 3:]), rot6d_to_matrix(g[:, 3:]))
    return {
        "trans_err_m": float(trans.mean()),
        "rot_err_deg": float(ang.mean()),
        "rot6d_l2": float((p[:, 3:] - g[:, 3:]).norm(dim=-1).mean()),
        "n": int(p.shape[0]),
    }
