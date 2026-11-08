"""Per-hand 9-D actions: file adapter, normalizer and SO(3)/rot6d helpers.

Action row k of a `sensor_actions_*.pt` file is the transition frame k -> k+1
(in the 15 FPS extraction grid), expressed in the sensor frame at time k:
    a_k = [dp(3), Rot6D(dR)(6)],  dp = R_k^T (p_{k+1} - p_k),  dR = R_k^T R_{k+1}
Rot6D = first two COLUMNS of dR flattened as [c1x,c1y,c1z, c2x,c2y,c2z]
(not identity-subtracted). This mirrors world_model/common/actions.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from vtm_cosmos.streams import ACTION_DIM, ACTION_FILE_KEYS, ACTION_STREAMS, LEGACY36_SLICES, N_ACTIONS, VIEW_ACTION_STREAMS


# ---------------------------------------------------------------------------
# rotation helpers (paired encoder/decoder, verified by round-trip tests)
# ---------------------------------------------------------------------------
def matrix_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 6): first two columns, column-major."""
    cols = R[..., :, :2]
    return cols.transpose(-1, -2).reshape(*R.shape[:-2], 6)


def rot6d_to_matrix(r6: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 3, 3) by Gram-Schmidt; third column = c1 x c2."""
    if r6.shape[-1] != 6:
        raise ValueError(f"expected (...,6), got {tuple(r6.shape)}")
    a1, a2 = r6[..., :3], r6[..., 3:]
    c1 = a1 / a1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    a2 = a2 - (c1 * a2).sum(-1, keepdim=True) * c1
    c2 = a2 / a2.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    c3 = torch.cross(c1, c2, dim=-1)
    return torch.stack([c1, c2, c3], dim=-1)


def rotation_angle_deg(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """Geodesic angle between rotations, degrees. (...,3,3) x (...,3,3) -> (...)."""
    rel = R1.transpose(-1, -2) @ R2
    tr = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = (tr - 1.0) / 2.0
    skew = torch.stack([rel[..., 2, 1] - rel[..., 1, 2], rel[..., 0, 2] - rel[..., 2, 0], rel[..., 1, 0] - rel[..., 0, 1]], -1)
    sin = skew.norm(dim=-1) / 2.0
    return torch.rad2deg(torch.atan2(sin, cos))  # atan2 is exact near 0 where acos is not


def action9_to_se3(a9: torch.Tensor) -> torch.Tensor:
    """(..., 9) -> (..., 4, 4) homogeneous delta transform."""
    R = rot6d_to_matrix(a9[..., 3:9])
    T = torch.zeros(*a9.shape[:-1], 4, 4, dtype=a9.dtype, device=a9.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = a9[..., :3]
    T[..., 3, 3] = 1.0
    return T


def se3_to_action9(T: torch.Tensor) -> torch.Tensor:
    return torch.cat([T[..., :3, 3], matrix_to_rot6d(T[..., :3, :3])], dim=-1)


def compose_actions(a9_seq: torch.Tensor) -> torch.Tensor:
    """Ordered SE(3) product of a sequence of deltas: (..., K, 9) -> (..., 9).

    dT = dT_1 @ dT_2 @ ... @ dT_K  (spec §4; never add translations / average rot6d)."""
    T = action9_to_se3(a9_seq)
    out = T[..., 0, :, :]
    for k in range(1, T.shape[-3]):
        out = out @ T[..., k, :, :]
    return se3_to_action9(out)


def integrate_poses(T0: torch.Tensor, a9_seq: torch.Tensor) -> torch.Tensor:
    """Absolute poses from a start pose and deltas: T_t = T_{t-1} @ dT_t. (4,4),(K,9) -> (K+1,4,4)."""
    poses = [T0]
    for T in action9_to_se3(a9_seq):
        poses.append(poses[-1] @ T)
    return torch.stack(poses)


IDENTITY_ACTION9 = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# file adapter
# ---------------------------------------------------------------------------
class HandActionFile:
    """Loads `sensor_actions_*.pt` (per-episode) and serves per-hand windows.

    Keys: action_left/right (N-1, 9) float32, valid_left/right (N-1,), frame_idx (N,), meta."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        d = torch.load(self.path, map_location="cpu", weights_only=False)
        self.meta = d.get("meta", {})
        self.frame_idx = d["frame_idx"]
        self.actions = {}
        self.valid = {}
        for stream, (akey, vkey) in ACTION_FILE_KEYS.items():
            a = d[akey].float()
            if a.ndim != 2 or a.shape[1] != ACTION_DIM:
                raise ValueError(f"{self.path}: {akey} has shape {tuple(a.shape)}, expected (N-1, 9)")
            self.actions[stream] = a
            v = d.get(vkey)
            self.valid[stream] = v.bool() if v is not None else torch.ones(a.shape[0], dtype=torch.bool)
        n = self.actions["action_left"].shape[0]
        if self.actions["action_right"].shape[0] != n:
            raise ValueError(f"{self.path}: left/right row counts differ")
        if self.frame_idx.numel() != n + 1:
            raise ValueError(f"{self.path}: frame_idx has {self.frame_idx.numel()} entries, expected {n + 1}")
        self.num_transitions = n

    @property
    def num_frames(self) -> int:
        return self.num_transitions + 1

    def window(self, t_start: int, n: int = N_ACTIONS) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Rows t_start .. t_start+n-1 = transitions a_1..a_n of the window starting at frame t_start."""
        if t_start < 0 or t_start + n > self.num_transitions:
            raise IndexError(f"{self.path}: window [{t_start},{t_start + n}) exceeds {self.num_transitions} rows")
        acts = {s: self.actions[s][t_start : t_start + n].clone() for s in ACTION_STREAMS}
        valid = {s: self.valid[s][t_start : t_start + n].clone() for s in ACTION_STREAMS}
        return acts, valid


def split_legacy36(action36: torch.Tensor) -> dict[str, torch.Tensor]:
    """(..., 36) view+tactile rows from vm_diffusion -> per-hand 9-D dict (adapter only)."""
    if action36.shape[-1] != 36:
        raise ValueError(f"expected (...,36), got {tuple(action36.shape)}")
    return {s: action36[..., a:b] for s, (a, b) in LEGACY36_SLICES.items()}


# ---------------------------------------------------------------------------
# normalizer
# ---------------------------------------------------------------------------
class ActionNormalizer:
    """Per-channel normalization of the 9-D action, shared by both hands.

    mode "meanstd": (x - mean) / std.
    mode "quantile": 2 * (x - q01) / (q99 - q01) - 1  -> roughly [-1, 1]; the same map the Cosmos3
    action datasets use (``cosmos_framework/.../action_normalization.py``), so a head initialized
    from a Nano embodiment domain sees inputs on the scale it was trained with."""

    def __init__(self, a: torch.Tensor, b: torch.Tensor, mode: str = "meanstd", std_floor: float = 1e-6, meta: dict | None = None):
        if mode not in ("meanstd", "quantile"):
            raise ValueError(mode)
        self.mode = mode
        a = a.float().reshape(ACTION_DIM)
        b = b.float().reshape(ACTION_DIM)
        if mode == "meanstd":
            self.mean, self.std = a, b.clamp_min(std_floor)
            self.q01 = self.q99 = None
        else:
            self.q01, self.q99 = a, b
            self.q99 = torch.where(self.q99 - self.q01 < std_floor, self.q01 + std_floor, self.q99)
            self.mean = (self.q01 + self.q99) / 2.0
            self.std = (self.q99 - self.q01) / 2.0
        self.meta = dict(meta or {})

    @classmethod
    def identity(cls) -> "ActionNormalizer":
        return cls(torch.zeros(ACTION_DIM), torch.ones(ACTION_DIM), meta={"kind": "identity"})

    @classmethod
    def fit(cls, rows: torch.Tensor, std_floor: float = 1e-6, meta: dict | None = None, mode: str = "meanstd") -> "ActionNormalizer":
        """rows: (M, 9) physical actions (both hands stacked)."""
        rows = rows.float()
        if mode == "meanstd":
            return cls(rows.mean(0), rows.std(0, unbiased=False), mode=mode, std_floor=std_floor, meta=meta)
        q = torch.quantile(rows, torch.tensor([0.01, 0.99]), dim=0)
        return cls(q[0], q[1], mode=mode, std_floor=std_floor, meta=meta)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x)) / self.std.to(x)  # identical algebra for both modes

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x) + self.mean.to(x)

    def state_dict(self) -> dict:
        d = {"mode": self.mode, "mean": self.mean.tolist(), "std": self.std.tolist(), "meta": self.meta}
        if self.mode == "quantile":
            d["q01"], d["q99"] = self.q01.tolist(), self.q99.tolist()
        return d

    @classmethod
    def from_state_dict(cls, d: dict) -> "ActionNormalizer":
        mode = d.get("mode", "meanstd")
        if mode == "quantile":
            return cls(torch.tensor(d["q01"]), torch.tensor(d["q99"]), mode=mode, meta=d.get("meta"))
        return cls(torch.tensor(d["mean"]), torch.tensor(d["std"]), mode=mode, meta=d.get("meta"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.state_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ActionNormalizer":
        return cls.from_state_dict(json.loads(Path(path).read_text()))


class ActionNormalizers:
    """Sensor-action normalizer (``action_left/right``: delta pose + force, 9-D) plus, for the
    ``action_full`` task, a separate view-action normalizer (``view_action_left/right``: per-frame
    calibrated points (pos, normal, up) x (u/W, v/H, z m), 9-D). Both hands share one table each.
    Every action stream fed to the Generator goes through ``normalize_dict``."""

    def __init__(self, sensor: ActionNormalizer, view: ActionNormalizer | None = None):
        self.sensor = sensor
        self.view = view

    @classmethod
    def wrap(cls, obj) -> "ActionNormalizers":
        return obj if isinstance(obj, ActionNormalizers) else cls(obj)

    @property
    def mode(self) -> str:
        return self.sensor.mode

    def for_stream(self, stream: str) -> ActionNormalizer:
        if stream in ACTION_STREAMS:
            return self.sensor
        if stream in VIEW_ACTION_STREAMS:
            if self.view is None:
                raise ValueError(f"{stream}: no view-action normalizer fitted (action_full needs one)")
            return self.view
        raise KeyError(stream)

    def normalize(self, x: torch.Tensor, stream: str = "action_left") -> torch.Tensor:
        return self.for_stream(stream).normalize(x)

    def denormalize(self, x: torch.Tensor, stream: str = "action_left") -> torch.Tensor:
        return self.for_stream(stream).denormalize(x)

    def normalize_dict(self, actions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {s: self.for_stream(s).normalize(x) for s, x in actions.items()}

    def state_dict(self) -> dict:
        return {"format": 2, "sensor": self.sensor.state_dict(), "view": None if self.view is None else self.view.state_dict()}

    @classmethod
    def from_state_dict(cls, d: dict) -> "ActionNormalizers":
        if "sensor" not in d:  # legacy flat ActionNormalizer dict (runs before 2026-09-17)
            return cls(ActionNormalizer.from_state_dict(d), None)
        view = d.get("view")
        return cls(ActionNormalizer.from_state_dict(d["sensor"]), None if view is None else ActionNormalizer.from_state_dict(view))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.state_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ActionNormalizers":
        return cls.from_state_dict(json.loads(Path(path).read_text()))
