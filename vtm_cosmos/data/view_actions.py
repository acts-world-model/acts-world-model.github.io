"""Per-frame calibrated view points, distinct from sensor-frame delta+force actions.

Each hand has 9 coordinates: (pos, normal, up) x (u/W, v/H, z metres).
A 16-frame window uses all 16 rows at ticks 0..15, without differencing.
This reader only establishes units; train/sampler subsequently apply the separate
train-fitted view-action normalizer once, over all nine coordinates.
"""
from pathlib import Path
import torch
from vtm_cosmos.streams import VIEW_ACTION_STREAMS, WINDOW_FRAMES


class ViewActionFile:
    def __init__(self, path: str | Path, sensor_file, fps_stride: int, fps: float):
        self.path = Path(path)
        d = torch.load(self.path, map_location='cpu', weights_only=False)
        if not torch.equal(d['frame_idx'].long(), sensor_file.frame_idx.long()):
            raise ValueError(f'{path}: view/sensor frame_idx mismatch')
        if int(d['fps_stride']) != fps_stride or abs(float(d['fps']) - fps) > 1e-6:
            raise ValueError(f'{path}: view/latent sampling clock mismatch')
        v = d['view_middle_norm'].float().clone()
        if v.shape != (sensor_file.num_frames, 2, 3, 3):
            raise ValueError(f'{path}: expected (frames,2,3,3), got {v.shape}')
        v[..., 2] /= 1000.0  # *_norm already normalizes u/v, but depth is millimetres.
        self.actions = {s: v[:, side].reshape(-1, 9) for side, s in enumerate(VIEW_ACTION_STREAMS)}
        self.valid = {s: torch.isfinite(v[:, side]).all(dim=(-1, -2)) & (v[:, side, :, 2] > 0).all(-1)
                      for side, s in enumerate(VIEW_ACTION_STREAMS)}
        # Outside-image points remain known projected coordinates; visibility is not action validity.
        self.visible = d['view_middle_valid'].bool()

    def window(self, start: int):
        if start < 0 or start + WINDOW_FRAMES > next(iter(self.actions.values())).shape[0]:
            raise IndexError(f'{self.path}: incomplete view action window')
        sl = slice(start, start + WINDOW_FRAMES)
        return ({s: a[sl].clone() for s, a in self.actions.items()},
                {s: v[sl].clone() for s, v in self.valid.items()})
