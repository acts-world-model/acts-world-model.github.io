"""Video dataset for finetuning the Wan2.2 causal VAE on tactile clips.

Reads a `path,text` CSV (the `text` column is ignored — VAE reconstruction needs
no captions), decodes each mp4 with the ffmpeg CLI (robust in this env), and
returns a random `num_frames`-length clip as a [3, T, H, W] float tensor in
[-1, 1]. Frame counts are trimmed to the form 4k+1 required by the VAE's 4x
temporal compression.

With `fps > 0` the clip is subsampled to that rate before the random crop in time
(`frames[::stride]`), mirroring how the precomputed optical flow was extracted in
alltracker/batch_tactile_flow.py. With `return_flow=True` each item additionally
carries a normalized motion-magnitude map derived from that flow, for weighting
the reconstruction loss toward moving regions.
"""

import csv
import os
import random
import shutil
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

FLOW_SUFFIX = "_flow.npz"


def _find_ffmpeg():
    """$FFMPEG, then imageio-ffmpeg's bundled binary (pinned to 0.6.0 = ffmpeg 7.0.2, the build the
    released latents were encoded with), then PATH."""
    if os.environ.get("FFMPEG"):
        return os.environ["FFMPEG"]
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return shutil.which("ffmpeg") or "ffmpeg"


FFMPEG = _find_ffmpeg()


def flow_path_for(mp4_path, suffix=FLOW_SUFFIX):
    """Sibling flow file for a clip: strip `.mp4`, append `suffix`.

    Mirrors the generator's naming (alltracker/batch_tactile_flow.py), which writes
    `_flow.npz` (adjacent, frame_t->frame_{t+1}) and `_refflow.npz` (reference,
    frame_0->frame_t). Default keeps the adjacent suffix so existing callers are unchanged.
    """
    return os.path.splitext(mp4_path)[0] + suffix


def largest_4kp1(n):
    """Largest value of the form 4k+1 that is <= n (min 1)."""
    if n < 1:
        return 1
    return ((n - 1) // 4) * 4 + 1


def scale_crop_geom(src_h, src_w, out_h, out_w):
    """Geometry of `scale=W:H:force_original_aspect_ratio=increase,crop=W:H`.

    Returns (mid_h, mid_w, crop_y, crop_x). Applying this to the flow grid lands on
    the identical field of view as the ffmpeg filter applies to the video, because
    the flow grid (384x512) is the same 4:3 frame as the 640x480 source.
    For 384x512 -> 256x256 this is mid=(256, 341), crop=(0, 42).
    """
    factor = max(out_w / src_w, out_h / src_h)
    mid_h, mid_w = int(round(src_h * factor)), int(round(src_w * factor))
    return mid_h, mid_w, (mid_h - out_h) // 2, (mid_w - out_w) // 2


def read_video_ffmpeg(path, height, width, max_frames=None, duration=None, start=None):
    """Decode <path> to uint8 [T, H, W, 3], scaled + center-cropped to (H, W).

    `start`/`duration` (seconds, input-side `-ss`/`-t`) decode only a window. `max_frames`
    slices only after ffmpeg has decoded the entire file into memory, which is wasteful when
    the caller wants a short piece of a long clip. Both default None == original behaviour.

    Note for callers that pair frames with flow: the returned array is indexed from the
    START of the decoded window, not of the file.
    """
    vf = (f"scale={width}:{height}:force_original_aspect_ratio=increase,"
          f"crop={width}:{height}")
    cmd = [FFMPEG, "-v", "error"]
    if start is not None:
        cmd += ["-ss", str(start)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-i", path, "-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed for {path}: "
                           f"{proc.stderr.decode(errors='ignore')[:200]}")
    frame_bytes = height * width * 3
    n = len(proc.stdout) // frame_bytes
    if n == 0:
        raise RuntimeError(f"no frames decoded from {path}")
    if max_frames is not None:
        n = min(n, max_frames)
    arr = np.frombuffer(proc.stdout[: n * frame_bytes], dtype=np.uint8)
    return arr.reshape(n, height, width, 3).copy()  # writable (frombuffer is read-only)


def to_tensor(frames_uint8):
    """[T, H, W, 3] uint8 -> [3, T, H, W] float in [-1, 1]."""
    x = torch.from_numpy(np.ascontiguousarray(frames_uint8))
    x = x.permute(3, 0, 1, 2).contiguous().float()  # [C, T, H, W]
    return x.div_(255.0).mul_(2.0).sub_(1.0)


class TactileVideoDataset(Dataset):
    def __init__(self, csv_path, num_frames=17, height=256, width=256,
                 split="train", val_size=64, seed=0, max_retries=8,
                 fps=0.0, src_fps=30.0, return_flow=False, flow_floor=1.5,
                 flow_suffix=FLOW_SUFFIX, deterministic_window=False):
        assert height % 16 == 0 and width % 16 == 0, "H/W must be multiples of 16"
        self.num_frames = largest_4kp1(num_frames)
        self.height = height
        self.width = width
        self.max_retries = max_retries
        self.return_flow = return_flow
        self.flow_floor = flow_floor
        # True => center temporal crop (reproducible eval). False => random crop (train aug).
        self.deterministic_window = deterministic_window
        # Which sibling flow file to weight by: "_flow.npz" (adjacent, default) or
        # "_refflow.npz" (reference frame_0->frame_t). Both share the same array layout.
        self.flow_suffix = flow_suffix
        # fps=0 keeps every native frame (the original behaviour). fps=6 with 30fps
        # sources gives stride 5 — the same stride the flow was extracted at, so each
        # kept frame lines up 1:1 with a flow frame.
        self.stride = max(1, int(round(src_fps / fps))) if fps > 0 else 1

        rows = []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)  # skip `path,text` header
            for r in reader:
                if not r:
                    continue
                p = r[0].strip()
                if p:
                    rows.append(p)

        existing = [p for p in rows if os.path.isfile(p)]
        n_skipped = len(rows) - len(existing)
        if n_skipped:
            print(f"[dataset] skipped {n_skipped}/{len(rows)} rows with missing files")

        # deterministic train/val split (val = last `val_size` after a fixed shuffle)
        rng = random.Random(seed)
        rng.shuffle(existing)
        if val_size > 0 and len(existing) > val_size:
            if split == "val":
                self.paths = existing[-val_size:]
            else:
                self.paths = existing[:-val_size]
        else:
            self.paths = existing
        print(f"[dataset] split={split}: {len(self.paths)} clips "
              f"(num_frames={self.num_frames}, {self.width}x{self.height}, "
              f"stride={self.stride}{f', {fps:g}fps' if fps > 0 else ', native fps'}"
              f"{f', +flow ({self.flow_suffix})' if return_flow else ''})")

    def __len__(self):
        return len(self.paths)

    def _load_clip(self, path):
        """-> ([3,T,H,W] in [-1,1], sel) where sel are the clip's native frame indices."""
        frames = read_video_ffmpeg(path, self.height, self.width)  # [T,H,W,3]
        # Subsample to the target fps BEFORE picking the window, mirroring the flow
        # generator's `frames[::stride]` so the kept frames are exactly the ones flow
        # was computed on.
        idx = np.arange(0, frames.shape[0], self.stride)
        t = len(idx)
        want = self.num_frames
        if t >= want:
            # deterministic_window pins the crop (center) so evaluation is reproducible:
            # otherwise this random.randint depends on per-worker RNG, and every adapter /
            # every run scores a different temporal window -> ~0.5 dB of spurious swing on
            # the moving-region split. Training keeps the random window (augmentation).
            start = (t - want) // 2 if self.deterministic_window \
                else random.randint(0, t - want)
            sel = idx[start:start + want]
        else:  # too short: loop-pad (== the old concatenate([frames]*reps)[:want])
            sel = idx[np.arange(want) % t]
        sel = sel[:largest_4kp1(len(sel))]
        return to_tensor(frames[sel]), sel

    def _flow_mhat(self, path, sel):
        """-> [1,T,H,W] normalized motion magnitude in [0,1]; zeros if flow is unusable.

        `sel` are native frame indices; flow[i] covers native [stride*i, stride*i+stride).
        Zeros mean a uniform weight of 1 downstream, so a missing npz degrades to the
        unweighted loss rather than failing.
        """
        zeros = torch.zeros(1, len(sel), self.height, self.width, dtype=torch.float32)
        try:
            # Context manager: NpzFile holds an fd, and persistent workers doing this
            # every __getitem__ would leak them to "Too many open files".
            with np.load(flow_path_for(path, self.flow_suffix)) as d:
                flow = d["flow"]  # [Tf,2,384,512] float16
                Tf, fh, fw = flow.shape[0], flow.shape[2], flow.shape[3]
                if Tf < 2:
                    return zeros
                fstride = int(d["stride"]) if "stride" in d.files else 5
                # The LAST flow frame has no successor and is stored all-zero; it IS
                # reachable from a clip at the end of the video -> clamp to Tf-2.
                fi = np.clip(np.asarray(sel) // fstride, 0, Tf - 2)
                uniq, inv = np.unique(fi, return_inverse=True)
                # fp32 before pow: max observed magnitude ~113px leaves fp16 little
                # headroom on a heavy tail.
                f = torch.from_numpy(flow[uniq].astype(np.float32))
        except Exception:
            return zeros

        mag = f.pow(2).sum(1).sqrt()  # [U,384,512], px on the flow grid
        # Resize the MAGNITUDE, not the flow: bilinear on opposing vectors cancels them
        # to zero. Magnitudes stay in flow-grid px — do NOT rescale to crop px, the
        # flow_floor below is defined in these units.
        mid_h, mid_w, cy, cx = scale_crop_geom(fh, fw, self.height, self.width)
        mag = F.interpolate(mag[:, None], size=(mid_h, mid_w),
                            mode="bilinear", align_corners=False)[:, 0]
        mag = mag[:, cy:cy + self.height, cx:cx + self.width]  # [U,H,W]
        mag = mag[torch.from_numpy(inv)]  # [T,H,W]
        # Floor the normalizer. p99 alone is scale-free, which also means a clip holding
        # nothing but sub-pixel flow noise gets that noise stretched to the same contrast
        # as real motion. The floor binds on ~23% of 17-frame clips at 6fps (and ~2/3 at
        # native fps, where the 0.57s window rarely catches a motion burst). Also makes
        # the p99==0 case safe. Units: px on the flow grid — see the no-rescale note above.
        den = torch.clamp(torch.quantile(mag.flatten(), 0.99), min=self.flow_floor)
        return (mag / den).clamp(0, 1)[None]  # [1,T,H,W]

    def __getitem__(self, idx):
        for attempt in range(self.max_retries):
            path = self.paths[idx]
            try:
                x, sel = self._load_clip(path)
            except Exception as e:  # bad/corrupt file -> try another sample
                if attempt == 0:
                    print(f"[dataset] read error on {path}: {e}; retrying")
                idx = random.randint(0, len(self.paths) - 1)
                continue
            if not self.return_flow:
                return x
            # Outside the retry: a missing npz has its own uniform-weight fallback, and
            # resampling here would hide a systematic flow problem behind the video
            # read-failure path.
            return x, self._flow_mhat(path, sel)
        raise RuntimeError("too many consecutive video read failures")
