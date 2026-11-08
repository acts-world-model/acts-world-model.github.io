"""Window dataset over the existing Wan2.2 latent trees + per-hand action files.

Reads (never writes) the caches described in the spec §3:
  <latent_root>/visual/<ep>.pt : z_visual_middle (Nw,48,4,16,16) fp16 + meta(starts, fps, fps_stride, ...)
  <latent_root>/tac/<ep>.pt    : z_tactile_left / z_tactile_right
  <latent_root>/wrist/<ep>.pt  : z_wrist_left / z_wrist_right
  <action_root>/<ep stem>/<action_file> : action_left/right (N-1,9), valid_*, frame_idx (N,)
Each window = 16 extraction-fps frames -> 4 clean latents (history 0,1 / future 2,3)
and 15 transitions a_1..a_15 (history a_1..a_7 / future a_8..a_15).
Outputs the logical batch schema of spec §3.2 (physical actions, normalizer applied in the model).
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from vtm_cosmos.data.actions import HandActionFile
from vtm_cosmos.streams import (
    ACTION_STREAMS, ALL_ACTION_STREAMS, ALL_STREAMS, VIEW_ACTION_STREAMS,
    ACTION_TICKS,
    EXTRACTION_FPS,
    IMAGE_STREAMS,
    LATENT_CHANNELS,
    LATENT_FILE_KEYS,
    LATENT_HW,
    LATENT_T,
    N_ACTIONS,
    NUM_STREAMS,
    OPTIONAL_STREAMS,
    STREAM_ID,
    STREAMS,
    WINDOW_FRAMES,
    WRIST_STREAMS,
    active_pattern,
    streams_from_pattern,
)

SUBDIRS = ("visual", "tac", "wrist")


def _load_pt(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


class VtmWindowDataset(Dataset):
    def __init__(
        self,
        latent_root: str | Path,
        action_root: str | Path,
        episodes: list[str],
        action_file: str = "sensor_actions_with_dforce_fps15.pt",
        window_stride: int = 1,
        max_windows_per_episode: int | None = None,
        cache_episodes: int = 4,
        require_wrist: bool = True,
        optional_streams: tuple[str, ...] = (),
        include_view_actions: bool = False,
        view_action_file: str = "view_actions_fps15.pt",
        require_valid_actions: bool = False,
        active_streams: tuple[str, ...] | None = None,
    ):
        """``active_streams`` (model.active_streams): streams the run models; every other stream is
        reported absent (``stream_present=False``) while its files are still read and validated, so
        window validity and action statistics are identical to the full-stream run.
        ``optional_streams`` (spec-MM): streams that may be absent from a recording. An absent
        optional stream is reported as ``stream_present=False`` and its latent key is simply not
        emitted (never a zero tensor). ``require_wrist=False`` is shorthand for optional wrists."""
        self.latent_root = Path(latent_root)
        self.action_root = Path(action_root)
        self.action_file = action_file
        self.include_view_actions = include_view_actions
        self.view_action_file = view_action_file
        self._view_actions = {}
        self.episodes = list(episodes)
        self.window_stride = int(window_stride)
        self.max_windows_per_episode = max_windows_per_episode
        self.cache_size = int(cache_episodes)
        opt = set(optional_streams) | (set() if require_wrist else set(WRIST_STREAMS))
        bad = opt - set(OPTIONAL_STREAMS)
        if bad:
            raise ValueError(f"only {OPTIONAL_STREAMS} may be optional, got {sorted(bad)}")
        self.optional_streams = tuple(s for s in STREAMS if s in opt)
        self.require_wrist = not any(s in opt for s in WRIST_STREAMS)
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._actions: dict[str, HandActionFile] = {}

        self.require_valid_actions = bool(require_valid_actions)
        self.active_streams = None if active_streams is None else tuple(active_streams)
        if self.active_streams is not None and "video" not in self.active_streams:
            raise ValueError("active_streams must include 'video'")
        self.dropped_invalid_windows = 0
        self.index: list[tuple[int, int]] = []  # (episode idx, window idx into starts)
        self.episode_info: list[dict] = []
        for ei, ep in enumerate(self.episodes):
            info = self._scan_episode(ep)
            self.episode_info.append(info)
            n = info["num_windows"]
            wins = list(range(0, n, self.window_stride))
            if self.require_valid_actions:
                ok = self._window_valid_mask(ep, info["starts"])
                kept = [w for w in wins if ok[w]]
                self.dropped_invalid_windows += len(wins) - len(kept)
                wins = kept
            if self.max_windows_per_episode is not None:
                wins = wins[: self.max_windows_per_episode]
            self.index.extend((ei, w) for w in wins)

    def _window_valid_mask(self, ep: str, starts: torch.Tensor) -> torch.Tensor:
        """Per window: every sensor transition (and, with view actions, every view frame) is valid."""
        starts = starts.long()
        act = self._action_file(ep)
        flags = [(v, N_ACTIONS) for v in act.valid.values()]
        if self.include_view_actions:
            flags += [(v, WINDOW_FRAMES) for v in self._view_actions[ep].valid.values()]
        ok = torch.ones(starts.numel(), dtype=torch.bool)
        for v, n in flags:
            bad = torch.cat([torch.zeros(1, dtype=torch.long), (~v).long().cumsum(0)])
            ok &= (bad[starts + n] - bad[starts]) == 0
        return ok

    # ------------------------------------------------------------------ scan
    def _paths(self, ep: str) -> dict[str, Path]:
        return {sd: self.latent_root / sd / ep for sd in SUBDIRS}

    def _scan_episode(self, ep: str) -> dict:
        paths = self._paths(ep)
        has_wrist = paths["wrist"].exists()
        for sd, p in paths.items():
            if sd == "wrist" and not self.require_wrist:
                continue
            if not p.exists():
                raise FileNotFoundError(f"{ep}: missing {sd} latent file {p}")
        tac = _load_pt(paths["tac"])
        vis = _load_pt(paths["visual"])
        starts = tac["starts"]
        if not torch.equal(starts, vis["starts"]):
            raise ValueError(f"{ep}: visual/tac starts differ")
        if int(tac.get("window_size", WINDOW_FRAMES)) != WINDOW_FRAMES:
            raise ValueError(f"{ep}: window_size {tac.get('window_size')} != {WINDOW_FRAMES}")
        shape = tuple(tac["z_tactile_left"].shape)
        if shape[1:] != (LATENT_CHANNELS, LATENT_T, LATENT_HW, LATENT_HW):
            raise ValueError(f"{ep}: latent shape {shape}, expected (Nw,48,4,16,16)")
        if tuple(vis[LATENT_FILE_KEYS["video"][1]].shape) != shape:
            raise ValueError(f"{ep}: visual latent shape differs from tactile")
        if has_wrist:  # present wrist files must be valid even when optional (spec-MM §3)
            wr = _load_pt(paths["wrist"])
            if not torch.equal(starts, wr["starts"]):
                raise ValueError(f"{ep}: wrist starts differ")
            if tuple(wr["z_wrist_left"].shape) != shape:
                raise ValueError(f"{ep}: wrist latent shape differs from tactile")
        fps = float(tac.get("fps", EXTRACTION_FPS))
        if abs(fps - EXTRACTION_FPS) > 1e-6:
            raise ValueError(f"{ep}: latent tree fps {fps} != extraction fps {EXTRACTION_FPS}")
        act = self._action_file(ep)
        if self.include_view_actions:
            from vtm_cosmos.data.view_actions import ViewActionFile
            # pure-pose 9D (meta.force_informed falsy) or delta-force 9D; absolute force is a per-frame state
            if act.meta.get("force_informed") and act.meta.get("force_term") != "delta":
                raise ValueError(f"{ep}: force-informed sensor action must use force_term=delta")
            if not torch.equal(act.frame_idx.long(), torch.arange(act.num_frames) * int(tac["fps_stride"])):
                raise ValueError(f"{ep}: sensor frame IDs do not match latent sampling grid")
            self._view_actions[ep] = ViewActionFile(self.action_root / Path(ep).stem / self.view_action_file,
                                                   act, int(tac["fps_stride"]), fps)
        num_frames = int(tac.get("num_frames", starts.max().item() + WINDOW_FRAMES))
        if act.num_frames != num_frames:
            raise ValueError(f"{ep}: action file has {act.num_frames} frames, latent tree has {num_frames}")
        # every window must have its 15 transitions available
        max_start = int(starts.max().item())
        if max_start + N_ACTIONS > act.num_transitions:
            raise ValueError(f"{ep}: last window start {max_start} exceeds action rows {act.num_transitions}")
        present = [True] * (len(ALL_STREAMS) if self.include_view_actions else NUM_STREAMS)
        for st in WRIST_STREAMS:
            present[STREAM_ID[st]] = has_wrist
        if self.active_streams is not None:
            for st, on in list(zip(ALL_STREAMS, present)):
                present[STREAM_ID[st]] = on and st in self.active_streams
        return {
            "episode": ep,
            "present": tuple(present),
            "has_wrist": has_wrist,
            "num_windows": int(starts.numel()),
            "num_frames": num_frames,
            "fps_stride": int(tac.get("fps_stride", 1)),
            "source": str(tac.get("source", "")),
            "starts": starts.clone(),
        }

    def _action_file(self, ep: str) -> HandActionFile:
        if ep not in self._actions:
            stem = Path(ep).stem
            p = self.action_root / stem / self.action_file
            if not p.exists():
                raise FileNotFoundError(f"{ep}: action file {p} not found")
            self._actions[ep] = HandActionFile(p)
        return self._actions[ep]

    def _episode(self, ep: str) -> dict:
        if ep in self._cache:
            self._cache.move_to_end(ep)
            return self._cache[ep]
        paths = self._paths(ep)
        d = {"visual": _load_pt(paths["visual"]), "tac": _load_pt(paths["tac"])}
        if paths["wrist"].exists():
            d["wrist"] = _load_pt(paths["wrist"])
        self._cache[ep] = d
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return d

    # ------------------------------------------------------------------ api
    def __len__(self) -> int:
        return len(self.index)

    def presence_summary(self) -> dict:
        n_w = sum(1 for e in self.episode_info if e["has_wrist"])
        w_win = sum(e["num_windows"] for e in self.episode_info if e["has_wrist"])
        return {"episodes": len(self.episode_info), "episodes_with_wrist": n_w,
                "windows": len(self.index), "windows_with_wrist_episodes": w_win}

    def window_meta(self, i: int) -> tuple[str, int]:
        ei, w = self.index[i]
        info = self.episode_info[ei]
        return info["episode"], int(info["starts"][w].item())

    def item_at(self, ep: str, t_start: int) -> dict | None:
        """Window of episode ``ep`` starting at frame ``t_start`` regardless of window_stride; None if absent."""
        ei = self.episodes.index(ep)
        starts = self.episode_info[ei]["starts"]
        hit = (starts == int(t_start)).nonzero()
        if hit.numel() == 0:
            return None
        return self._window(ei, int(hit[0].item()))

    def __getitem__(self, i: int) -> dict:
        ei, w = self.index[i]
        return self._window(ei, w)

    def _window(self, ei: int, w: int) -> dict:
        info = self.episode_info[ei]
        ep = info["episode"]
        t_start = int(info["starts"][w].item())
        files = self._episode(ep)
        present = torch.tensor(info["present"], dtype=torch.bool)
        latents = {}
        for s in IMAGE_STREAMS:
            if not present[STREAM_ID[s]]:
                continue  # absent optional stream: no key at all (never a zero latent)
            sd, key = LATENT_FILE_KEYS[s]
            latents[s] = files[sd][key][w].float().clone()  # (48,4,16,16)
        acts, valid = self._action_file(ep).window(t_start, N_ACTIONS)
        if self.include_view_actions:
            view, view_valid = self._view_actions[ep].window(t_start)
            acts.update(view)
            valid.update(view_valid)
        return {
            "episode_id": ep,
            "window_start_index": torch.tensor(t_start),
            "extraction_fps": torch.tensor(EXTRACTION_FPS),
            "frame_timestamps_s": torch.arange(WINDOW_FRAMES, dtype=torch.float32) / EXTRACTION_FPS,
            "action_timestamps_s": torch.tensor(ACTION_TICKS, dtype=torch.float32) / EXTRACTION_FPS,
            "latents": latents,
            "actions": acts,
            "action_valid": valid,
            "stream_present": present,
        }


def collect_action_rows(ds: VtmWindowDataset, window_stride: int = 8, max_windows: int | None = None,
                        streams=ACTION_STREAMS) -> torch.Tensor:
    """Stack both hands' physical 9-D rows of ``streams`` for normalizer fitting (train split only).
    Pass ``streams=VIEW_ACTION_STREAMS`` for the per-frame view points of the action_full task."""
    rows = []
    n = 0
    for i in range(0, len(ds), window_stride):
        item = ds[i]
        for s in streams:
            if s not in item["actions"]:
                raise KeyError(f"{s} not in dataset items (include_view_actions?)")
            a, v = item["actions"][s], item["action_valid"][s]
            rows.append(a[v])
        n += 1
        if max_windows is not None and n >= max_windows:
            break
    return torch.cat(rows, 0)



# ---------------------------------------------------------------------------
# collate: group samples by active-stream pattern (spec-MM §3)
# ---------------------------------------------------------------------------
class WristDropout:
    """Paired, whole-clip wrist dropout decided once per sample per presentation.

    ``probability`` is applied only to samples that actually have wrists; a hit removes BOTH
    wrist streams for history and future. ``hide_all=True`` forces the drop for every sample
    that has wrists (the "artificially hide wrist" validation arm)."""

    def __init__(self, probability: float = 0.0, hide_all: bool = False, generator: torch.Generator | None = None):
        self.p = float(probability)
        self.hide_all = bool(hide_all)
        self.generator = generator

    def __call__(self, present: torch.Tensor) -> torch.Tensor:
        dropped = torch.zeros_like(present)
        has_wrist = all(bool(present[STREAM_ID[s]]) for s in WRIST_STREAMS)
        if not has_wrist:
            return dropped
        hit = self.hide_all or (self.p > 0.0 and float(torch.rand((), generator=self.generator)) < self.p)
        if hit:
            for s in WRIST_STREAMS:
                dropped[STREAM_ID[s]] = True
        return dropped


def collate_by_active(items: list[dict], dropout: WristDropout | None = None) -> list[dict]:
    """Group a list of dataset items by their ACTIVE pattern (present & ~dropped) and stack
    each group's tensors. Returns a list of group batches; absent streams have no key."""
    dropout = dropout or WristDropout(0.0)
    groups: dict[tuple[bool, ...], list[tuple[dict, torch.Tensor]]] = {}
    for it in items:
        present = it["stream_present"].bool()
        dropped = dropout(present)
        active = present & ~dropped
        groups.setdefault(tuple(bool(x) for x in active), []).append((it, dropped))
    out = []
    for pat in sorted(groups.keys(), reverse=True):  # deterministic order: fullest pattern first
        members = groups[pat]
        active = streams_from_pattern(pat)
        lat = {s: torch.stack([m[0]["latents"][s] for m in members]) for s in IMAGE_STREAMS if s in active}
        out.append({
            "active": active,
            "pattern": pat,
            "latents": lat,
            "actions": {s: torch.stack([m[0]["actions"][s] for m in members]) for s in ALL_ACTION_STREAMS if s in active},
            "action_valid": {s: torch.stack([m[0]["action_valid"][s] for m in members]) for s in ALL_ACTION_STREAMS if s in active},
            "stream_present": torch.stack([m[0]["stream_present"] for m in members]),
            "stream_dropped": torch.stack([m[1] for m in members]),
            "episode_id": [m[0]["episode_id"] for m in members],
            "window_start_index": torch.stack([m[0]["window_start_index"] for m in members]),
        })
    return out
