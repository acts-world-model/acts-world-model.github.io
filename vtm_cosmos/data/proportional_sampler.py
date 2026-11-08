"""Rank-synchronous wrist/no-wrist batches with committed, prefetch-safe cursors."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import deque
from pathlib import Path

import torch
from torch.utils.data import Sampler

from vtm_cosmos.streams import STREAM_ID


def wrist_pools(dataset) -> tuple[list[int], list[int], str]:
    """Index only retained windows; fingerprint their identities and backing file metadata."""
    pools = ([], [])
    digest = hashlib.sha256()
    for info in dataset.episode_info:
        digest.update(json.dumps({k: info[k] for k in ("episode", "present", "fps_stride")},
                                 sort_keys=True).encode())
        ep = info["episode"]
        paths = [dataset.latent_root / folder / ep for folder in ("visual", "tac", "wrist")]
        paths.append(dataset.action_root / Path(ep).stem / dataset.action_file)
        if dataset.include_view_actions:
            paths.append(dataset.action_root / Path(ep).stem / dataset.view_action_file)
        for path in paths:
            path = path.resolve()
            stat = path.stat() if path.exists() else None
            digest.update(json.dumps([str(path), None if stat is None else
                                      [stat.st_size, stat.st_mtime_ns]]).encode())
    for index, (ei, wi) in enumerate(dataset.index):
        info = dataset.episode_info[ei]
        left, right = (bool(info["present"][STREAM_ID[s]]) for s in ("wrist_left", "wrist_right"))
        if left != right:
            raise ValueError(f"{info['episode']}: single-wrist samples need a separate sampling bucket")
        pools[0 if left else 1].append(index)
        digest.update(json.dumps([ei, int(info["starts"][wi]), left]).encode())
    return *pools, digest.hexdigest()


class ProportionalWristBatchSampler(Sampler[list[int]]):
    """All ranks draw disjoint slices of the same shuffled global bucket blocks.

    Iteration may run ahead through DataLoader prefetch. Only ``commit()`` moves
    the saved position, and must be called after each successful optimizer step.
    An iteration ends at the next reporting-epoch boundary, including on resume.
    """

    VERSION = 1

    def __init__(self, with_wrist, without_wrist, *, batch_size: int, steps_per_epoch: int,
                 rank: int = 0, world_size: int = 1, seed: int = 0, fingerprint: str = ""):
        self.pools = (tuple(with_wrist), tuple(without_wrist))
        self.batch_size, self.steps_per_epoch = int(batch_size), int(steps_per_epoch)
        self.rank, self.world_size, self.seed = int(rank), int(world_size), int(seed)
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be in [0, world_size)")
        if self.batch_size < 2 or self.steps_per_epoch < 1:
            raise ValueError("batch_size >= 2 and steps_per_epoch >= 1 are required")
        flat = self.pools[0] + self.pools[1]
        if any(not isinstance(i, int) or i < 0 for i in flat) or len(set(flat)) != len(flat):
            raise ValueError("bucket indices must be distinct nonnegative integers")
        self.n_w, self.n_n = map(len, self.pools)
        self.total = self.n_w + self.n_n
        if not self.n_w or not self.n_n:
            raise ValueError("dataset_proportional requires nonempty with-wrist AND without-wrist pools")
        if not self.total <= self.batch_size * self.n_w <= (self.batch_size - 1) * self.total:
            raise ValueError("Dataset ratio cannot include both buckets in every update at this batch_size; "
                             "increase the effective batch size, rather than oversampling the rare bucket")
        for size in (self.n_w, self.n_n):
            max_quota = (self.batch_size * size + self.total - 1) // self.total
            if size < self.world_size * max_quota:
                raise ValueError("Bucket too small for distinct global samples at the maximum quota")
        pool_hash = hashlib.sha256(json.dumps(self.pools).encode()).hexdigest()
        self.signature = dict(version=self.VERSION, batch_size=self.batch_size,
                              steps_per_epoch=self.steps_per_epoch, world_size=self.world_size,
                              seed=self.seed, sizes=[self.n_w, self.n_n], pools=pool_hash,
                              fingerprint=fingerprint)
        self._state = {"step": 0, "buckets": [dict(cycle=0, offset=0, drawn=0, dropped=0) for _ in self.pools]}
        self._pending = deque()
        self._permutations = {}
        self._generation = 0

    @classmethod
    def from_dataset(cls, dataset, **kwargs):
        with_wrist, without_wrist, fingerprint = wrist_pools(dataset)
        return cls(with_wrist, without_wrist, fingerprint=fingerprint, **kwargs)

    @property
    def completed_steps(self) -> int:
        return self._state["step"]

    def quotas(self, step: int) -> tuple[int, int]:
        if step < 0:
            raise ValueError("step must be nonnegative")
        w = ((step + 1) * self.batch_size * self.n_w) // self.total - (step * self.batch_size * self.n_w) // self.total
        return w, self.batch_size - w

    def _permutation(self, bucket: int, cycle: int):
        cached = self._permutations.get(bucket)
        if cached is None or cached[0] != cycle:
            key = hashlib.sha256(f"{self.seed}:{bucket}:{cycle}".encode()).digest()
            generator = torch.Generator().manual_seed(int.from_bytes(key[:8], "little") % (2**63))
            pool = self.pools[bucket]
            perm = [pool[i] for i in torch.randperm(len(pool), generator=generator).tolist()]
            self._permutations[bucket] = (cycle, perm)
        return self._permutations[bucket][1]

    def _draw(self, state, bucket, quota):
        cursor = state["buckets"][bucket]
        size, count = len(self.pools[bucket]), quota * self.world_size
        if cursor["offset"] + count > size:
            cursor["dropped"] += size - cursor["offset"]
            cursor["cycle"] += 1
            cursor["offset"] = 0
        perm = self._permutation(bucket, cursor["cycle"])
        start = cursor["offset"] + self.rank * quota
        indices = perm[start:start + quota]
        cursor["offset"] += count
        cursor["drawn"] += count
        return indices

    def __iter__(self):
        self._generation += 1
        generation = self._generation
        self._pending.clear()
        state = copy.deepcopy(self._state)
        remaining = self.steps_per_epoch - state["step"] % self.steps_per_epoch
        for _ in range(remaining):
            if generation != self._generation:
                raise RuntimeError("Concurrent sampler iterators are not supported")
            indices = []
            for bucket, quota in enumerate(self.quotas(state["step"])):
                indices.extend(self._draw(state, bucket, quota))
            state["step"] += 1
            self._pending.append(copy.deepcopy(state))
            yield indices

    def __len__(self):
        return self.steps_per_epoch - self.completed_steps % self.steps_per_epoch

    def commit(self):
        if not self._pending or self._pending[0]["step"] != self.completed_steps + 1:
            raise RuntimeError("commit() requires a yielded, uncommitted training batch")
        self._state = self._pending.popleft()

    def state_dict(self):
        return copy.deepcopy({"signature": self.signature, "cursor": self._state})

    def load_state_dict(self, state):
        if state.get("signature") != self.signature:
            raise ValueError("Incompatible sampling resume: dataset, quotas, seed, epoch length, or world size changed")
        cursor = state["cursor"]
        step = cursor["step"]
        if not isinstance(step, int) or step < 0 or len(cursor["buckets"]) != 2:
            raise ValueError("Invalid sampler cursor")
        drawn_w = (step * self.batch_size * self.n_w // self.total) * self.world_size
        for bucket, (c, size) in enumerate(zip(cursor["buckets"], (self.n_w, self.n_n))):
            expected = drawn_w if bucket == 0 else step * self.batch_size * self.world_size - drawn_w
            if (any(not isinstance(c[k], int) or c[k] < 0 for k in ("cycle", "offset", "drawn", "dropped"))
                    or c["offset"] > size or c["drawn"] != expected
                    or c["cycle"] * size + c["offset"] != c["drawn"] + c["dropped"]):
                raise ValueError("Invalid sampler bucket cursor")
        self._state = copy.deepcopy(cursor)
        self._pending.clear()
        self._permutations.clear()
        self._generation += 1

    def metrics(self):
        total = self.completed_steps * self.batch_size * self.world_size
        out = {"sampling/target_wrist_fraction": self.n_w / self.total,
               "sampling/drawn_wrist_fraction": self._state["buckets"][0]["drawn"] / max(total, 1)}
        for name, size, cursor in zip(("with_wrist", "without_wrist"), (self.n_w, self.n_n), self._state["buckets"]):
            out[f"sampling/{name}_draws_per_window"] = cursor["drawn"] / size
            out[f"sampling/{name}_cycles"] = cursor["cycle"]
            out[f"sampling/{name}_dropped_tail"] = cursor["dropped"]
        return out
