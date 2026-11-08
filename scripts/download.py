"""Download everything needed to run the three released models on a new machine.

    python scripts/download.py --what all                       # ~200 GB, everything
    python scripts/download.py --what data ckpts nano wan_vae --tasks rope   # one task, no raw

--what (any subset):
  data     preprocessed latents + actions   WitneyWW/vtm-cosmos-data :: mixed_<tree>_{latents,actions}/
  raw      5-stream mp4 + parquet inputs    WitneyWW/vtm-cosmos-data :: raw/  (needed for eval video
           decoding -- the VAE anchor frame is re-read from the mp4 -- and for re-running preprocessing)
  ckpts    epoch_019 LoRA ckpt per task + tactile VAE LoRA   WitneyWW/vtm-cosmos-nano-ckpts
  nano     base model                        nvidia/Cosmos3-Nano (transformer/ + json, ~29 GB)
  wan_vae  Wan2.2 VAE                        Wan-AI/Wan2.2-TI2V-5B :: Wan2.2_VAE.pth (2.8 GB)
Destinations: $ACTS_DATA_ROOT and $ACTS_CKPT_ROOT (source env.sh first). Token: $HF_TOKEN or `hf auth login`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

DATA_REPO = "WitneyWW/vtm-cosmos-data"
CKPT_REPO = "WitneyWW/vtm-cosmos-nano-ckpts"
TREES = {"umi_all": ("mixed_mb", "mixed_mb_wrist_holdout"),
         "rope": ("mixed_rope", "mixed_rope_wrist_holdout"),
         "pusht": ("mixed_pushT", "mixed_pusht_wrist_holdout")}
REPO = Path(__file__).resolve().parents[1]


def split_episodes(task: str, parts: list[str]) -> list[str]:
    s = json.loads((REPO / "splits" / f"{TREES[task][1]}.json").read_text())
    return [Path(e).stem for p in parts for e in s[p]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", nargs="+", default=["all"],
                    choices=["all", "data", "raw", "ckpts", "nano", "wan_vae"])
    ap.add_argument("--tasks", nargs="+", default=list(TREES), choices=list(TREES))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"],
                    help="restrict data/raw to these split parts (e.g. --splits test for eval only)")
    ap.add_argument("--data-root", default=os.environ.get("ACTS_DATA_ROOT", str(REPO / "data")))
    ap.add_argument("--ckpt-root", default=os.environ.get("ACTS_CKPT_ROOT", str(REPO / "checkpoints")))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    what = {"data", "raw", "ckpts", "nano", "wan_vae"} if "all" in a.what else set(a.what)
    data, ck = Path(a.data_root), Path(a.ckpt_root)

    if what & {"data", "raw"}:
        pats = []
        for t in a.tasks:
            tree = TREES[t][0]
            eps = split_episodes(t, a.splits)
            if "data" in what:
                for e in eps:
                    pats += [f"{tree}_latents/{sub}/{e}.pt" for sub in ("tac", "visual", "wrist")]
                    pats.append(f"{tree}_actions/{e}/*")
            if "raw" in what:
                for e in eps:
                    pats += [f"raw/videos/{e}/*", f"raw/parquet/{e}.parquet"]
        pats += ["splits/*", "raw/episodes.json"]
        print(f"[data] {len(pats)} patterns -> {data}")
        snapshot_download(DATA_REPO, repo_type="dataset", local_dir=data, allow_patterns=pats, max_workers=a.workers)
    if "ckpts" in what:
        pats = ["vae/*"] + [f"nano_action_causal_{t}_v1/*" for t in a.tasks]
        print(f"[ckpts] {pats} -> {ck}")
        snapshot_download(CKPT_REPO, local_dir=ck, allow_patterns=pats, max_workers=a.workers)
    if "nano" in what:
        print(f"[nano] nvidia/Cosmos3-Nano -> {ck / 'Cosmos3-Nano'}")
        snapshot_download("nvidia/Cosmos3-Nano", local_dir=ck / "Cosmos3-Nano",
                          allow_patterns=["*.json", "transformer/*", "*.md"], max_workers=a.workers)
    if "wan_vae" in what:
        print(f"[wan_vae] Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth -> {ck}")
        hf_hub_download("Wan-AI/Wan2.2-TI2V-5B", "Wan2.2_VAE.pth", local_dir=ck)
    print("done")


if __name__ == "__main__":
    main()
