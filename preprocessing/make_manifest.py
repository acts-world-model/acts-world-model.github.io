"""Write the encoder manifest (episodes.jsonl) for one task from the downloaded raw/ tree.

    python preprocessing/make_manifest.py --raw-root $ACTS_DATA_ROOT/raw --task rope --out OUT/episodes_rope.jsonl

raw/ layout (WitneyWW/vtm-cosmos-data):
    raw/videos/<episode>/{view_middle,tactile_left,tactile_right,wrist_left,wrist_right}.mp4   (30 fps)
    raw/parquet/<episode>.parquet      per-native-frame sensor poses + calibrated normal force
    raw/episodes.json                  [{episode, tree, num_frames}] for the 166 split episodes
Row schema = what third_party/wan22_vae/encode_tactile_wan22.py reads."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TREES = {"umi_all": "mixed_mb", "rope": "mixed_rope", "pusht": "mixed_pushT"}
STREAMS = ["view_middle", "tactile_left", "tactile_right", "wrist_left", "wrist_right"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--task", required=True, choices=sorted(TREES))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", nargs="*", default=None, help="subset (episode stems)")
    a = ap.parse_args()
    rows = [r for r in json.loads((a.raw_root / "episodes.json").read_text()) if r["tree"] == TREES[a.task]]
    if a.episodes:
        rows = [r for r in rows if r["episode"] in set(a.episodes)]
    out = []
    for r in rows:
        d = a.raw_root / "videos" / r["episode"]
        row = {"episode": r["episode"], "num_frames": int(r["num_frames"]), "task": a.task}
        for s in STREAMS:
            p = d / f"{s}.mp4"
            if not p.is_file():
                raise FileNotFoundError(p)
            row[s] = str(p.resolve())
        out.append(row)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in out))
    print(f"{len(out)} episodes -> {a.out}")


if __name__ == "__main__":
    main()
