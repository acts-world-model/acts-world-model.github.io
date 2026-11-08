"""Run one fixed test case (short 8->8 window + 15-window autoregressive rollout) and render
GT/prediction videos for the five image streams.

    python eval/run_test_case.py --task rope --case 0            # one case
    python eval/run_test_case.py --task rope --case all          # every case, sequentially

The case list (episode, dataset index, contact class) is eval/cases/<task>_test_cases.json; it is the
exact set used for the epoch-019 test numbers. Outputs go to
$ACTS_RUNS_ROOT/<run_name>/test_epoch_019/<output_subdir>/ (window_eval.json, reference_rollout.json,
strip PNGs, short_gt_pred.mp4, long_gt_pred.mp4). Summarise with eval/summarize.py.
Needs env.sh sourced (ACTS_DATA_ROOT, ACTS_CKPT_ROOT, VTM_SOURCE_VIDEO_ROOT)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
STREAMS = ["video", "wrist_left", "wrist_right", "tactile_left", "tactile_right"]
TREES = {"umi_all": "mixed_mb", "rope": "mixed_rope", "pusht": "mixed_pushT"}


def env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"${name} is not set -- run `source env.sh` first")
    return v


def run_case(man: dict, case: dict, ckpt: Path, out_root: Path, steps: int, skip_done: bool):
    out = out_root / case["output_subdir"]
    if skip_done and (out / "complete.json").is_file():
        print("[skip] done:", out, flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / "case.json").write_text(json.dumps(case, indent=2))
    print("[case]", json.dumps({k: case[k] for k in ("task", "episode", "category", "index", "t_start", "rollout_windows")}), flush=True)
    data, ckroot = env("ACTS_DATA_ROOT"), env("ACTS_CKPT_ROOT")
    tree = TREES[man["task"]]
    overrides = [
        f"data.latent_root={data}/{tree}_latents",
        f"data.action_root={data}/{tree}_actions",
        f"data.split_file={REPO / man['split_file']}",
        f"model.nano_checkpoint={ckroot}/Cosmos3-Nano",
        f"out_dir={out_root}",
    ]
    subprocess.run([sys.executable, "-u", "-m", "vtm_cosmos.infer.rollout",
                    "--ckpt", str(ckpt), "--split", man["split"], "--window-stride", "1",
                    "--windows", str(case["index"]), "--rollout-windows", str(case["rollout_windows"]),
                    "--steps", str(steps), "--solver", man.get("solver", "unipc"), "--seed", str(man.get("seed", 0)),
                    "--out-dir", str(out), "--override", *overrides], check=True, cwd=REPO)

    w = json.loads((out / "window_eval.json").read_text())["results"][0]
    if (w["episode"], int(w["t_start"])) != (case["episode"], int(case["t_start"])):
        raise RuntimeError(f"dataset index {case['index']} resolved to {w['episode']}@{w['t_start']}, "
                           f"expected {case['episode']}@{case['t_start']} (wrong split / data version?)")

    # Each saved strip holds 16 frames, GT on top and prediction below. Show the 8 observed frames once,
    # then append each rollout window's 8 future frames.
    stem = f"{Path(case['episode']).stem}_t{case['t_start']:05d}"
    if case["rollout_windows"]:
        rolls = json.loads((out / "reference_rollout.json").read_text())
        assert len(rolls) == 1 and len(rolls[0]["windows"]) == case["rollout_windows"], "incomplete rollout"
        assert not any("stopped" in x for x in rolls[0]["windows"]), "rollout stopped early"
    for mode in (("short", "long") if case["rollout_windows"] else ("short",)):
        video_path = out / f"{mode}_gt_pred.mp4"
        count = 0
        with imageio.get_writer(video_path, fps=15, codec="libx264", quality=8) as writer:
            for k in range(1 if mode == "short" else case["rollout_windows"]):
                suffix = "" if mode == "short" else f"_roll{k}"
                strips = [imageio.imread(out / f"{stem}{suffix}_{s}.png") for s in STREAMS]
                h = (strips[0].shape[0] - 4) // 2
                wd = strips[0].shape[1] // 16 - 2
                for t in range(0 if k == 0 else 8, 16):
                    canvas = Image.new("RGB", (len(STREAMS) * wd, 2 * h + 64), "black")
                    draw = ImageDraw.Draw(canvas)
                    for col, (stream, strip) in enumerate(zip(STREAMS, strips)):
                        x = t * (wd + 2)
                        canvas.paste(Image.fromarray(strip[:h, x:x + wd]), (col * wd, 32))
                        canvas.paste(Image.fromarray(strip[h + 4:h + 4 + h, x:x + wd]), (col * wd, h + 64))
                        draw.text((col * wd + 4, 16), "GT: " + stream, fill="white")
                        draw.text((col * wd + 4, h + 48), "PRED: " + stream, fill="white")
                    label = "observed history" if count < 8 else "predicted future"
                    draw.text((4, 2), f"{mode} | {case['category']} | frame {count} | {label}", fill="white")
                    writer.append_data(np.asarray(canvas))
                    count += 1
        print("[video]", video_path, "frames=", count, flush=True)
    (out / "complete.json").write_text(json.dumps(dict(
        short_frames=16, long_frames=8 + 8 * case["rollout_windows"] if case["rollout_windows"] else 0,
        rollout_windows=case["rollout_windows"])))
    print("[case done]", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TREES))
    ap.add_argument("--case", required=True, help="case number, comma list, or 'all'")
    ap.add_argument("--ckpt", default=None, help="default: $ACTS_CKPT_ROOT/<run_name>/epoch_019.pt")
    ap.add_argument("--out-root", default=None, help="default: $ACTS_RUNS_ROOT/<run_name>/test_epoch_019")
    ap.add_argument("--steps", type=int, default=None, help="UniPC steps per latent block (default 35 = paper)")
    ap.add_argument("--short-only", action="store_true", help="skip the 15-window rollout (~10x faster)")
    ap.add_argument("--redo", action="store_true", help="re-run cases that already have complete.json")
    a = ap.parse_args()

    man = json.loads((REPO / "eval" / "cases" / f"{a.task}_test_cases.json").read_text())
    ckpt = Path(a.ckpt or Path(env("ACTS_CKPT_ROOT")) / man["checkpoint"])
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt} (python scripts/download.py --what ckpts)")
    runs = Path(os.environ.get("ACTS_RUNS_ROOT", REPO / "runs"))
    out_root = Path(a.out_root or runs / man["run_name"] / "test_epoch_019")
    out_root.mkdir(parents=True, exist_ok=True)
    ids = range(len(man["cases"])) if a.case == "all" else [int(x) for x in a.case.split(",")]
    steps = a.steps or man.get("denoising_steps_per_latent_block", 35)
    for i in ids:
        case = dict(man["cases"][i])
        if a.short_only:
            case["rollout_windows"] = 0
        run_case(man, case, ckpt, out_root, steps, skip_done=not a.redo)


if __name__ == "__main__":
    main()
