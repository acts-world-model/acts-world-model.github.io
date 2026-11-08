"""Aggregate the per-case results of eval/run_test_case.py into a markdown table, optionally
paired against the reference numbers shipped in eval/cases/<task>_reference_metrics.json.

    python eval/summarize.py --task rope [--root DIR] [--reference]

short = 8 observed -> 8 predicted frames (metrics on the 8 predicted frames);
long  = mean over the 15 autoregressive windows (120 generated frames).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STREAMS = ["video", "wrist_left", "wrist_right", "tactile_left", "tactile_right"]
METRICS = ["psnr", "ssim", "lpips"]


def load(root: Path, man: dict) -> dict:
    out = {}
    for c in man["cases"]:
        d = root / c["output_subdir"]
        if not (d / "complete.json").is_file():
            continue
        w = json.loads((d / "window_eval.json").read_text())["results"][0]
        r = {"category": c["category"], "short": {s: {m: w["image"][s][m] for m in METRICS} for s in STREAMS}}
        rr = d / "reference_rollout.json"
        if c["rollout_windows"] and rr.is_file():
            wins = json.loads(rr.read_text())[0]["windows"]
            r["long_mean"] = {s: {m: sum(x["image"][s][m] for x in wins) / len(wins) for m in METRICS} for s in STREAMS}
        out[c["output_subdir"]] = r
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def table(res: dict, hz: str, title: str) -> list[str]:
    rows = [r for r in res.values() if hz in r]
    lines = [f"### {title} (n={len(rows)})", "", "| stream | PSNR ↑ | SSIM ↑ | LPIPS ↓ |", "|---|---:|---:|---:|"]
    for s in STREAMS:
        v = [mean([r[hz][s][m] for r in rows]) for m in METRICS]
        lines.append(f"| {s} | {v[0]:.2f} | {v[1]:.4f} | {v[2]:.4f} |")
    return lines + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["umi_all", "rope", "pusht"])
    ap.add_argument("--root", default=None, help="default: $ACTS_RUNS_ROOT/<run_name>/test_epoch_019")
    ap.add_argument("--reference", action="store_true", help="also print paired deltas vs the shipped reference metrics")
    a = ap.parse_args()
    man = json.loads((REPO / "eval" / "cases" / f"{a.task}_test_cases.json").read_text())
    runs = Path(os.environ.get("ACTS_RUNS_ROOT", REPO / "runs"))
    root = Path(a.root or runs / man["run_name"] / "test_epoch_019")
    res = load(root, man)
    print(f"# {man['run_name']} epoch 019 test ({len(res)}/{len(man['cases'])} cases complete) -- {root}\n")
    print("\n".join(table(res, "short", "short horizon: 8 -> 8 frames")))
    print("\n".join(table(res, "long_mean", "long horizon: mean over 15 chained windows")))
    if a.reference:
        ref = {r["output_subdir"]: r for r in json.loads(
            (REPO / "eval" / "cases" / f"{a.task}_reference_metrics.json").read_text())["cases"]}
        print("### paired difference vs reference (this run − reference), PSNR dB\n")
        print("| horizon | stream | n | mean Δ | max |Δ| |\n|---|---|---:|---:|---:|")
        for hz in ("short", "long_mean"):
            for s in STREAMS:
                d = [res[k][hz][s]["psnr"] - ref[k][hz][s]["psnr"] for k in res if hz in res[k] and hz in ref.get(k, {})]
                if d:
                    print(f"| {hz} | {s} | {len(d)} | {mean(d):+.3f} | {max(abs(x) for x in d):.3f} |")


if __name__ == "__main__":
    main()
