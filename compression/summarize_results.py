#!/usr/bin/env python3
"""Summarize replay_compress.py results into results/summary.json.

For all tasks together and for each run:
  tasks:      captured, and failed or never triggered
  memory_mib: cgroup usage at the freeze, the image the codecs compress, and the zero pages dropped from it
              (min / median / max)
  codecs:     median ratio, compress and decompress time (ms) and throughput (MiB/s)

Usage (from this directory; results/ is root-owned, hence sudo):
  sudo ../.venv/bin/python summarize_results.py                 # results/ -> results/summary.json
  sudo ../.venv/bin/python summarize_results.py RESULTS_DIR     # RESULTS_DIR/summary.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MIB = 2**20
PAGE = 4096


def spread(values) -> dict:
    a = np.asarray(values, dtype=float) / MIB
    return {"min": round(a.min(), 1), "median": round(float(np.median(a)), 1), "max": round(a.max(), 1)}


def summarize(trials: list[dict], failed: int) -> dict:
    codecs = {}
    for t in trials:
        for c in t["codecs"]:
            codecs.setdefault(c["codec"], []).append(c)
    med = lambda rows, f: float(np.median([f(c) for c in rows]))
    return {
        "tasks": {"captured": len(trials), "failed_or_untriggered": failed},
        "memory_mib": {
            "usage_at_freeze": spread([t["freeze_usage_in_bytes"] for t in trials]),
            "image": spread([t["capture_bytes"] for t in trials]),
            "zero_pages_dropped": spread([t["capture_zero_pages"] * PAGE for t in trials]),
        },
        "codecs": {name: {
            "ratio": round(med(rows, lambda c: c["ratio"]), 2),
            "compress_ms": round(med(rows, lambda c: c["compress_s"]) * 1e3, 1),
            "decompress_ms": round(med(rows, lambda c: c["decompress_s"]) * 1e3, 1),
            "compress_mib_s": round(med(rows, lambda c: c["compress_mibps"]), 1),
            "decompress_mib_s": round(med(rows, lambda c: c["decompress_mibps"]), 1),
        } for name, rows in codecs.items()},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path, nargs="?", default=ROOT / "results")
    ap.add_argument("--out", type=Path, help="default: RESULTS/summary.json")
    args = ap.parse_args()
    args.out = args.out or args.results / "summary.json"

    runs = {}
    for path in sorted(args.results.glob("*/*/trial.json")):
        trial = json.loads(path.read_text())
        ok = trial.get("triggered") and trial.get("codecs") and not trial.get("error")
        runs.setdefault(path.parent.parent.name, ([], []))[0 if ok else 1].append(trial)

    everything = [t for ok, _ in runs.values() for t in ok]
    summary = {"all": summarize(everything, sum(len(bad) for _, bad in runs.values()))}
    summary |= {run: summarize(ok, len(bad)) for run, (ok, bad) in runs.items()}
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out}: {len(everything)} tasks over {len(runs)} runs")


if __name__ == "__main__":
    main()
