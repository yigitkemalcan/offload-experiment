#!/usr/bin/env python3
"""Plot task demotion/promotion times against offloaded size.

Usage:
    ../.venv/bin/python plot_offload.py
    ../.venv/bin/python plot_offload.py --summary results/summary.json --out results/offload_times.png

Requires matplotlib. Each task contributes two crosses at the same x coordinate.
Size is demote.written_to_disk_mib (swap + cache snapshot), not Docker image size
or promotion read volume. Times are the transfer timings recorded in summary.json.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "results" / "summary.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "offload_times.png")
    args = parser.parse_args()

    tasks = json.loads(args.summary.read_text())["tasks"]
    if not tasks:
        parser.error("The summary contains no tasks to plot.")
    sizes = [task["demote"]["written_to_disk_mib"] for task in tasks]
    demote = [task["demote"]["total_ms"] for task in tasks]
    promote = [task["promote"]["total_ms"] for task in tasks]

    blue, orange = "#0072B2", "#D55E00"
    fig, left = plt.subplots(figsize=(9, 5.5), layout="constrained")
    right = left.twinx()
    demote_points = left.scatter(sizes, demote, marker="x", s=38, linewidths=1.4,
                                 color=blue, alpha=0.8, label="Demotion (left axis)")
    promote_points = right.scatter(sizes, promote, marker="x", s=38, linewidths=1.4,
                                    color=orange, alpha=0.8, label="Promotion (right axis)")
    left.set_xlabel("Offloaded size: swap + cache snapshot (MiB)")
    left.set_ylabel("Demotion time (ms)", color=blue)
    right.set_ylabel("Promotion time (ms)", color=orange)
    left.tick_params(axis="y", colors=blue)
    right.tick_params(axis="y", colors=orange)
    left.spines["left"].set_color(blue)
    right.spines["right"].set_color(orange)
    # Both quantities have the same units: matched scales avoid misleading comparisons.
    ceiling = max(max(demote), max(promote), 1) * 1.08
    left.set_ylim(0, ceiling)
    right.set_ylim(0, ceiling)
    left.set_xlim(left=0)
    left.set_axisbelow(True)
    left.grid(alpha=0.2)
    left.set_title(f"Container offload transfer times — {len(tasks)} tasks")
    left.legend(handles=[demote_points, promote_points], loc="upper left", frameon=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out} ({len(tasks)} tasks, {2 * len(tasks)} crosses)")


if __name__ == "__main__":
    main()
