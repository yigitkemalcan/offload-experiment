#!/usr/bin/env python3
"""Demote and promote time against container memory, one panel per experiment, log-log.

Usage (from this directory):
    ../.venv/bin/python plot_scaling.py
    ../.venv/bin/python plot_scaling.py --panel "with snapshot (files)=results/summary.json" ...

Default panels: (a) snapshot-promote/ (demote writes the cache snapshot, promote reads it back) and
(b) no-snapshot/ (no snapshot; promote re-reads the container's files). x is the container's memory at the
freeze (memory_at_freeze_mib.total), the same in every experiment. Each line is a constant-throughput fit,
time = size / throughput; the throughput in the legend is the geometric mean of the per-task throughputs
(the least-squares fit of a slope-1 line in log-log). Writes plots/snapshot_scaling.png.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator

ROOT = Path(__file__).resolve().parent
# same colors as offload_times.png / offload_cdf.png
BLUE, ORANGE = "#0072B2", "#D55E00"
DEFAULT_PANELS = [f"with snapshot={ROOT / 'snapshot-promote/results/summary.json'}",
                  f"without snapshot={ROOT / 'no-snapshot/results/summary.json'}"]


def fit_throughput(size_mib, time_s):
    """Geometric-mean throughput (MiB/s)."""
    return float(np.exp(np.mean(np.log(size_mib) - np.log(time_s))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", action="append", metavar="TITLE=SUMMARY_JSON",
                    help="repeat for each panel (default: snapshot-promote, no-snapshot)")
    ap.add_argument("--out", type=Path, default=ROOT / "plots" / "snapshot_scaling.png")
    args = ap.parse_args()

    panels = []
    for spec in args.panel or DEFAULT_PANELS:
        title, path = spec.split("=", 1)
        tasks = json.loads(Path(path).read_text())["tasks"]
        size = np.array([t["memory_at_freeze_mib"]["total"] for t in tasks])
        times = {k: np.array([t[k]["total_ms"] for t in tasks]) / 1e3 for k in ("demote", "promote")}
        panels.append((title, size, times))

    plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 4.6), sharey=True,
                             layout="constrained", squeeze=False)
    all_size = np.concatenate([p[1] for p in panels])
    all_time = np.concatenate([t for p in panels for t in p[2].values()])
    xlim = (all_size.min() / 1.15, all_size.max() * 1.15)
    ylim = (all_time.min() / 1.4, all_time.max() * 2.5)  # headroom for the legend
    xs = np.geomspace(*xlim, 50)

    for i, (ax, (title, size, times)) in enumerate(zip(axes[0], panels)):
        for key, label, color, marker in [("demote", "demote", BLUE, "o"), ("promote", "promote", ORANGE, "^")]:
            thr = fit_throughput(size, times[key])
            ax.scatter(size, times[key], s=26, marker=marker, color=color, alpha=0.55, linewidths=0,
                       label=f"{label}  ({thr:,.0f} MiB/s)")
            ax.plot(xs, xs / thr, color=color, linewidth=2)
        ax.set(xscale="log", yscale="log", xlim=xlim, ylim=ylim, xlabel="Container memory at freeze (MiB)")
        ax.set_title(f"({chr(97 + i)}) {title}", loc="left")
        ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_locator(LogLocator(subs=(1, 2.5, 5)))
        fmt = FuncFormatter(lambda v, _: f"{v:,.3g}")
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        ax.grid(axis="y", which="major", color="#d0d0d0", linewidth=0.8)
        ax.legend(loc="upper left", frameon=False, handletextpad=0.6)
    axes[0][0].set_ylabel("Time (s)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
