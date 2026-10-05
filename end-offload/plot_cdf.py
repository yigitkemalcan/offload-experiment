#!/usr/bin/env python3
"""CDFs of offload time and size, plus a p50/p90/p99 table.

Usage:
    ../.venv/bin/python plot_cdf.py
    ../.venv/bin/python plot_cdf.py --summary ../peak/results/summary.json --out ../peak/plots

Writes OUT/offload_cdf.png (left: demote and promote time, right: offloaded size)
and OUT/percentiles.md. Size is demote.written_to_disk_mib (swap + cache snapshot),
the same x axis as offload_times.png. Each task has equal weight.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
QUANTILES = (0.5, 0.9, 0.99)
# Same colors as offload_times.png, so demote/promote read the same in both plots.
BLUE, ORANGE = "#0072B2", "#D55E00"

# (label, unit, getter); the rows of percentiles.md
METRICS = [
    ("Container memory at freeze", "MiB", lambda t: t["memory_at_freeze_mib"]["total"]),
    ("Offloaded (swap + cache snapshot)", "MiB", lambda t: t["demote"]["written_to_disk_mib"]),
    ("Demote time", "ms", lambda t: t["demote"]["total_ms"]),
    ("Promote time", "ms", lambda t: t["promote"]["total_ms"]),
    ("Recovered fraction", "", lambda t: t["recovery"]["fraction_of_total"]),
    ("Not recovered", "MiB", lambda t: t["recovery"]["not_restored_mib"]["total"]),
]


def cdf(values):
    x = np.sort(np.asarray(values, dtype=float))
    return x, np.arange(1, len(x) + 1) / len(x) * 100


def percentile_table(tasks) -> str:
    head = "| Metric | p50 | p90 | p99 | min | max |\n|---|---|---|---|---|---|\n"
    rows = []
    for label, unit, get in METRICS:
        v = np.array([get(t) for t in tasks], dtype=float)
        cells = [*np.quantile(v, QUANTILES), v.min(), v.max()]
        fmt = "{:.2f}" if not unit else "{:.1f}"
        name = f"{label} ({unit})" if unit else label
        rows.append(f"| {name} | " + " | ".join(fmt.format(c) for c in cells) + " |")
    return head + "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "results" / "summary.json")
    parser.add_argument("--out", type=Path, default=ROOT / "plots")
    args = parser.parse_args()

    tasks = json.loads(args.summary.read_text())["tasks"]
    if not tasks:
        parser.error("The summary contains no tasks to plot.")
    args.out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140,
                         "savefig.dpi": 180, "axes.titleweight": "bold"})
    fig, (t_ax, s_ax) = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")

    for ax in (t_ax, s_ax):
        for q in QUANTILES:
            ax.axhline(q * 100, color="gray", linestyle=":", linewidth=1, alpha=.6)
        ax.set(ylim=(0, 101), ylabel="Tasks at or below (%)")
        ax.grid(alpha=.2)

    # Time: one line per direction, percentile dots on each line, values in the legend.
    for label, key, color in [("Demote", "demote", BLUE), ("Promote", "promote", ORANGE)]:
        x, y = cdf([t[key]["total_ms"] for t in tasks])
        qs = np.quantile(x, QUANTILES)
        legend = f"{label}: p50 {qs[0]:.0f} · p90 {qs[1]:.0f} · p99 {qs[2]:.0f} ms"
        t_ax.step(x, y, where="post", color=color, linewidth=2, label=legend)
        t_ax.plot(qs, np.array(QUANTILES) * 100, "o", color=color, markersize=7,
                  markeredgecolor="white", markeredgewidth=1.5, zorder=3)
    t_ax.set(title=f"Offload time CDF · {len(tasks)} tasks", xlabel="Time (ms)")
    t_ax.legend(loc="lower right", frameon=False)

    # Size: one line, same percentile style as the lazy-promote slowdown CDF.
    x, y = cdf([t["demote"]["written_to_disk_mib"] for t in tasks])
    s_ax.step(x, y, where="post", color="#176b93", linewidth=2)
    for q, color in [(0.5, "#618264"), (.9, "#d28a21"), (.99, "#b04759")]:
        v = np.quantile(x, q)
        s_ax.axvline(v, color=color, linestyle="--", alpha=.7, label=f"p{q*100:g}: {v:.1f} MiB")
    s_ax.set(title=f"Offloaded size CDF · {len(tasks)} tasks",
             xlabel="Offloaded size: swap + cache snapshot (MiB)")
    s_ax.legend(loc="lower right")

    for ax in (t_ax, s_ax):
        ax.set_xlim(left=0)
    fig.supxlabel("Each task has equal weight; dotted lines mark p50 / p90 / p99", fontsize=10)
    fig.savefig(args.out / "offload_cdf.png", bbox_inches="tight")
    plt.close(fig)

    table = percentile_table(tasks)
    (args.out / "percentiles.md").write_text(f"Source: {args.summary}\n\n{table}")
    print(table)
    print(f"Saved {args.out / 'offload_cdf.png'} and {args.out / 'percentiles.md'}")


if __name__ == "__main__":
    main()
