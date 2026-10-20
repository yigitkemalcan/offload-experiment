#!/usr/bin/env python3
"""Demote time and promote time by mode, in two separate figures, each bar split into its parts.
DABStep's plot_promote_modes.py (../../../offload-experiment-dabstep/end-offload/promote-modes/), copied, without its
parallel-all mode (not run here).

    ../../.venv/bin/python plot_promote_modes.py                         # results-snapshot/ -> plots-snapshot/
    ../../.venv/bin/python plot_promote_modes.py results-no-snapshot/<run> --out plots-no-snapshot

Reads replay_promote_modes.py's trial.json files (good trials only, median per mode; modes serial and parallel-swap)
and writes, for the snapshot variant:
  demote_modes.png   one bar per mode: the page cache copied into the snapshot, the agent's memory paged out by
                     threads (parallel modes: the swap writes are started and the pages leave the process), then the
                     squeeze (in serial, the agent's memory written to swap; the page cache dropped; the wait for
                     the swap writes; also the flush, a few ms)
  promote_modes.png  one bar per mode: the agent's memory read back from swap, then the page cache read from the
                     snapshot; the label has the swap-in rate (swapped bytes / the first part)
For the no-snapshot variant (results-no-snapshot/) the page cache is dropped by the squeeze and re-read from the
container's files instead.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
MIB = 2**20
SURFACE, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#898781"
LABELS = {  # variant -> (demote labels, promote labels)
    "snapshot": ({"serial": "serial\nsnapshot write and squeeze, one thread",
                  "parallel-swap": "parallel swap\nmemory paged out in {t} threads"},
                 {"serial": "serial\none read at a time, one snapshot pass",
                  "parallel-swap": "parallel swap\nswap reads in {t} threads"}),
    "no-snapshot": ({"serial": "serial\none thread (the squeeze)",
                     "parallel-swap": "parallel swap\nmemory paged out in {t} threads"},
                    {"serial": "serial\none read at a time",
                     "parallel-swap": "parallel swap\nswap reads in {t} threads"}),
}


def medians(trials: pd.DataFrame) -> pd.DataFrame:
    ok = trials[trials.error.isna() & (trials.faithful == True)]  # noqa: E712
    ok = ok.assign(**{c: ok[c].fillna(0.0) if c in ok else 0.0
                      for c in ("demote_pageout_s", "demote_cache_write_s", "cache_snapshot_bytes")})
    return ok.groupby("mode", sort=False).agg(
        trials=("repeat", "size"), task=("task_id", "first"), swapped=("swapped_bytes", "median"),
        threads=("promote_threads", "max"), demote_s=("demote_s", "median"),
        usage=("freeze_usage_in_bytes", "median"), page_cache=("freeze_cache", "median"),
        recorded_cache=("container_cache_pages", lambda c: c.median() * 4096),
        pageout_s=("demote_pageout_s", "median"),
        snapshot_write_s=("demote_cache_write_s", "median"), snapshot_bytes=("cache_snapshot_bytes", "median"),
        process_s=("promote_process_s", "median"), cache_s=("promote_cache_s", "median"),
        promote_s=("promote_s", "median"))


def bars(ax, y, segments, total, note, limit):
    """segments: [(width, color)] stacked from 0; the total and a note after the bar."""
    left = 0.0
    for width, color in segments:
        if width <= 0:
            continue
        ax.barh(y, width, left=left, color=color, height=0.55, edgecolor=SURFACE, linewidth=2)
        if width > 0.1 * limit:
            ax.text(left + width / 2, y, f"{width:.2f} s", ha="center", va="center", fontsize=8.5, color="white")
        left += width
    ax.text(total + 0.012 * limit, y, f"{total:.2f} s{note}", va="center", fontsize=9, color=INK)


def style(ax, title, limit, legend):
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, limit)
    ax.set_xlabel("Time (s)", color=INK2, fontsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="x", colors=INK2, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, loc="left", color=INK, fontsize=10.5, pad=52)
    ax.legend([Patch(color=c) for c, _ in legend], [n for _, n in legend], loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=1, frameon=False, fontsize=8, borderaxespad=0.2)


def figure(m: pd.DataFrame, rows: dict, title: str, legend: list, out: Path, name: str):
    """One horizontal stacked bar per mode. rows: mode -> (label, [(seconds, color)], total, note)."""
    fig, ax = plt.subplots(figsize=(9, 1.6 + 0.9 * len(rows)), facecolor=SURFACE)
    limit = max(total for _, _, total, _ in rows.values()) * 1.35
    y = list(range(len(rows)))[::-1]
    for yi, (label, segments, total, note) in zip(y, rows.values()):
        bars(ax, yi, segments, total, note, limit)
    ax.set_yticks(y, [label for label, _, _, _ in rows.values()], fontsize=9, color=INK)
    n = int(m.trials.min())
    style(ax, f"{title}\n{'median of ' + str(n) + ' trials' if n > 1 else 'one trial'} per mode", limit, legend)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out / name}")


def plot(m: pd.DataFrame, variant: str, out: Path):
    demote_label, promote_label = LABELS[variant]
    m = m.loc[[mode for mode in demote_label if mode in m.index]]
    t = int(m.threads.max()) if m.threads.notna().any() else 8
    task, usage, swapped = m.task.iloc[0], m.usage.median() / MIB, m.swapped.median() / MIB
    if variant == "snapshot":
        snap = m.snapshot_bytes.median() / MIB
        demote = {mode: [(r.snapshot_write_s, ORANGE), (r.pageout_s, BLUE),
                         (r.demote_s - r.snapshot_write_s - r.pageout_s, GRAY)] for mode, r in m.iterrows()}
        demote_title = (f"Demote, task {task}: {usage:.0f} MiB in the container, {snap:.0f} MiB page cache to the "
                        f"snapshot + {swapped:.0f} MiB agent memory to swap")
        demote_legend = [(ORANGE, "Page cache copied into the snapshot file"),
                         (BLUE, "Agent's memory paged out by threads (swap writes start)"),
                         (GRAY, "Squeeze: agent's memory to swap (serial) or its swap writes completed; "
                                "page cache dropped")]
        promote_title = (f"Promote, task {task}: {swapped:.0f} MiB agent memory from swap + {snap:.0f} MiB page "
                         f"cache read from the snapshot")
        promote_legend = [(BLUE, "Agent's memory read back from swap"), (ORANGE, "Page cache read from the snapshot")]
    else:
        demote = {mode: [(r.pageout_s, BLUE), (r.demote_s - r.pageout_s, GRAY)] for mode, r in m.iterrows()}
        demote_title = (f"Demote, task {task}: {usage:.0f} MiB in the container, {swapped:.0f} MiB agent memory to "
                        f"swap + {m.page_cache.median() / MIB:.0f} MiB page cache dropped")
        demote_legend = [(BLUE, "Agent's memory paged out by threads (swap writes start)"),
                         (GRAY, "Squeeze: the rest (in serial, all of it), page cache dropped; swap writes completed")]
        promote_title = (f"Promote, task {task}: {swapped:.0f} MiB agent memory from swap + "
                         f"{m.recorded_cache.median() / MIB:.0f} MiB page cache re-read from files")
        promote_legend = [(BLUE, "Agent's memory read back from swap"), (ORANGE, "Page cache re-read from files")]
    figure(m, {mode: (demote_label[mode].format(t=t), demote[mode], r.demote_s, "") for mode, r in m.iterrows()},
           demote_title, demote_legend, out, "demote_modes.png")
    figure(m, {mode: (promote_label[mode].format(t=t), [(r.process_s, BLUE), (r.cache_s, ORANGE)], r.promote_s,
                      f"  (swap-in {r.swapped / MIB / r.process_s:.0f} MiB/s)") for mode, r in m.iterrows()},
           promote_title, promote_legend, out, "promote_modes.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path, nargs="?", help="<results>/<run> (default: the one in results-snapshot/)")
    ap.add_argument("--out", type=Path, default=HERE / "plots-snapshot")
    args = ap.parse_args()
    results = args.results or next(p for p in sorted((HERE / "results-snapshot").iterdir()) if p.is_dir())
    # every finished trial's own record (trials.csv is only rewritten at the end of a run)
    trials = pd.DataFrame([json.loads(p.read_text()) for p in sorted(results.glob("*/*/trial.json"))])
    trials["task_id"] = trials.task_id.astype(str)
    if "error" not in trials:
        trials["error"] = None
    variant = "snapshot" if (trials.get("variant") == "snapshot-promote").any() else "no-snapshot"
    plot(medians(trials), variant, args.out)


if __name__ == "__main__":
    main()
