#!/usr/bin/env python3
"""Plots for replay_reuse.py results.

    ../.venv/bin/python plot_reuse.py results-v5-full [--out DIR] [--unit calls|steps]

All finished tasks under the results directory (every run) go into the same plots, written to
<results>/plots by default:
reuse_per_task.png       one bar per task: % of page-uses that were reuse, split by page source
reuse_per_task.csv       the numbers behind it
reuse_distance_cdf.png   CDF of reuse distance over all reused page-uses, overall and per source

Sources: environment = file cache outside /testbed (conda, Python, libraries, tools, config);
task repo = file cache under /testbed; other = process (anonymous) memory, /tmp, shm and the rest.
accessed.npz does not store each page's file name, so for the distance CDF file pages are matched to
their file through objects.csv statistics; page-uses that cannot be matched unambiguously are left out
of the per-source lines (the share is printed) but are included in the overall line.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_reuse import keepalive_keys, keepalive_objects

ANON = np.uint64(1 << 63)
SOURCES = ("environment", "task repo", "other")
LABEL = {"environment": "Environment (conda, Python, libs, tools)", "task repo": "Task repo (/testbed)",
         "other": "Other (process memory, /tmp, shm)"}
COLOR = {"environment": "#2a78d6", "task repo": "#eb6834", "other": "#1baf7a", "all": "#0b0b0b"}
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def source(kind: str, path: str) -> str:
    if kind == "anon" or kind == "shm" or path.startswith("/tmp"):
        return "other"
    if path.startswith("/testbed"):
        return "task repo"
    return "environment"


def tasks(results: Path) -> list[Path]:
    return [p.parent for p in sorted(results.glob("**/trial.json")) if (p.parent / "objects.csv").exists()]


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out / name}.png")


# ---------------------------------------------------------------- reuse per task

def per_task(dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for d in dirs:
        o = keepalive_objects(d, pd.read_csv(d / "objects.csv"))
        o["source"] = [source(k, str(p)) for k, p in zip(o.kind, o.object)]
        total = o.page_calls.sum()
        reused = o.groupby("source").reused_page_calls.sum()
        rows.append({"task": d.name, "page_uses": total,
                     **{s: 100 * reused.get(s, 0) / total for s in SOURCES}})
    df = pd.DataFrame(rows)
    df["total"] = df[list(SOURCES)].sum(axis=1)
    return df.sort_values("total").reset_index(drop=True)


def plot_per_task(df: pd.DataFrame, out: Path):
    n = len(df)
    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.13 * n), facecolor=SURFACE)
    left = np.zeros(n)
    for s in SOURCES:
        ax.barh(np.arange(n), df[s], left=left, height=0.8, color=COLOR[s], label=LABEL[s],
                edgecolor=SURFACE, linewidth=0.6)
        left += df[s].to_numpy()
    median = df.total.median()
    ax.axvline(median, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(median, n - 0.3, f" median {median:.0f}%", color=INK2, fontsize=7, va="bottom")
    ax.set_yticks(np.arange(n), df.task.str.split("__").str[-1], fontsize=5)
    ax.set_ylim(-0.6, n + 0.4)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 10))
    ax.xaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("Reused page-uses (% of all page-uses in the task)")
    ax.set_title(f"Page reuse per task (n = {n}), split by page source", loc="left", fontsize=10)
    ax.legend(loc="lower right", frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=7, labelcolor=INK2,
              title="Reused pages from", title_fontsize=7)
    style(ax)
    save(fig, out, "reuse_per_task")


# ---------------------------------------------------------------- reuse distance CDF

def distances(dirs: list[Path], unit: str) -> tuple[dict, float]:
    """Reuse distances of every reused page-use, overall and per source; share of file reuses left unmatched."""
    out = {s: [] for s in ("all", *SOURCES)}
    unmatched = file_reused = 0
    sig = ["pages_accessed", "page_calls", "reused_page_calls"]
    for d in dirs:
        z = np.load(d / "accessed.npz")
        key, reused, dist = z["key"].astype(np.uint64), z["reused"], z[f"reuse_distance_{unit}"]
        reused = reused & ~keepalive_keys(d, key)  # harness process, touched by our freeze/thaw
        out["all"].append(dist[reused])
        anon = (key & ANON) != 0
        out["other"].append(dist[reused & anon])
        # file pages: match each file id to its objects.csv row by (pages, page-uses, reused page-uses)
        f = ~anon
        fid = (key[f] >> np.uint64(32)).astype(np.int64)
        g = pd.DataFrame({"fid": fid, "key": key[f], "reused": reused[f]}).groupby("fid").agg(
            pages_accessed=("key", "nunique"), page_calls=("key", "size"), reused_page_calls=("reused", "sum"))
        o = pd.read_csv(d / "objects.csv")
        o = o[o.kind != "anon"].copy()
        o["source"] = [source(k, str(p)) for k, p in zip(o.kind, o.object)]
        s = o.groupby(sig).source.agg(lambda x: x.iloc[0] if x.nunique() == 1 else None)
        src = g.join(s, on=sig).source.reindex(fid).to_numpy()
        r = reused[f]
        file_reused += r.sum()
        unmatched += (r & pd.isna(src)).sum()
        for name in SOURCES:
            out[name].append(dist[f][r & (src == name)])
    return {k: np.concatenate(v) for k, v in out.items()}, unmatched / max(file_reused, 1)


def plot_cdf(dist: dict, unit: str, out: Path):
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    top = int(dist["all"].max())
    for name in ("all", *SOURCES):
        d = dist[name]
        if not len(d):
            continue
        x = np.arange(1, top + 1)
        y = np.cumsum(np.bincount(d, minlength=top + 1)[1:]) / len(d)
        label = "All reused pages" if name == "all" else LABEL[name]
        ax.step(x, y, where="post", color=COLOR[name], linewidth=2.4 if name == "all" else 1.8,
                linestyle="-" if name != "all" else (0, (4, 2)), label=f"{label} (n = {len(d):,})", zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(1, top)
    low = np.floor(min(np.mean(d == 1) for d in dist.values() if len(d)) * 10) / 10  # CDF starts at its first step
    ax.set_ylim(low, 1.01)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.grid(True, which="major", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"Reuse distance (tool {unit[:-1]}s since the previous use, log scale)" if unit == "calls"
                  else "Reuse distance (agent steps since the previous use, log scale)")
    ax.set_ylabel(f"Cumulative share of reused page-uses (axis starts at {low:.0%})")
    ax.set_title("CDF of reuse distance", loc="left", fontsize=10)
    ax.legend(loc="lower right", frameon=False, fontsize=7, labelcolor=INK2)
    style(ax)
    save(fig, out, f"reuse_distance_cdf{'' if unit == 'calls' else '_steps'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--unit", choices=("calls", "steps"), default="calls")
    args = ap.parse_args()
    out = args.out or args.results / "plots"
    out.mkdir(parents=True, exist_ok=True)
    dirs = tasks(args.results)
    if not dirs:
        raise SystemExit(f"no finished tasks under {args.results}")

    df = per_task(dirs)
    df.to_csv(out / "reuse_per_task.csv", index=False)
    plot_per_task(df, out)

    dist, unmatched = distances(dirs, args.unit)
    plot_cdf(dist, args.unit, out)
    for name, d in dist.items():
        if len(d):
            q = np.percentile(d, [50, 90, 99])
            print(f"  {name:12s} n={len(d):>10,}  at 1: {np.mean(d == 1):6.1%}  median {q[0]:.0f}  p90 {q[1]:.0f}  "
                  f"p99 {q[2]:.0f}  max {d.max()}")
    print(f"  file reuses not matched to a source (in 'all' only): {unmatched:.1%}")


if __name__ == "__main__":
    main()
