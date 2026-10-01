#!/usr/bin/env python3
"""Plot replay_compress.py results: the figures of compression_experiment.py, for containers.

Usage (from this directory):
    ../.venv/bin/python plot_compression.py                      # every results/<run>/
    ../.venv/bin/python plot_compression.py results/<run> --primary zstd-3 --net

Reads results/<run>/compression.csv and writes plots/<run>/ (plots-net/<run>/ with --net), plus plots/all/ over
every run given, task indices continuing from one run to the next (the 100-task set of compression_experiment.py).
results/ is written by root, so the plots go beside it rather than into it:
  compress_time.png, decompress_time.png  per-task time for the primary codec, task index on x
  compress_time_lz4.png, decompress_time_lz4.png  additional per-task LZ4 timings
  codec_tradeoff.png                      median ratio against median time, one point per codec
  size_vs_time.png, size_vs_ratio.png     how image size drives time and ratio (fastest, primary, slowest)
Sizes are the dense image (populated pages) in MiB and times are in ms, since container images are tens of
MiB. --net plots times with the codec binary's start-up (timed on an empty input) subtracted.
"""

import argparse
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
MIB = 2**20
# Style of mini-swe-agent-jovan-main/plot_run.py (copied: importing it needs that repo's venv).
SURFACE = "#fcfcfb"
INK, INK_MUTED = "#0b0b0b", "#52514e"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a"]


def figure(title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=13, loc="left", pad=12)
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=10)
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d6d5d1")
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    return fig, ax


def save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  {path}")


def load(run: Path, net: bool, offset: int = 0) -> list[dict]:
    try:
        df = pd.read_csv(run / "compression.csv")
    except pd.errors.EmptyDataError:  # replay_compress.py writes an empty file when no task was timed
        return []
    order = {t: offset + i for i, t in enumerate(dict.fromkeys(df.task_id))}  # task index = order in the run
    suffix = "_net_s" if net else "_s"
    return [{"codec": r.codec, "index": order[r.task_id], "dense_mib": r.dense_bytes / MIB, "ratio": r.ratio,
             "compress_ms": getattr(r, "compress" + suffix) * 1e3,
             "decompress_ms": getattr(r, "decompress" + suffix) * 1e3} for r in df.itertuples()]


def make_figures(rows: list[dict], plots: Path, primary: str, net: bool) -> None:
    plots.mkdir(parents=True, exist_ok=True)
    tag = " (start-up subtracted)" if net else ""
    median = lambda field, codec: statistics.median(r[field] for r in rows if r["codec"] == codec)
    codecs = sorted({r["codec"] for r in rows},
                    key=lambda c: median("dense_mib", c) / max(median("compress_ms", c), 1e-9), reverse=True)

    # 1. Keep the primary-codec plots and always provide named LZ4 plots alongside them.
    detailed = [(primary, ""), ("lz4", "_lz4")]
    for codec, filename_suffix in detailed:
        subset = [r for r in rows if r["codec"] == codec]
        if not subset:
            continue
        for field, title, name in (
            ("compress_ms", f"Time to compress a frozen container ({codec}){tag}", "compress_time"),
            ("decompress_ms", f"Time to decompress a container ({codec}){tag}", "decompress_time"),
        ):
            fig, ax = figure(title, "Task index", "Milliseconds")
            ax.scatter([r["index"] for r in subset], [r[field] for r in subset],
                       s=26, color=PALETTE[0], edgecolor="none", alpha=0.85)
            m = statistics.median(r[field] for r in subset)
            ax.axhline(m, color=PALETTE[1], linewidth=1.5, linestyle="--")
            ax.annotate(f"median {m:.1f} ms", (0, m), xytext=(4, 5), textcoords="offset points",
                        color=INK_MUTED, fontsize=9)
            ax.set_ylim(bottom=0)
            save(fig, plots / f"{name}{filename_suffix}.png")

    # 2. The design space: what each codec buys in capacity and costs in time.
    fig, ax = figure(f"Compression design space for container memory{tag}", "Compression ratio (median)",
                     "Milliseconds per container (median)")
    for slot, field in enumerate(("compress_ms", "decompress_ms")):
        x = [median("ratio", c) for c in codecs]
        y = [median(field, c) for c in codecs]
        ax.plot(x, y, color=PALETTE[slot], linewidth=2, marker="o", markersize=7,
                label="compress" if field == "compress_ms" else "decompress")
        if field == "compress_ms":
            for cx, cy, name in zip(x, y, codecs):
                ax.annotate(name, (cx, cy), xytext=(6, 4), textcoords="offset points", color=INK_MUTED, fontsize=9)
    ax.legend(frameon=False, labelcolor=INK_MUTED, fontsize=9)
    ax.set_yscale("log")
    save(fig, plots / "codec_tradeoff.png")

    # 3. and 4. How container size drives time, and whether it drives ratio.
    shown = list(dict.fromkeys([codecs[0], primary if primary in codecs else codecs[len(codecs) // 2],
                                codecs[-1]]))[:3]
    for field, title, ylabel, name in (
        ("compress_ms", f"Compression time against container size{tag}", "Compress milliseconds",
         "size_vs_time.png"),
        ("ratio", "Compression ratio against container size", "Ratio", "size_vs_ratio.png"),
    ):
        fig, ax = figure(title, "Populated container memory (MiB)", ylabel)
        for slot, codec in enumerate(shown):
            subset = [r for r in rows if r["codec"] == codec]
            ax.scatter([r["dense_mib"] for r in subset], [r[field] for r in subset], s=24,
                       color=PALETTE[slot % len(PALETTE)], edgecolor="none", alpha=0.8, label=codec)
        ax.legend(frameon=False, labelcolor=INK_MUTED, fontsize=9)
        ax.set_ylim(bottom=0)
        save(fig, plots / name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="*", help="results/<run> directories (default: all)")
    ap.add_argument("--primary", default="zstd-3", help="codec for the main per-task scatter plots (named LZ4 plots are also generated)")
    ap.add_argument("--net", action="store_true", help="subtract the codec start-up time")
    args = ap.parse_args()
    runs = args.runs or sorted(p.parent for p in (ROOT / "results").glob("*/compression.csv"))
    if not runs:
        ap.error("no results/<run>/compression.csv found")
    out = ROOT / ("plots-net" if args.net else "plots")
    everything = []
    for run in runs:
        rows = load(run, args.net, offset=len({r["index"] for r in everything}))
        if not rows:
            print(f"{run}: no codec rows, skipped")
            continue
        print(f"{run}: {len({r['index'] for r in rows})} tasks -> {out / run.name}")
        make_figures(rows, out / run.name, args.primary, args.net)
        everything += rows
    if len(runs) > 1 and everything:
        print(f"all runs: {len({r['index'] for r in everything})} tasks -> {out / 'all'}")
        make_figures(everything, out / "all", args.primary, args.net)


if __name__ == "__main__":
    main()
