#!/usr/bin/env python3
"""Container memory of the same SWE-bench tasks with the agent outside the container (the original 4-worker run:
the container holds only the commands) and inside it (agent-in-container replays with --private: the container
holds the agent and the commands, on its own files).

Memory is Docker's definition, as in the original runs' plots: cgroup usage minus inactive file cache. Each pair
is compared on the coarser of the two sampling intervals (1 s for the capacity smoke run, 0.1 s otherwise),
taking the sample nearest each grid point, so a finer sampler does not catch peaks the other cannot.
Time is seconds since the task started (agent ready); the inside agent's first, cold run is used.

  ../../mini-swe-agent/.venv/bin/python plot_outside_vs_inside.py      # -> plots/container_memory_outside_vs_inside.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
ORIGINAL = PROJECT / "swebench-runs" / "qwen-run-20261017T063628Z-XIPXDG"
SMOKE = PROJECT / "evaluation" / "results" / "smoke-3agents-20261019T041807Z"  # recorded inference times
TRIAL = ROOT / "results" / "qwen-run-20261017T063628Z-XIPXDG-c2-private-20261019T031129Z"  # replies at once
MIB = 2**20
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
OUTSIDE, INSIDE = "#2a78d6", "#eb6834"  # reference palette slots 1-2 (validated)


def samples(run: Path) -> pd.DataFrame:
    c = pd.read_csv(run / "system_metrics" / "container_samples.csv")
    cache = c.mem_total_inactive_file_bytes
    c["mib"] = (c.mem_usage_bytes - cache.where(cache < c.mem_usage_bytes, 0)) / MIB
    return c


def window(trajectory: Path, run_samples: pd.DataFrame) -> dict:
    t = json.loads(trajectory.read_text())
    timing = t["info"]["task_timing"]
    s = run_samples[run_samples.container_id == t["info"]["container_id"]]
    s = s[(s.t_mono_ns >= timing["start_mono_ns"]) & (s.t_mono_ns <= timing["end_mono_ns"])]
    return {"t": s.t_mono_ns.to_numpy(), "mib": s.mib.to_numpy(), "ready": timing["environment_ready_mono_ns"],
            "start": timing["start_mono_ns"], "end": timing["end_mono_ns"]}


def on_grid(w: dict, interval: float) -> tuple[np.ndarray, np.ndarray]:
    """Nearest sample to every interval from the first sample to the end (seconds since ready, MiB)."""
    grid = np.arange(w["t"][0], w["end"], interval * 1e9)
    idx = np.clip(np.searchsorted(w["t"], grid), 1, len(w["t"]) - 1)
    idx = np.where(np.abs(w["t"][idx - 1] - grid) <= np.abs(w["t"][idx] - grid), idx - 1, idx)
    return (grid - w["ready"]) / 1e9, w["mib"][idx]


def interval(run: Path) -> float:
    return json.loads((run / "system_metrics" / "meta.json").read_text())["cpu_interval_s"]


def main():
    orig_samples = samples(ORIGINAL)
    smoke_samples, trial_samples = samples(SMOKE), samples(TRIAL)
    smoke_agents = {json.loads((SMOKE / "agents" / a / "cycle-000.traj.json").read_text())["instance_id"]: a
                    for a in sorted(p.name for p in (SMOKE / "agents").iterdir())}
    pairs = []  # (task, outside window, inside window, grid interval, timing replayed)
    for task, agent in smoke_agents.items():
        inside = window(SMOKE / "agents" / agent / "cycle-000.traj.json", smoke_samples)
        pairs.append((task, window(ORIGINAL / "results" / task / f"{task}.traj.json", orig_samples), inside,
                      max(interval(ORIGINAL), interval(SMOKE)), True))
    for task in json.loads((TRIAL / "launch.json").read_text())["tasks"]:
        pairs.append((task, window(ORIGINAL / "results" / task / f"{task}.traj.json", orig_samples),
                      window(TRIAL / "results" / task / f"{task}.traj.json", trial_samples),
                      max(interval(ORIGINAL), interval(TRIAL)), False))

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK_2, "axes.labelcolor": INK_2, "xtick.color": INK_2,
                         "ytick.color": INK_2, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(13, 8.2))
    grid = fig.add_gridspec(2, 6, height_ratios=[1.15, 1], hspace=0.45, wspace=0.9, top=0.84)
    timed = [p for p in pairs if p[4]]
    ymax = max(max(on_grid(o, iv)[1].max(), on_grid(i, iv)[1].max()) for _, o, i, iv, _ in timed) * 1.12
    for k, (task, outside, inside, iv, _) in enumerate(timed):
        ax = fig.add_subplot(grid[0, 2 * k:2 * k + 2])
        for w, color, label in ((outside, OUTSIDE, "Agent outside (original run)"), (inside, INSIDE, "Agent inside (replay)")):
            x, y = on_grid(w, iv)
            ax.plot(x, y, color=color, lw=2, label=label, solid_capstyle="round")
        ax.set_title(task.split("__")[1], color=INK, fontsize=10, loc="left")
        ax.set_ylim(0, ymax)
        ax.set_xlim(left=min((outside["t"][0] - outside["ready"]) / 1e9, (inside["t"][0] - inside["ready"]) / 1e9))
        ax.set_xlabel("Seconds since the task started")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.axvline(0, color=GRID, lw=1, zorder=0)
        if k == 0:
            ax.set_ylabel("Container memory (MiB)")
            x, y = on_grid(inside, iv)
            ax.annotate("agent inside", (x[-1], y[-1]), xytext=(-4, 6), textcoords="offset points", ha="right",
                        color=INK_2, fontsize=8)
            x, y = on_grid(outside, iv)
            ax.annotate("agent outside", (x[-1], y[-1]), xytext=(-4, 6), textcoords="offset points", ha="right",
                        color=INK_2, fontsize=8)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.06, 0.93), ncol=2, frameon=False, fontsize=9)

    rows = []
    for task, outside, inside, iv, replayed_timing in pairs:
        (_, yo), (_, yi) = on_grid(outside, iv), on_grid(inside, iv)
        rows.append({"task": task.split("__")[1] + ("" if replayed_timing else " *"), "outside_mean": yo.mean(),
                     "inside_mean": yi.mean(), "outside_peak": yo.max(), "inside_peak": yi.max(), "interval_s": iv})
    table = pd.DataFrame(rows)
    (ROOT / "plots").mkdir(exist_ok=True)
    table.round(1).to_csv(ROOT / "plots" / "container_memory_outside_vs_inside.csv", index=False)
    for k, (stat, title) in enumerate((("mean", "Average over the task"), ("peak", "Peak"))):
        ax = fig.add_subplot(grid[1, 3 * k:3 * k + 3])
        y = np.arange(len(table))[::-1]
        ax.hlines(y, table[f"outside_{stat}"], table[f"inside_{stat}"], color=GRID, lw=2, zorder=1)
        for col, color in ((f"outside_{stat}", OUTSIDE), (f"inside_{stat}", INSIDE)):
            ax.scatter(table[col], y, s=64, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
            for xi, yi in zip(table[col], y):
                ax.annotate(f"{xi:.0f}", (xi, yi), xytext=(0, 8), textcoords="offset points", ha="center",
                            color=INK_2, fontsize=8)
        ax.set_yticks(y, table.task)
        ax.set_ylim(-0.7, len(table) - 0.3)
        ax.set_xlim(0, max(table.inside_peak.max(), table.outside_peak.max()) * 1.12)
        ax.set_xlabel("Container memory (MiB)")
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.tick_params(axis="y", length=0)

    fig.suptitle("Container memory: agent outside the container vs inside it", x=0.06, y=1.0, ha="left",
                 fontsize=13, color=INK, fontweight="bold")
    fig.text(0.06, 0.945, "Same tasks. Outside: the original 4-worker run; the container holds only the commands. "
             "Inside: replays with the agent in its container, on its own files\n(--private), its first (cold) run; "
             "the agent process itself is about 200 MiB (RSS 217 MiB). Memory: cgroup usage minus inactive file cache "
             "(Docker's definition).", fontsize=8.5, color=INK_2, ha="left", va="bottom")
    fig.text(0.06, -0.01, "Top: the three tasks replayed with the recorded inference times (same timeline as the original), "
             "sampled every 1 s; before 0 s the agent is starting inside its container.\n"
             "* Replayed with model replies returned at once: 5-6 min compressed to 10-15 s, so compare the peak, not "
             "the average; sampled every 0.1 s. Each pair is compared on its coarser interval.",
             fontsize=8, color=INK_2, ha="left", va="top")
    out = ROOT / "plots" / "container_memory_outside_vs_inside.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(table.round(1).to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
