#!/usr/bin/env python3
"""Audit lazy replay results and save static PNG plots plus ranked commands.

Run with ../../mini-swe-agent/.venv/bin/python plot_lazy.py
Outputs default to the plots subdirectory of the supplied results directory.
"""
import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_lazy import load, pair
from replay_lazy import ro

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="?", type=Path, default=ROOT / "results")
    ap.add_argument("--runs", type=Path, default=ROOT.parents[1] / "swebench-runs")
    ap.add_argument("--out", type=Path, help="output directory (default: <results>/plots)")
    args = ap.parse_args()
    if args.out is None:
        args.out = args.results / "plots"
    args.out.mkdir(parents=True, exist_ok=True)
    raw = load(args.results)
    calls = pair(raw)
    metadata, issues, trial_count = [], [], 0
    for run in sorted(args.results.glob("qwen*")):
        for task in ro.load_tasks(args.runs / run.name, 99, 1.0):
            for i, a in enumerate(task["actions"], 1):
                metadata.append(dict(run=run.name, task_id=task["task_id"], call_index=i,
                                     command=a["command"], original_outcome=a["orig_outcome"],
                                     original_exit_status=task["orig_exit_status"]))
            for mode in ("regular", "lazy"):
                d = run / task["task_id"] / mode
                try:
                    trial = json.loads((d / "trial.json").read_text())
                    frame = pd.read_csv(d / "calls.csv")
                    trial_count += 1
                    if trial.get("error"):
                        issues.append(f"{d}: {trial['error']}")
                    if frame.call_index.tolist() != list(range(1, len(task["actions"]) + 1)):
                        issues.append(f"{d}: missing, repeated or out-of-order calls")
                    if (frame.call_s <= 0).any() or frame.call_s.isna().any():
                        issues.append(f"{d}: invalid durations")
                except (OSError, ValueError) as e:
                    issues.append(f"{d}: {e}")
    calls = calls.merge(pd.DataFrame(metadata), on=["run", "task_id", "call_index"],
                        how="left", validate="one_to_one")
    calls["original_behavior_differs"] = (
        (calls.orig_returncode.notna() & (calls.regular_returncode != calls.orig_returncode))
        | (calls.outcome != calls.original_outcome))
    calls["lazy_read_mib"] = calls.call_blkio_read_bytes / 2**20
    calls["residual_user_mib"] = calls.residual_user_bytes / 2**20
    ok = calls[calls.same_behavior].copy()
    ranked = ok.sort_values("slowdown", ascending=False)
    ranked.to_csv(args.out / "ranked_calls.csv", index=False)
    ok.sort_values("added_s", ascending=False).to_csv(args.out / "ranked_added_time.csv", index=False)
    calls[calls.original_behavior_differs].to_csv(args.out / "original_mismatches.csv", index=False)

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140,
                         "savefig.dpi": 180, "axes.titleweight": "bold"})
    def save(fig, name):
        fig.savefig(args.out / f"{name}.png", bbox_inches="tight")
        plt.close(fig)

    x = np.sort(ok.slowdown.to_numpy())
    y = np.arange(1, len(x) + 1) / len(x) * 100
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax in axes:
        ax.step(x, y, where="post", color="#176b93", linewidth=2)
        ax.set(xlabel="Slowdown (lazy / regular)", ylabel="Tool calls at or below slowdown (%)")
        ax.grid(alpha=.2)
    axes[0].set(title=f"Per-call slowdown CDF · {len(ok):,} paired calls", ylim=(0, 100))
    axes[1].set(title="Upper tail · slowest 10%", ylim=(90, 100), xlim=(np.quantile(x, .9), x.max()*1.02))
    for q, color in [(0.5, "#618264"), (.9, "#d28a21"), (.99, "#b04759")]:
        v = np.quantile(x, q)
        axes[0].axvline(v, color=color, linestyle="--", alpha=.7, label=f"p{q*100:g}: {v:.3f}×")
        if q >= .9:
            axes[1].plot(v, q*100, "o", color=color)
            axes[1].annotate(f" p{q*100:g}: {v:.3f}×", (v, q*100),
                             xytext=(5, 12 if q == .9 else -18), textcoords="offset points")
    axes[0].legend(loc="lower right")
    fig.supxlabel("Each call has equal weight; demotion and freeze/thaw time excluded", fontsize=10)
    save(fig, "slowdown_cdf")

    top = ranked.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(13, 8), layout="constrained")
    positions = np.arange(len(top))
    ax.barh(positions-.18, top.regular_s, height=.34, label="Regular", color="#176b93")
    ax.barh(positions+.18, top.lazy_s, height=.34, label="Lazy", color="#dc7841")
    labels = [f"{r.task_id} · call {r.call_index}\n{textwrap.shorten(' '.join(r.command.split()), width=65)}"
              for r in top.itertuples()]
    ax.set_yticks(positions, labels, fontsize=8)
    for i, r in enumerate(top.itertuples()):
        ax.text(r.lazy_s+.015, i, f"{r.slowdown:.2f}×  (+{r.added_s*1000:.0f} ms)", va="center", fontsize=9)
    ax.set(xlabel="Call duration (seconds)", title="15 largest per-call slowdown ratios", xlim=(0, top.lazy_s.max()*1.3))
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=.2)
    save(fig, "worst_calls")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    axes[0].scatter(ok.regular_s, ok.lazy_s, s=9, alpha=.25, color="#176b93")
    lo, hi = min(ok.regular_s.min(), ok.lazy_s.min()), max(ok.regular_s.max(), ok.lazy_s.max())
    axes[0].plot([lo, hi], [lo, hi], "--", color="gray", label="Equal duration")
    axes[0].set(xscale="log", yscale="log", xlabel="Regular seconds", ylabel="Lazy seconds", title="Call durations")
    axes[0].legend()
    axes[1].scatter(ok.lazy_read_mib, ok.added_s*1000, s=9, alpha=.25, color="#176b93")
    axes[1].set(xlabel="Lazy disk reads (MiB)", ylabel="Added call time (ms)", title="Disk reads and added latency")
    for ax in axes: ax.grid(alpha=.2)
    save(fig, "latency_diagnostics")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    tasks = ok.groupby(["run", "task_id"])[["regular_s", "lazy_s"]].sum()
    ts = np.sort((tasks.lazy_s/tasks.regular_s).to_numpy())
    axes[0].step(ts, np.arange(1,len(ts)+1)/len(ts)*100, where="post", color="#176b93")
    axes[0].set(xlabel="Task slowdown (sum lazy / sum regular)", ylabel="Tasks at or below (%)", title="Task-level slowdown CDF")
    axes[1].step(np.sort(ok.residual_user_mib), y, where="post", color="#b04759")
    axes[1].set(xlabel="User memory remaining after demotion (MiB)", ylabel="Calls at or below (%)", xscale="log", title="Demotion residual CDF")
    for ax in axes: ax.grid(alpha=.2)
    save(fig, "task_and_demotion")

    # Use every completed lazy boundary, independently of call-behavior filtering.
    memory = raw[raw["mode"] == "lazy"].copy()
    memory["demoted_mib"] = memory.demoted_bytes / 2**20
    memory["before_mib"] = memory.usage_before_demote_bytes / 2**20
    memory["user_residual_pct_of_total_before"] = (
        100 * memory.residual_user_bytes / memory.usage_before_demote_bytes)
    memory["total_residual_pct_of_total_before"] = (
        100 * memory.residual_bytes / memory.usage_before_demote_bytes)
    memory[["run", "task_id", "call_index", "before_mib", "demoted_mib",
            "residual_user_bytes", "residual_kmem_bytes",
            "user_residual_pct_of_total_before", "total_residual_pct_of_total_before"]].to_csv(
                args.out / "demoted_memory.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    my = np.arange(1, len(memory) + 1) / len(memory) * 100
    for col, label, color in [("demoted_mib", "Demoted", "#176b93"),
                              ("before_mib", "Total before demotion", "#999999")]:
        axes[0].step(np.sort(memory[col]), my, where="post", label=label, color=color)
    axes[0].set(title=f"Demoted memory CDF · {len(memory):,} boundaries",
                xlabel="Memory (MiB)", ylabel="Demotions at or below (%)", ylim=(0, 100), xlim=(0, None))
    for col, label, color in [
            ("user_residual_pct_of_total_before", "Estimated user residual", "#b04759"),
            ("total_residual_pct_of_total_before", "Total residual (including kernel)", "#999999")]:
        axes[1].step(np.sort(memory[col]), my, where="post", label=label, color=color)
    axes[1].set(title="How much remained in RAM?",
                xlabel="Residual / total memory before demotion (%)",
                ylabel="Demotions at or below (%)", ylim=(0, 100), xlim=(0, None))
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(loc="lower right", fontsize=9)
    fig.supxlabel("Demoted = reduction in resident usage, not disk-write volume. User residual is estimated; denominator includes kernel memory.", fontsize=9)
    save(fig, "demoted_memory_cdf")
    memory_summary = {
        col: {"median": float(memory[col].median()), "p90": float(memory[col].quantile(.9)),
              "p99": float(memory[col].quantile(.99)), "max": float(memory[col].max())}
        for col in ["demoted_mib", "user_residual_pct_of_total_before", "total_residual_pct_of_total_before"]}
    (args.out / "demoted_memory_summary.json").write_text(json.dumps(memory_summary, indent=2)+"\n")

    summary = dict(trials=trial_count, expected_calls=len(metadata), paired_calls=len(calls),
                   compared_calls=len(ok), issues=issues,
                   behavior_differences_between_modes=int((~calls.same_behavior).sum()),
                   original_behavior_differences=int(calls.original_behavior_differs.sum()),
                   original_returncode_unavailable=int(calls.orig_returncode.isna().sum()),
                   median_slowdown=float(ok.slowdown.median()), p99_slowdown=float(ok.slowdown.quantile(.99)),
                   max_slowdown=float(ok.slowdown.max()), overall_slowdown=float(ok.lazy_s.sum()/ok.regular_s.sum()),
                   residual_user_mib_max=float(ok.residual_user_mib.max()))
    (args.out / "validation.json").write_text(json.dumps(summary, indent=2)+"\n")
    report = ["# Lazy-promotion results", "", "```json", json.dumps(summary, indent=2), "```", "",
              "All plots compare lazy / regular replay time. Demotion and boundary time are excluded.",
              "Matching return codes/outcomes do not prove identical output. One trial per mode; regular always ran first.",
              "Missing original return codes are not counted as mismatches.", "", "## Worst 15 calls", ""]
    for r in ranked.head(15).itertuples():
        report += [f"### {r.task_id}, call {r.call_index} (step {r.step}, action {r.action})",
                   f"Run: {r.run}. {r.slowdown:.3f}×; {r.regular_s:.4f} → {r.lazy_s:.4f} s; +{r.added_s*1000:.1f} ms. Lazy reads: {r.lazy_read_mib:.1f} MiB.",
                   "", "```bash", r.command, "```", ""]
    report += ["## Calls differing from the original recording", ""]
    for r in calls[calls.original_behavior_differs].itertuples():
        report += [f"- {r.task_id}, call {r.call_index}: original rc={r.orig_returncode:g}; replay rc={r.regular_returncode:g}."]
    (args.out / "report.md").write_text("\n".join(report)+"\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote plots, ranked CSVs, validation.json and report.md to {args.out}")


if __name__ == "__main__":
    main()
