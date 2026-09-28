#!/usr/bin/env python3
"""Summarize replay_offload.py results into one JSON file.

Reads every results/<run>/<task>/trial.json and writes:
  summary: mean, min, max and percentiles (p50/p90/p95/p99) of every per-task metric, over all tasks
  tasks:   one record per task (sizes in MiB, times in ms)

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
PERCENTILES = (50, 90, 95, 99)


def mib(value):
    return round(value / MIB, 3)


def ms(seconds):
    return round(seconds * 1000, 3)


def task_record(run: str, t: dict) -> dict:
    freeze, promoted = t["mem_at_freeze"], t["mem_after_promote"]
    file_cache = freeze["cache"] - freeze["shmem"]
    promoted_file_cache = promoted["cache"] - promoted["shmem"]
    written = t["swapped_bytes"] + t["cache_snapshot_bytes"]
    return {
        "run": run,
        "task_id": t["task_id"],
        "trigger": {
            "step": t.get("step"),
            "action": t.get("action"),
            "between_tool_calls": t.get("between_tool_calls"),
            "tool_call_running_s": t.get("tool_call_running_s"),
            "threshold_mib": mib(t["threshold_bytes"]),
            "orig_p99_usage_mib": mib(t["orig_p_usage_bytes"]),
            "frozen_call_returncode_ok": None if t.get("between_tool_calls") else
                str(t.get("frozen_call_returncode_after_thaw")) == str(t.get("frozen_call_orig_returncode")),
        },
        "memory_at_freeze_mib": {
            "total": mib(freeze["usage_in_bytes"]),
            "process_anon": mib(freeze["rss"]),
            "file_cache": mib(file_cache),
            # pages found by the file scan; hard-linked files are counted once per path
            "file_cache_recorded": mib(t["container_cache_pages"] * PAGE),
            "shmem": mib(freeze["shmem"]),
            "dirty": mib(freeze["dirty"]),
            "kernel": mib(freeze["kmem_usage_in_bytes"]),
        },
        "demote": {
            "total_ms": ms(t["demote_s"]),
            "flush_dirty_ms": ms(t["demote_flush_s"]),
            "cache_snapshot_write_ms": ms(t["demote_cache_write_s"]),
            "swap_out_ms": ms(t["demote_reclaim_s"]),
            "swapped_mib": mib(t["swapped_bytes"]),
            "cache_snapshot_mib": mib(t["cache_snapshot_bytes"]),
            "written_to_disk_mib": mib(written),
            "throughput_mib_s": round(written / MIB / t["demote_s"], 1) if t["demote_s"] else None,
            "left_in_ram_mib": mib(t["residual_bytes"]),
        },
        "promote": {
            "total_ms": ms(t["promote_s"]),
            "process_memory_ms": ms(t["promote_process_s"]),
            "file_cache_ms": ms(t["promote_cache_s"]),
            "read_mib": mib(t["promote_bytes_read"]),
            "throughput_mib_s": round(t["promote_bytes_read"] / MIB / t["promote_s"], 1) if t["promote_s"] else None,
            "read_errors": t["promote_read_errors"],
        },
        "recovery": {
            "usage_after_promote_mib": mib(t["usage_after_promote_bytes"]),
            "fraction_of_total": round(t["promote_recovered_fraction"], 4),
            # freeze minus after-promote, per kind of memory (dropped clean cache is re-read on demand)
            "not_restored_mib": {
                "total": mib(freeze["usage_in_bytes"] - t["usage_after_promote_bytes"]),
                "process_anon": mib(freeze["rss"] - promoted["rss"]),
                "shmem": mib(freeze["shmem"] - promoted["shmem"]),
                "file_cache": mib(file_cache - promoted_file_cache),
                "kernel": mib(freeze["kmem_usage_in_bytes"] - promoted["kmem_usage_in_bytes"]),
            },
            "process_memory_fully_back": promoted["rss"] >= freeze["rss"],
            "swap_empty_after_promote": t["swap_after_promote_bytes"] == 0,
        },
    }


def numeric_leaves(record: dict, prefix=""):
    for key, value in record.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from numeric_leaves(value, name + ".")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            yield name, value


def stats(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    out = {"n": len(a), "mean": round(float(a.mean()), 3), "min": round(float(a.min()), 3)}
    out.update({f"p{p}": round(float(np.percentile(a, p)), 3) for p in PERCENTILES})
    out["max"] = round(float(a.max()), 3)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path, nargs="?", default=ROOT / "results")
    ap.add_argument("--out", type=Path, help="default: RESULTS/summary.json")
    args = ap.parse_args()
    args.out = args.out or args.results / "summary.json"

    tasks, errors = [], []
    for path in sorted(args.results.glob("*/*/trial.json")):
        trial = json.loads(path.read_text())
        run = path.parent.parent.name
        if trial.get("error") or not trial.get("triggered"):
            errors.append({"run": run, "task_id": trial.get("task_id"), "error": trial.get("error"),
                           "triggered": trial.get("triggered")})
            continue
        tasks.append(task_record(run, trial))

    columns = {}
    for record in tasks:
        for name, value in numeric_leaves({k: v for k, v in record.items() if k not in ("run", "task_id")}):
            if name not in ("trigger.step", "trigger.action"):
                columns.setdefault(name, []).append(value)
    checks = {
        "frozen_call_returncode_ok": sum(r["trigger"]["frozen_call_returncode_ok"] is True for r in tasks),
        "frozen_between_tool_calls": sum(bool(r["trigger"]["between_tool_calls"]) for r in tasks),
        "process_memory_fully_back": sum(r["recovery"]["process_memory_fully_back"] for r in tasks),
        "swap_empty_after_promote": sum(r["recovery"]["swap_empty_after_promote"] for r in tasks),
        "promote_read_errors": sum(r["promote"]["read_errors"] > 0 for r in tasks),
    }
    summary = {
        "results_dir": str(args.results.resolve()),
        "runs": sorted({r["run"] for r in tasks}),
        "n_tasks": len(tasks),
        "n_failed_or_untriggered": len(errors),
        "units": "sizes in MiB, times in ms (tool_call_running_s in s), throughput in MiB/s",
        "checks_task_counts": checks,
        "stats": {name: stats(values) for name, values in columns.items()},
    }
    args.out.write_text(json.dumps({"summary": summary, "failed_tasks": errors, "tasks": tasks}, indent=2))
    print(f"wrote {args.out}: {len(tasks)} tasks, {len(errors)} failed/untriggered")


if __name__ == "__main__":
    main()
