#!/usr/bin/env python3
"""Summarize replay_end.py results into one JSON file, in the layout of ../peak/summarize_results.py.

Reads every results/<run>/<task>/trial.json and writes:
  summary: mean, min, max and percentiles (p50/p90/p95/p99) of every per-task metric, over all tasks
  tasks:   one record per task (sizes in MiB, times in ms); peak's "trigger" block is replaced by "end"

Usage (from this directory; results/ is root-owned, hence sudo):
  sudo ../.venv/bin/python summarize_end.py                 # results/ -> results/summary.json
  sudo ../.venv/bin/python summarize_end.py RESULTS_DIR     # RESULTS_DIR/summary.json
  (also for no-snapshot/results and snapshot-promote/results)

The summary has the fields ../peak/plot_offload.py reads:
  ../.venv/bin/python ../peak/plot_offload.py --summary results/summary.json --out results/offload_times.png
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT.parent / "peak"))
from summarize_results import mib, ms, numeric_leaves, stats, task_record  # noqa: E402


def end_record(run: str, t: dict) -> dict:
    record = task_record(run, t)
    del record["trigger"]
    end = {
        "orig_exit_status": t["orig_exit_status"],
        "hit_step_limit": t["hit_step_limit"],
        "orig_steps": t["orig_steps"],
        "actions_replayed": t["actions_replayed"],
        "returncode_mismatches": t["returncode_mismatches"],
        "replay_s": round(t["replay_s"], 3),
        "processes_alive": t["pids"],  # the container's sleep + whatever the tool calls left running
        "orig_p99_usage_mib": mib(t["orig_p_usage_bytes"]),
        "peak_usage_during_replay_mib": mib(t["peak_usage_during_replay_bytes"]),
    }
    if "variant" in t:  # offload_variants.py (no-snapshot/, snapshot-promote/)
        end["variant"] = t["variant"]
        record["promote"].update(files_ms=ms(t["promote_files_s"]), snapshot_read_ms=ms(t["promote_snapshot_s"]),
                                 snapshot_read_mib=mib(t["promote_snapshot_bytes_read"]))
        if "snapshot_pages_resident_before_promote" in t:
            record["promote"]["snapshot_pages_cached_before"] = t["snapshot_pages_resident_before_promote"]
    return {"run": run, "task_id": t["task_id"], "end": end,
            **{k: v for k, v in record.items() if k not in ("run", "task_id")}}


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
        tasks.append(end_record(run, trial))

    columns = {}
    for record in tasks:
        for name, value in numeric_leaves({k: v for k, v in record.items() if k not in ("run", "task_id")}):
            columns.setdefault(name, []).append(value)
    checks = {
        "hit_step_limit": sum(r["end"]["hit_step_limit"] for r in tasks),
        "all_actions_returncode_ok": sum(r["end"]["returncode_mismatches"] == 0 for r in tasks),
        "only_sleep_alive": sum(r["end"]["processes_alive"] == 1 for r in tasks),
        "process_memory_fully_back": sum(r["recovery"]["process_memory_fully_back"] for r in tasks),
        "swap_empty_after_promote": sum(r["recovery"]["swap_empty_after_promote"] for r in tasks),
        "promote_read_errors": sum(r["promote"]["read_errors"] > 0 for r in tasks),
        "snapshot_cached_before_promote": sum(r["promote"].get("snapshot_pages_cached_before", 0) > 0 for r in tasks),
    }
    summary = {
        "results_dir": str(args.results.resolve()),
        "runs": sorted({r["run"] for r in tasks}),
        "n_tasks": len(tasks),
        "n_failed": len(errors),
        "units": "sizes in MiB, times in ms (replay_s in s), throughput in MiB/s",
        "checks_task_counts": checks,
        "stats": {name: stats(values) for name, values in columns.items()},
    }
    args.out.write_text(json.dumps({"summary": summary, "failed_tasks": errors, "tasks": tasks}, indent=2))
    print(f"wrote {args.out}: {len(tasks)} tasks, {len(errors)} failed")


if __name__ == "__main__":
    main()
