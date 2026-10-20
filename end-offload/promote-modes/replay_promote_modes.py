#!/usr/bin/env python3
"""SWE-bench end-of-task offload, agent inside the container, with different ways of moving the container's memory
out and back, on one task. DABStep's promote-modes (../../../offload-experiment-dabstep/end-offload/promote-modes/)
on the mini-swe-agent replay.

Each trial is ../replay_end.py --agent-inside (run_trial_inside()): the task's agent replays the recorded run inside
its container, and when it has finished, the container is frozen, demoted, promoted, thawed and the agent released.

Two variants, each with two modes; only how the agent's memory goes to swap and comes back differs between the modes
(offload_variants.py demote_mode, promote_mode):

  --variant snapshot (the default): the snapshot-promote variant (../snapshot-promote/)
  mode           demote                                        promote
  serial         flush, snapshot write, squeeze                swap-in (one /proc/<pid>/mem read after another),
                 (the existing demote)                         snapshot read (the existing promote)
  parallel-swap  flush, snapshot write, the process pages      swap-in by --threads threads, snapshot read
                 paged out to swap by --threads threads
                 (MADV_PAGEOUT), squeeze

  --variant no-snapshot: the no-snapshot variant (../no-snapshot/)
  serial         flush, squeeze                                swap-in one read after another, page cache re-read
                                                               from the container's files
  parallel-swap  flush, page-out by the threads, squeeze       swap-in by the threads, page cache re-read from the
                                                               files

The page cache (snapshot write and read, file re-read, drop) is moved by one thread in every mode: DABStep's third
mode, parallel-all, which also moves it with threads, is not run here.

Each trial is a fresh container: host cache dropped, the task replayed to its end, frozen, demoted, promoted
with one mode, thawed, and the replay checked against the original (the agent must survive). Modes are
interleaved (serial, parallel-swap, serial, ...) so that drift on the shared host hits all of them
alike. Serial is run again here as the control; the variant's 100-task --agent-inside run is printed next to it for
reference when it exists.

The default task, django__django-11099 (run qwen-run-20260923T053154Z-PF12hU), is a median one in the agent-inside
compression run: 400 MiB in the container at the end (177 MiB the agent's process, 197 MiB page cache), and a
3.6 s replay.

Usage (from this directory; root, swap on):
  sudo ../../.venv/bin/python replay_promote_modes.py ../../../swebench-runs/qwen-run-20260923T053154Z-PF12hU
  sudo ../../.venv/bin/python replay_promote_modes.py ../../../swebench-runs/<run> --variant no-snapshot
  sudo ../../.venv/bin/python replay_promote_modes.py ../../../swebench-runs/<run> --summary-only

Writes <out>/<run>/<task>/<mode>-<k>/ (run_trial_inside()'s trial files), <out>/<run>/trials.csv and summary.csv;
<out> is results-snapshot/ or results-no-snapshot/ unless --out is given.
"""

import argparse
import functools
import json
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent))
import replay_end  # noqa: E402

# measure_offload_variant()'s settings per mode, the same in both variants
MODES = {"serial": {"demote_mode": "squeeze", "promote_mode": "serial"},
         "parallel-swap": {"demote_mode": "pageout-threads", "promote_mode": "threads"}}
VARIANTS = ("snapshot", "no-snapshot")
REFERENCE = {"snapshot": HERE.parent / "snapshot-promote" / "results-agent-inside",  # the 100-task runs (serial)
             "no-snapshot": HERE.parent / "no-snapshot" / "results-agent-inside"}
OUT = {"snapshot": HERE / "results-snapshot", "no-snapshot": HERE / "results-no-snapshot"}


def settings(variant: str, mode: str, threads: int) -> dict:
    """Keyword arguments of measure_offload_variant() for one trial."""
    snapshot = variant == "snapshot"
    return {"write_snapshot": snapshot, "promote_from_snapshot": snapshot, "promote_threads": threads,
            "demote_threads": threads, **MODES[mode]}
COLUMNS = ["task_id", "mode", "repeat", "faithful", "container_exit_code", "freeze_usage_in_bytes", "freeze_rss",
           "freeze_cache", "swapped_bytes", "demote_s", "demote_flush_s", "demote_cache_write_s",
           "cache_snapshot_bytes", "cache_hardlink_duplicate_pages", "demote_reclaim_s", "demote_threads",
           "demote_pageout_s", "demote_pageout_bytes", "demote_pageout_errors", "residual_bytes", "promote_s",
           "promote_process_s", "promote_threads", "promote_cache_s", "promote_files_s", "promote_snapshot_s",
           "promote_snapshot_bytes_read", "promote_read_errors", "promote_recovered_fraction",
           "swap_after_promote_bytes", "error"]


def trials(tasks: list[str], modes: list[str], repeats: int) -> list[tuple[str, str, int]]:
    """(task, mode, repeat), modes interleaved within each repeat."""
    return [(task, mode, k) for task in tasks for k in range(1, repeats + 1) for mode in modes]


def summarize(rows: pd.DataFrame, reference: pd.DataFrame | None) -> pd.DataFrame:
    """Per task and mode: median demote time, process-promote time and swap-in rate, and total promote time."""
    ok = rows[rows.error.isna() & (rows.faithful == True)]  # noqa: E712
    ok = ok.assign(swap_in_mib_s=ok.swapped_bytes / 2**20 / ok.promote_process_s)
    out = ok.groupby(["task_id", "mode"]).agg(
        trials=("repeat", "size"), swapped_mib=("swapped_bytes", lambda s: s.median() / 2**20),
        demote_s=("demote_s", "median"), process_s=("promote_process_s", "median"),
        swap_in_mib_s=("swap_in_mib_s", "median"), promote_s=("promote_s", "median"),
        recovered=("promote_recovered_fraction", "median")).reset_index()
    if reference is not None:
        ref = reference[reference.task_id.astype(str).isin(out.task_id.astype(str).unique())]
        out = pd.concat([out, pd.DataFrame({
            "task_id": ref.task_id.astype(str), "mode": "serial (100-task run)", "trials": 1,
            "swapped_mib": ref.swapped_bytes / 2**20, "demote_s": ref.demote_s, "process_s": ref.promote_process_s,
            "swap_in_mib_s": ref.swapped_bytes / 2**20 / ref.promote_process_s, "promote_s": ref.promote_s,
            "recovered": ref.promote_recovered_fraction})], ignore_index=True)
    return out.round(3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--tasks", nargs="+", default=["django__django-11099"],
                    help="task ids (default: django__django-11099, a median task of PF12hU)")
    ap.add_argument("--variant", choices=VARIANTS, default="snapshot")
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=8, help="threads for the parallel modes")
    ap.add_argument("--out", type=Path, help="default: results-snapshot/ or results-no-snapshot/")
    ap.add_argument("--summary-only", action="store_true", help="only summarize existing trials")
    ap.add_argument("--sample-interval", type=float, default=0.005, help="during the measurement (s)")
    ap.add_argument("--replay-sample-interval", type=float, default=0.5, help="while replaying (s)")
    ap.add_argument("--task-timeout", type=float, default=900, help="seconds to wait for a task's end")
    args = ap.parse_args()
    out_root = (args.out or OUT[args.variant]) / args.run_dir.name
    if not args.summary_only:
        if os.geteuid() != 0:
            sys.exit("must run as root (cgroup writes, /proc/<pid>/mem, drop_caches)")
        if not Path("/proc/swaps").read_text().strip().count("\n"):
            sys.exit("no swap enabled: run ../../setup_swap.sh first")
        replay_end.check_agent_runtime()
        sys.setswitchinterval(0.0005)
        tasks = {t["task_id"]: t for t in replay_end.load_tasks(args.run_dir) if t["task_id"] in args.tasks}
        if missing := set(args.tasks) - set(tasks):
            sys.exit(f"not in {args.run_dir.name}: {sorted(missing)}")
        plan = trials(args.tasks, args.modes, args.repeats)
        for n, (task_id, mode, k) in enumerate(plan, 1):
            out_dir = out_root / task_id / f"{mode}-{k}"
            if (out_dir / "trial.json").exists():
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            measure = functools.partial(replay_end.measure_offload_variant,
                                        **settings(args.variant, mode, args.threads))
            try:
                replay_end.ro.drop_caches()
                rec = replay_end.run_trial_inside(tasks[task_id], out_dir, args, measure)
            except Exception as e:  # keep going; record the failure
                rec = {"task_id": task_id, "triggered": False, "error": repr(e)}
            rec |= {"mode": mode, "repeat": k, "experiment_variant": args.variant}
            (out_dir / "trial.json").write_text(json.dumps(rec, indent=1, default=str))
            print(f"[{n}/{len(plan)}] {task_id} {mode}-{k}: faithful={rec.get('faithful')} "
                  f"swapped={(rec.get('swapped_bytes') or 0) / 2**20:.1f}MiB "
                  f"process={rec.get('promote_process_s')} promote={rec.get('promote_s')} "
                  f"recovered={rec.get('promote_recovered_fraction')} err={rec.get('error')}", flush=True)
    rows = pd.DataFrame([json.loads(p.read_text()) for p in sorted(out_root.glob("*/*/trial.json"))])
    if rows.empty:
        sys.exit(f"no trials in {out_root}")
    rows = rows.reindex(columns=COLUMNS)
    rows["task_id"] = rows.task_id.astype(str)
    rows.to_csv(out_root / "trials.csv", index=False)
    ref_csv = REFERENCE[args.variant] / args.run_dir.name / "trials.csv"
    summary = summarize(rows, pd.read_csv(ref_csv) if ref_csv.exists() else None)
    summary.to_csv(out_root / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"wrote {out_root / 'trials.csv'} and summary.csv", flush=True)


if __name__ == "__main__":
    main()
