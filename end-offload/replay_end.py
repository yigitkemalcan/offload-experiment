#!/usr/bin/env python3
"""Replay logged mini-SWE-agent tool calls in fresh Docker containers and measure swap-based offload at
the end of each task.

Same measurement as ../peak/replay_offload.py, different trigger. Per task: start the task image exactly
like the original run and replay every executed tool call in order, to the last one (tasks that stopped
at the step limit are replayed up to their last step, like any other). Where the original run would now
destroy the container, it is frozen instead (cgroup freezer), and its footprint is demoted to disk and
promoted back with peak's measure_offload(), unchanged:

  demote  = fsync the writable layer, write every page-cache page the container loaded to a snapshot
            file, then shrink memory.limit_in_bytes to ~0 (anon/shmem -> swap, page cache dropped).
  promote = lift the limit and fault back every page that was resident before demotion.

The tool-call processes have exited by then, so what is left is the page cache and shared files the
task accumulated, plus the anonymous memory of whatever is still alive: the container's `sleep` and any
process a tool call left behind (background jobs, calls killed by the tool timeout).

The host page cache is dropped before every task, as in the peak experiment. Must run as root on the
cgroup v1 host, with swap on (../setup_swap.sh). All times are CLOCK_MONOTONIC.

Usage (from this directory):
  sudo ../.venv/bin/python replay_end.py ../../swebench-runs/<run>            # -> results/<run>/
  sudo ../.venv/bin/python replay_end.py ../../swebench-runs/<run> --tasks django__django-10097
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT.parent / "peak"))
import replay_offload as ro  # noqa: E402
from replay_offload import now  # noqa: E402

STEP_LIMIT_STATUS = "LimitsExceeded"


def load_tasks(run_dir: Path) -> list[dict]:
    """Peak's task list (p99 and its 80% threshold kept for comparison) plus the original step count."""
    steps = pd.read_csv(run_dir / "task_metrics" / "tasks.csv").set_index("task_id").steps
    tasks = ro.load_tasks(run_dir, 99, 0.8)
    for task in tasks:
        task["orig_steps"] = int(steps[task["task_id"]])
        task["hit_step_limit"] = task["orig_exit_status"] == STEP_LIMIT_STATUS
    return tasks


def replay(task: dict, cid: str) -> list[dict]:
    env = task["env"]
    exec_env = [x for k, v in env["env"].items() for x in ("-e", f"{k}={v}")]
    log = []
    for i, a in enumerate(task["actions"]):
        cmd = ["docker", "exec", "-w", env["cwd"], *exec_env, cid, *env["interpreter"], a["command"]]
        start = now()
        try:
            rc = subprocess.run(cmd, capture_output=True, timeout=env["timeout"]).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        log.append({"index": i, "step": a["step"], "action": a["action"], "start_ns": start, "end_ns": now(),
                    "returncode": rc, "orig_returncode": a["orig_returncode"], "command": a["command"]})
    return log


def run_trial(task: dict, out_dir: Path, args) -> dict:
    env = task["env"]
    name = f"end-offload-{task['task_id'].replace('__', '-').lower()[:40]}-{os.getpid()}"
    rec = {k: task[k] for k in ("task_id", "orig_exit_status", "orig_steps", "hit_step_limit",
                                "orig_p_usage_bytes", "threshold_bytes")}
    rec |= {"image": env["image"], "n_actions": len(task["actions"]), "triggered": False}
    cid = ro.docker("run", "-d", "--name", name, "-w", env["cwd"], "--rm", env["image"],
                    "sleep", env["container_timeout"]).stdout.strip()
    rec["container_id"] = cid
    cg = ro.Cgroup(cid)
    samplers = []
    try:
        info = json.loads(ro.docker("inspect", cid).stdout)[0]
        init_pid, upper = info["State"]["Pid"], Path(info["GraphDriver"]["Data"]["UpperDir"])
        roots = [Path(info["GraphDriver"]["Data"]["MergedDir"]), Path(f"/proc/{init_pid}/root/dev/shm")]
        rec["usage_at_start_bytes"] = cg.usage()

        # coarse while the task runs (minutes), fine during the measurement (as in the peak experiment)
        replay_sampler = ro.Sampler(cid, args.replay_sample_interval, out_dir / "replay_samples.csv")
        samplers.append(replay_sampler)
        replay_sampler.start()
        t0 = now()
        log = replay(task, cid)
        rec["replay_s"] = (now() - t0) / 1e9
        pd.DataFrame(log).to_csv(out_dir / "actions.csv", index=False)
        rec["actions_replayed"] = len(log)
        rec["returncode_mismatches"] = sum(str(l["returncode"]) != str(l["orig_returncode"])
                                           for l in log if l["orig_returncode"] is not None)
        if log:
            rec.update(step=log[-1]["step"], action=log[-1]["action"], action_index=log[-1]["index"],
                       command=log[-1]["command"])

        # ---- end of the task: freeze instead of destroying the container, then demote / promote
        t_end = now()
        cg.freeze()
        rec.update(triggered=True, trigger_source="end", trigger_ns=t_end, usage_at_trigger_bytes=cg.usage())
        rec["frozen_ok"] = cg.wait_frozen()
        rec["freeze_latency_s"] = (now() - t_end) / 1e9
        replay_sampler.stop_event.set()
        rec["peak_usage_during_replay_bytes"] = max((r[6] for r in replay_sampler.rows), default=0)
        sampler = ro.Sampler(cid, args.sample_interval, out_dir / "samples.csv")
        sampler.phase = "frozen"
        samplers.append(sampler)
        sampler.start()
        # empty baseline: the host cache was dropped before this task, so every cached page is the task's
        ro.measure_offload(cg, cid, upper, roots, {}, sampler, rec, out_dir)
    finally:
        try:
            cg.thaw()
        except OSError:
            pass
        for s in samplers:
            s.finish()
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True)
        cg.close()
    return rec


# ---------------------------------------------------------------- driver

SUMMARY = ["task_id", "orig_exit_status", "hit_step_limit", "orig_steps", "n_actions", "actions_replayed",
           "returncode_mismatches", "replay_s", "triggered", "step", "action", "command", "pids",
           "orig_p_usage_bytes", "peak_usage_during_replay_bytes", "usage_at_trigger_bytes",
           "freeze_usage_in_bytes", "freeze_rss", "freeze_cache", "freeze_shmem", "freeze_mapped_file",
           "freeze_dirty", "freeze_kmem_usage_in_bytes", "writable_layer_bytes", "freeze_latency_s",
           "demote_s", "demote_flush_s", "demote_cache_write_s", "demote_reclaim_s", "cache_snapshot_bytes",
           "demoted_bytes", "swapped_bytes",
           "residual_bytes", "residual_kmem_bytes", "cache_pages_still_resident_after_demote",
           "promote_s", "promote_process_s", "promote_cache_s", "promote_bytes_read", "promote_read_errors",
           "usage_after_promote_bytes", "promote_recovered_fraction", "error"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--sample-interval", type=float, default=0.005, help="during the measurement (s)")
    ap.add_argument("--replay-sample-interval", type=float, default=0.5, help="while replaying (s)")
    ap.add_argument("--tasks", nargs="*", help="only these task ids")
    args = ap.parse_args()
    if os.geteuid() != 0:
        sys.exit("must run as root (cgroup writes, /proc/<pid>/mem, overlay dirs)")
    if not Path("/proc/swaps").read_text().strip().count("\n"):
        sys.exit("no swap enabled: run setup_swap.sh first")
    sys.setswitchinterval(0.0005)
    out_root = args.out / args.run_dir.name
    out_root.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(args.run_dir)
    if args.tasks:
        tasks = [t for t in tasks if t["task_id"] in args.tasks]
    print(f"{args.run_dir.name}: {len(tasks)} tasks", flush=True)
    for n, task in enumerate(tasks, 1):
        out_dir = out_root / task["task_id"]
        if (out_dir / "trial.json").exists():
            continue
        out_dir.mkdir(exist_ok=True)
        try:
            ro.drop_caches()
            rec = run_trial(task, out_dir, args)
        except Exception as e:  # keep going; record the failure
            rec = {"task_id": task["task_id"], "triggered": False, "error": repr(e)}
        (out_dir / "trial.json").write_text(json.dumps(rec, indent=1, default=str))
        mib = lambda b: f"{b / 2**20:.1f}MiB" if isinstance(b, (int, float)) else "-"
        print(f"[{n}/{len(tasks)}] {task['task_id']}: status={task['orig_exit_status']} "
              f"steps={task['orig_steps']} actions={rec.get('actions_replayed')} "
              f"size={mib(rec.get('freeze_usage_in_bytes'))} cache={mib(rec.get('freeze_cache'))} "
              f"demote={rec.get('demote_s')} promote={rec.get('promote_s')} err={rec.get('error')}", flush=True)
    rows = [json.loads(p.read_text()) for p in sorted(out_root.glob("*/trial.json"))]
    pd.DataFrame(rows).reindex(columns=SUMMARY).to_csv(out_root / "trials.csv", index=False)
    print(f"wrote {out_root / 'trials.csv'}", flush=True)


if __name__ == "__main__":
    main()
