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

--agent-inside: the same experiment with the task's agent inside its container (run_trial_inside()), instead of
one controller on the host running the commands with `docker exec`. The container runs
../agent-in-container/agent.py --hold: mini-swe-agent's DefaultAgent replaying the recorded model replies (no
inference delay, as the docker-exec replay) and running the commands itself. As in the DABStep experiment
(../../offload-experiment-dabstep/end-offload/replay_end.py): when the agent has finished, it writes done.json and
waits, with its conversation in memory; the container is frozen and measured as above, so the agent's memory (its
process, and the page cache of its runtime, ../agent-in-container/runtime, mounted read-only) is part of what is
demoted and promoted; then it is thawed and released: the agent saves its trajectory and exits (it must have
survived), and the replay is compared with the original run. The page-cache walk covers the runtime too. Results go to results-agent-inside/ (or
<variant>/results-agent-inside/). Two differences besides the agent:
  - the default measurement is the variants' "baseline" (offload_variants.py: demote and promote as
    measure_offload(), but promote faults back only the process pages that belong to no file). The agent maps
    tens of MiB of libraries; measure_offload() would fault those through /proc/<pid>/mem, reading 4 MiB around
    each fault from md0.
  - on a tool timeout the agent kills the command's process group (mini-swe-agent's local environment);
    `docker exec` left it running. No command of the serial runs timed out.

--parallel-swap THREADS (the variants only): the processes' memory is paged out to swap and read back by THREADS
threads (offload_variants.py demote_mode pageout-threads, promote_mode threads; promote-modes/'s parallel-swap mode);
the page cache is moved as before. Results go to <variant>/results[-agent-inside]-parallel-swap/.

Usage (from this directory):
  sudo ../.venv/bin/python replay_end.py ../../swebench-runs/<run>            # -> results/<run>/
  sudo ../.venv/bin/python replay_end.py ../../swebench-runs/<run> --tasks django__django-10097
  sudo ../.venv/bin/python replay_end.py ../../swebench-runs/<run> --agent-inside   # -> results-agent-inside/<run>/
  sudo ../.venv/bin/python snapshot-promote/replay_snapshot_promote.py ../../swebench-runs/<run> --agent-inside \
      --parallel-swap 8                       # -> snapshot-promote/results-agent-inside-parallel-swap/<run>/
"""

import argparse
import functools
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT.parent / "peak"))
import replay_offload as ro  # noqa: E402
from replay_offload import now  # noqa: E402
from offload_variants import SUMMARY_EXTRA, measure_offload_variant  # noqa: E402

AGENT_IN_CONTAINER = ROOT.parent / "agent-in-container"
sys.path.insert(0, str(AGENT_IN_CONTAINER))
import replay_in_container as ric  # noqa: E402

STEP_LIMIT_STATUS = "LimitsExceeded"


def load_tasks(run_dir: Path) -> list[dict]:
    """Peak's task list (p99 and its 80% threshold kept for comparison) plus the original step count."""
    steps = pd.read_csv(run_dir / "task_metrics" / "tasks.csv").set_index("task_id").steps
    tasks = ro.load_tasks(run_dir, 99, 0.8)
    for task in tasks:
        task["orig_steps"] = int(steps[task["task_id"]])
        task["hit_step_limit"] = task["orig_exit_status"] == STEP_LIMIT_STATUS
        task["trace"] = (run_dir / "results" / task["task_id"] / f"{task['task_id']}.traj.json").resolve()
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


def run_trial(task: dict, out_dir: Path, args, measure=ro.measure_offload) -> dict:
    """measure has measure_offload()'s signature and runs while the container is frozen."""
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
        measure(cg, cid, upper, roots, {}, sampler, rec, out_dir)
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


# ---------------------------------------------------------------- the agent inside the container

AGENT_MOUNTS = {"/opt/mswea-agent": ric.RUNTIME, "/opt/mswea-replay/agent.py": AGENT_IN_CONTAINER / "agent.py"}
BASELINE_INSIDE = functools.partial(measure_offload_variant, write_snapshot=True, promote_from_snapshot=False)
# extra trials.csv columns of --agent-inside
SUMMARY_INSIDE = ["replay_exit_status", "faithful", "faithful_ignoring_order", "container_exit_code", "agent_ready_s",
                  "end_marker_to_freeze_s"]


def run_trial_inside(task: dict, out_dir: Path, args, measure=BASELINE_INSIDE) -> dict:
    """run_trial() with the task's agent inside its container (--agent-inside, see the module doc)."""
    env = task["env"]
    name = f"end-offload-in-{task['task_id'].replace('__', '-').lower()[:40]}-{os.getpid()}"
    rec = {k: task[k] for k in ("task_id", "orig_exit_status", "orig_steps", "hit_step_limit",
                                "orig_p_usage_bytes", "threshold_bytes")}
    rec |= {"image": env["image"], "n_actions": len(task["actions"]), "triggered": False, "agent_inside": True}
    agent_dir = out_dir / "agent"
    shutil.rmtree(agent_dir, ignore_errors=True)  # from an interrupted trial: its done.json would end this one at once
    agent_dir.mkdir()
    mounts = AGENT_MOUNTS | {"/opt/mswea-replay/trace.json": task["trace"]}
    start = now()
    cid = ro.docker("run", "-d", "--name", name, "-w", env["cwd"], "--init",
                    *(x for dst, src in mounts.items() for x in ("-v", f"{src}:{dst}:ro")), "-v", f"{agent_dir}:/output",
                    env["image"], "/opt/mswea-agent/python/bin/python3", "-I", "-B", "-u", "/opt/mswea-replay/agent.py",
                    "--trace", "/opt/mswea-replay/trace.json", "--output", "/output", "--start-ns", str(start),
                    "--sample-interval", "0", "--hold").stdout.strip()
    rec["container_id"] = cid
    cg = ro.Cgroup(cid)
    samplers = []
    try:
        info = json.loads(ro.docker("inspect", cid).stdout)[0]
        init_pid, upper = info["State"]["Pid"], Path(info["GraphDriver"]["Data"]["UpperDir"])
        roots = [Path(info["GraphDriver"]["Data"]["MergedDir"]), Path(f"/proc/{init_pid}/root/dev/shm"), ric.RUNTIME]
        rec["usage_at_start_bytes"] = cg.usage()

        replay_sampler = ro.Sampler(cid, args.replay_sample_interval, out_dir / "replay_samples.csv")
        samplers.append(replay_sampler)
        replay_sampler.start()
        done, deadline = agent_dir / "done.json", time.monotonic() + args.task_timeout
        while not done.exists():
            if not cg.mem.exists():
                raise RuntimeError("the container exited before the end of the task: see agent/container.log")
            if time.monotonic() > deadline:
                raise TimeoutError(f"the task did not end within {args.task_timeout} s")
            time.sleep(0.005)

        # ---- end of the task, the agent waiting: freeze, then demote / promote
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
        measure(cg, cid, upper, roots, {}, sampler, rec, out_dir)

        # ---- the agent must survive: thaw, release it (it saves its trajectory and exits)
        cg.thaw()
        sampler.phase = "thawed"
        (agent_dir / "release").touch()
        try:
            rec["container_exit_code"] = int(subprocess.run(["docker", "wait", cid], capture_output=True, text=True,
                                                             timeout=120).stdout.strip() or -1)
        except (subprocess.TimeoutExpired, ValueError):
            rec["container_exit_code"] = None
    finally:
        try:
            cg.thaw()
        except OSError:
            pass
        for s in samplers:
            s.finish()
        logs = subprocess.run(["docker", "logs", cid], capture_output=True, text=True)
        (agent_dir / "container.log").write_text(logs.stdout + logs.stderr)
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True)
        cg.close()
    return rec | agent_run(task, out_dir, start, t_end)


def agent_run(task: dict, out_dir: Path, start_ns: int, end_ns: int) -> dict:
    """From the agent's trajectory: actions.csv (as run_trial()'s) and the comparison with the original run."""
    done = json.loads((out_dir / "agent" / "done.json").read_text())
    timing = {"agent_ready_s": (done["ready_mono_ns"] - start_ns) / 1e9,
              "replay_s": (done["end_mono_ns"] - done["ready_mono_ns"]) / 1e9,
              "end_marker_to_freeze_s": (end_ns - done["end_mono_ns"]) / 1e9}
    path = out_dir / "agent" / task["trace"].name
    if not path.exists():  # the agent did not save it after the release
        return timing | {"replay_trajectory": False, "faithful": False}
    replay = json.loads(path.read_text())
    timings = [t for s in replay["step_timings"] if s["inference"]["outcome"] != "FormatError" for t in s["tools"]]
    orig = task["actions"]
    log = [{"index": i, "step": a["step"], "action": a["action"], "start_ns": t["start_mono_ns"],
            "end_ns": t["end_mono_ns"], "returncode": a["orig_returncode"],
            "orig_returncode": orig[i]["orig_returncode"] if i < len(orig) else None, "command": a["command"]}
           for i, (a, t) in enumerate(zip(ro.load_actions(replay), timings))]
    pd.DataFrame(log, dtype=object).to_csv(out_dir / "actions.csv", index=False)  # return codes stay integers
    rec = timing | {"actions_replayed": len(log)}
    if log:
        rec.update(step=log[-1]["step"], action=log[-1]["action"], action_index=log[-1]["index"],
                   command=log[-1]["command"])
    return rec | ric.compare_trajectories(json.loads(task["trace"].read_text()), replay)


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


# extra trials.csv columns of --parallel-swap
SUMMARY_PARALLEL_SWAP = ["demote_mode", "demote_threads", "demote_pageout_s", "demote_pageout_bytes",
                         "demote_pageout_errors", "promote_mode", "promote_threads"]


def add_agent_inside_args(ap: argparse.ArgumentParser):
    ap.add_argument("--agent-inside", action="store_true", help="the task's agent runs inside its container")
    ap.add_argument("--replay-sample-interval", type=float, default=0.5, help="while replaying (s)")
    ap.add_argument("--task-timeout", type=float, default=4 * 3600,
                    help="--agent-inside: give up on a task whose agent has not finished after this many seconds")


def check_agent_runtime():
    if not (ric.RUNTIME / "python" / "bin" / "python3").exists():
        sys.exit(f"no agent runtime: run `bash {AGENT_IN_CONTAINER / 'setup_runtime.sh'}` first")


def main(measure=ro.measure_offload, out=ROOT / "results", doc=__doc__, summary=SUMMARY):
    """Also the driver of the variants in no-snapshot/ and snapshot-promote/ (other measure, out, columns)."""
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, help=f"default: {out.name}/, with --agent-inside {out.name}-agent-inside/, "
                                             f"with --parallel-swap also -parallel-swap")
    ap.add_argument("--sample-interval", type=float, default=0.005, help="during the measurement (s)")
    ap.add_argument("--tasks", nargs="*", help="only these task ids")
    ap.add_argument("--parallel-swap", type=int, default=0, metavar="THREADS",
                    help="no-snapshot/ and snapshot-promote/: page the processes' memory out to swap and read it back "
                         "with THREADS threads (promote-modes/'s parallel-swap); results go to <out>-parallel-swap/")
    add_agent_inside_args(ap)
    args = ap.parse_args()
    if args.parallel_swap:
        if not (isinstance(measure, functools.partial) and measure.func is measure_offload_variant):
            ap.error("--parallel-swap is for the variants: run no-snapshot/ or snapshot-promote/")
        measure = functools.partial(measure, demote_mode="pageout-threads", promote_mode="threads",
                                    demote_threads=args.parallel_swap, promote_threads=args.parallel_swap)
        summary = summary + SUMMARY_PARALLEL_SWAP
    if os.geteuid() != 0:
        sys.exit("must run as root (cgroup writes, /proc/<pid>/mem, overlay dirs)")
    if not Path("/proc/swaps").read_text().strip().count("\n"):
        sys.exit("no swap enabled: run setup_swap.sh first")
    trial = run_trial
    if args.agent_inside:
        check_agent_runtime()
        trial = run_trial_inside
        if measure is ro.measure_offload:
            measure, summary = BASELINE_INSIDE, summary + SUMMARY_EXTRA
        summary = summary + SUMMARY_INSIDE
        out = out.with_name(f"{out.name}-agent-inside")
    if args.parallel_swap:
        out = out.with_name(f"{out.name}-parallel-swap")
    sys.setswitchinterval(0.0005)
    out_root = (args.out or out) / args.run_dir.name
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
            rec = trial(task, out_dir, args, measure)
        except Exception as e:  # keep going; record the failure
            rec = {"task_id": task["task_id"], "triggered": False, "error": repr(e)}
        (out_dir / "trial.json").write_text(json.dumps(rec, indent=1, default=str))
        mib = lambda b: f"{b / 2**20:.1f}MiB" if isinstance(b, (int, float)) else "-"
        print(f"[{n}/{len(tasks)}] {task['task_id']}: status={task['orig_exit_status']} "
              f"steps={task['orig_steps']} actions={rec.get('actions_replayed')} "
              f"size={mib(rec.get('freeze_usage_in_bytes'))} cache={mib(rec.get('freeze_cache'))} "
              f"demote={rec.get('demote_s')} promote={rec.get('promote_s')} err={rec.get('error')}", flush=True)
    rows = [json.loads(p.read_text()) for p in sorted(out_root.glob("*/trial.json"))]
    pd.DataFrame(rows).reindex(columns=summary).to_csv(out_root / "trials.csv", index=False)
    print(f"wrote {out_root / 'trials.csv'}", flush=True)


if __name__ == "__main__":
    main()
