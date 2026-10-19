#!/usr/bin/env python3
"""Replay a recorded mini-SWE-agent SWE-bench run with each task's agent INSIDE its container, N tasks at a time.

In the original run (swebench-runs/<run>, from run-swebench.sh) one controller process on the host drove
every task's container with `docker exec`. Here each task's container runs its own controller (agent.py:
mini-swe-agent's DefaultAgent with the recorded model replies, no vLLM), so the container's cgroup holds the
agent's memory and CPU together with its commands'. Per task:

  docker run -d --init --network none -w /testbed <the task's image> \
      /opt/mswea-agent/python/bin/python3 -I -B -u /opt/mswea-replay/agent.py --trace /opt/mswea-replay/trace.json ...

with the runtime (setup_runtime.sh) mounted read-only at /opt/mswea-agent, agent.py and the original
trajectory at /opt/mswea-replay, and the task's output directory at /output. The container exits when its
agent finishes; its log is saved and it is removed. Tasks run in instance-ID order (the original slice
order), --concurrency at a time: a pool, so the next task starts when one finishes.

By default the tasks share cached files they have in common (image layers of the same base or repository
version, the runtime): a page is charged to the container that read it first, and the host page cache is
not dropped. --private makes them share nothing (private_root.py): before its start, each task gets its
own copy of the image's directories and of the agent's files, written to disk and evicted from the page
cache, bind-mounted over the image's (Docker's init comes from the copy too); the copy is deleted after the
task. Each container then starts cold and every file page it reads is its own. Copying and deleting
(about 20 s and 3.6 GB per task) happen outside the task's timed window, but on the same disk as the
running tasks.

While tasks run, the original runs' collector (mini-swe-agent/characterization/system_metrics.py, CPU side
only, no GPU) samples every task container's cgroup; afterwards analyze_task_metrics.py writes task_metrics/
exactly as for the original run, and every replay is compared with its original (replay_summary.csv:
faithful = same exit status, model calls, commands, command return codes, and command outputs and
submission apart from dates and times, which a replay cannot reproduce: `ls -l`, diff headers). No root needed. All times are
CLOCK_MONOTONIC.

Output: results/<run>-c<N>[-delay][-private]-<UTC time>/
  launch.json, containers.jsonl, replay_summary.csv, collector.log
  results/<task>/<task>.traj.json, agent_process_samples.csv, container.log
  system_metrics/   container_samples.csv, container_events.csv, meta.json
  task_metrics/     tasks.csv, steps.csv, analysis.json

Usage (from this directory, after `bash setup_runtime.sh`):
  ../.venv/bin/python replay_in_container.py ../../swebench-runs/<run> --tasks ID ID --concurrency 2
  ../.venv/bin/python replay_in_container.py ../../swebench-runs/<run> --concurrency 2 --inference-delay
  ../.venv/bin/python replay_in_container.py ../../swebench-runs/<run> --concurrency 2 --private
"""
import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import private_root as pr

ROOT = Path(__file__).resolve().parent
MSWEA = ROOT.parents[1] / "mini-swe-agent"
MSWEA_PY = MSWEA / ".venv" / "bin" / "python"
CHARACTERIZATION = MSWEA / "characterization"
RUNTIME = ROOT / "runtime"
PRIVATE = ROOT / "private"
DOCKER_INIT = Path(os.environ.get("DOCKER_INIT", "/usr/libexec/docker/docker-init"))  # what --init mounts


def docker(*args: str, timeout: float | None = None) -> str:
    return subprocess.run(["docker", *args], check=True, capture_output=True, text=True, timeout=timeout).stdout.strip()


def load_tasks(run_dir: Path, ids: list[str] | None) -> list[dict]:
    traces = {p.parent.name: p for p in sorted((run_dir / "results").glob("*/*.traj.json"))}
    if missing := [i for i in ids or [] if i not in traces]:
        sys.exit(f"Not in {run_dir}: {missing}")
    tasks = []
    for task_id in ids or traces:
        trace = traces[task_id].resolve()
        image = json.loads(trace.read_text())["info"]["config"]["environment"]["image"]
        tasks.append({"task_id": task_id, "trace": trace, "image": image})
    return tasks


def agent_files(task: dict) -> dict[str, Path]:
    """The agent's files by their path in the container: under /opt, so --private copies them too."""
    return {"/opt/mswea-agent": RUNTIME, "/opt/mswea-replay/agent.py": ROOT / "agent.py",
            "/opt/mswea-replay/trace.json": task["trace"], "/opt/mswea-replay/docker-init": DOCKER_INIT}


def run_task(task: dict, name: str, out: Path, args) -> dict:
    task_dir = out / "results" / task["task_id"]
    task_dir.mkdir(parents=True)
    record = {"task_id": task["task_id"], "container": name, "image": task["image"]}
    agent = ["/opt/mswea-agent/python/bin/python3", "-I", "-B", "-u", "/opt/mswea-replay/agent.py",
             "--trace", "/opt/mswea-replay/trace.json", "--output", "/output",
             "--sample-interval", str(args.cpu_interval), "--owner", f"{os.getuid()}:{os.getgid()}",
             *(["--inference-delay"] if args.inference_delay else [])]
    private = PRIVATE / name
    try:
        if args.private:
            t = time.monotonic()
            record["private"] = pr.prepare(task["image"], private, f"agentprep-{name}", agent_files(task))
            record["private"]["prepare_s"] = round(time.monotonic() - t, 3)
            run_args, command = pr.mount_args(private), ["/opt/mswea-replay/docker-init", "--", *agent]
        else:
            files = {dst: src for dst, src in agent_files(task).items() if src != DOCKER_INIT}
            run_args = ["--init", *(x for dst, src in files.items() for x in ("-v", f"{src}:{dst}:ro"))]
            command = agent
        start = time.monotonic_ns()
        cid = docker("run", "-d", "--name", name, "--network", "none", "-w", "/testbed", *run_args,
                     "-v", f"{task_dir}:/output", task["image"], *command, "--start-ns", str(start))
        record |= {"container_id": cid, "start_mono_ns": start}
        try:
            record["exit_code"] = int(docker("wait", cid, timeout=args.task_timeout))
        except subprocess.TimeoutExpired:
            record["error"] = f"still running after {args.task_timeout} s"
        finally:
            record["end_mono_ns"] = time.monotonic_ns()
            logs = subprocess.run(["docker", "logs", cid], capture_output=True, text=True)
            (task_dir / "container.log").write_text(logs.stdout + logs.stderr)
            subprocess.run(["docker", "rm", "-f", cid], capture_output=True)
    finally:
        if private.exists():
            t = time.monotonic()
            pr.remove(task["image"], private, f"agentrm-{name}")
            record.setdefault("private", {})["remove_s"] = round(time.monotonic() - t, 3)
    return record


# Dates and times in command outputs (`ls -l`, `git log`, diff headers): the only thing a faithful replay may change.
TIMESTAMP = re.compile(r"\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d(\.\d+)?( [+-]\d{4})?"
                       r"|\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) +\d+ +(\d\d:\d\d|\d{4})\b")


def same_but_timestamps(a: str, b: str) -> bool:
    return TIMESTAMP.sub("<T>", a) == TIMESTAMP.sub("<T>", b)


def same_lines_any_order(a: str, b: str) -> bool:
    """Same lines apart from timestamps, possibly in another order: `find` and `grep -r` list files in the order
    the filesystem returns them, which differs on another filesystem (ext4 hashes names with a per-filesystem seed)."""
    return sorted(TIMESTAMP.sub("<T>", a).splitlines()) == sorted(TIMESTAMP.sub("<T>", b).splitlines())


def tool_extras(trajectory: dict) -> list[dict]:
    return [m["extra"] for m in trajectory["messages"] if m["role"] == "tool"]


def compare(task: dict, out: Path) -> dict:
    replay_path = out / "results" / task["task_id"] / task["trace"].name
    row = {"task_id": task["task_id"], "replay_trajectory": replay_path.exists()}
    if not row["replay_trajectory"]:
        return row
    return row | compare_trajectories(json.loads(task["trace"].read_text()), json.loads(replay_path.read_text()))


def compare_trajectories(o: dict, r: dict) -> dict:
    """Replay r against original o (see the module docstring for "faithful")."""
    pairs = list(zip(tool_extras(o), tool_extras(r)))
    duration = lambda t: (t["info"]["task_timing"]["end_mono_ns"] - t["info"]["task_timing"]["start_mono_ns"]) / 1e9
    tool_s = lambda t: sum(x["elapsed_s"] for s in t["step_timings"] for x in s["tools"])
    row = {
        "orig_exit_status": o["info"]["exit_status"], "replay_exit_status": r["info"]["exit_status"],
        "orig_api_calls": o["info"]["model_stats"]["api_calls"], "replay_api_calls": r["info"]["model_stats"]["api_calls"],
        "orig_commands": len(tool_extras(o)), "replay_commands": len(tool_extras(r)),
        "returncode_mismatches": sum(a["returncode"] != b["returncode"] for a, b in pairs),
        "output_mismatches": sum(a["raw_output"] != b["raw_output"] for a, b in pairs),
        "output_mismatches_beyond_timestamps": sum(not same_but_timestamps(a["raw_output"], b["raw_output"]) for a, b in pairs),
        "output_mismatches_beyond_order": sum(not same_lines_any_order(a["raw_output"], b["raw_output"]) for a, b in pairs),
        "same_submission": o["info"]["submission"] == r["info"]["submission"],
        "same_submission_beyond_timestamps": same_but_timestamps(o["info"]["submission"], r["info"]["submission"]),
        "orig_task_s": round(duration(o), 3), "replay_task_s": round(duration(r), 3),
        "orig_tool_s": round(tool_s(o), 3), "replay_tool_s": round(tool_s(r), 3),
    }
    same_run = (row["orig_exit_status"] == row["replay_exit_status"] and row["orig_api_calls"] == row["replay_api_calls"]
                and row["orig_commands"] == row["replay_commands"] and not row["returncode_mismatches"]
                and row["same_submission_beyond_timestamps"])
    row["faithful"] = same_run and not row["output_mismatches_beyond_timestamps"]
    row["faithful_ignoring_order"] = same_run and not row["output_mismatches_beyond_order"]
    return row


def start_collector(out: Path, prefix: str, interval: float) -> subprocess.Popen:
    metrics = out / "system_metrics"
    collector = subprocess.Popen(
        [MSWEA_PY, CHARACTERIZATION / "system_metrics.py", metrics, "--gpu-interval", "0",
         "--cpu-interval", str(interval), "--container-name-regex", f"^{re.escape(prefix)}"],
        stdout=(out / "collector.log").open("w"), stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 10
    while not (metrics / "meta.json").exists():
        if collector.poll() is not None or time.monotonic() > deadline:
            sys.exit(f"System metrics collector did not start; see {out / 'collector.log'}")
        time.sleep(0.1)
    return collector


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="original run: swebench-runs/<run>")
    ap.add_argument("--tasks", nargs="+", help="instance IDs (default: every task of the run)")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--inference-delay", action="store_true", help="each model call takes its recorded time")
    ap.add_argument("--private", action="store_true", help="share nothing: a private, uncached copy per task")
    ap.add_argument("--cpu-interval", type=float, default=0.1, help="cgroup and agent memory sampling (s)")
    ap.add_argument("--task-timeout", type=float, default=4 * 3600)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if not (RUNTIME / "python" / "bin" / "python3").exists():
        sys.exit("No runtime: run `bash setup_runtime.sh` first")
    tasks = load_tasks(args.run_dir, args.tasks)
    # fails early if an image is missing (no network in the containers)
    sizes = [int(docker("image", "inspect", "-f", "{{.Size}}", image)) for image in sorted({t["image"] for t in tasks})]
    if args.private:
        runtime_bytes = int(subprocess.run(["du", "-sb", RUNTIME], capture_output=True, text=True).stdout.split()[0])
        need, free = args.concurrency * (max(sizes) + runtime_bytes), shutil.disk_usage(ROOT).free
        if need * 1.2 > free:
            sys.exit(f"--private needs about {need / 1e9:.0f} GB for {args.concurrency} copies; {free / 1e9:.0f} GB free")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "-delay" * args.inference_delay + "-private" * args.private
    out = args.out or ROOT / "results" / f"{args.run_dir.resolve().name}-c{args.concurrency}{suffix}-{stamp}"
    out.mkdir(parents=True)
    prefix = f"agentin-{os.getpid()}-"
    (out / "launch.json").write_text(json.dumps({
        "trace_run": str(args.run_dir.resolve()), "tasks": [t["task_id"] for t in tasks],
        "concurrency": args.concurrency, "inference_delay": args.inference_delay, "private": args.private,
        "cpu_interval_s": args.cpu_interval,
        "container_prefix": prefix, "runtime_mini_swe_agent_commit": (RUNTIME / "mini-swe-agent-commit.txt").read_text().strip(),
        "started_utc": stamp,
    }, indent=2))
    print(f"Output: {out}\n{len(tasks)} tasks, {args.concurrency} at a time", flush=True)

    collector = start_collector(out, prefix, args.cpu_interval)
    records = []
    try:
        with ThreadPoolExecutor(args.concurrency) as pool, (out / "containers.jsonl").open("w") as log:
            futures = {pool.submit(run_task, t, f"{prefix}{i:03d}", out, args): t for i, t in enumerate(tasks)}
            try:
                for future in as_completed(futures):
                    record = future.result()
                    records.append(record)
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(f"[{len(records)}/{len(tasks)}] {record['task_id']}: exit code {record.get('exit_code')}, "
                          f"{(record['end_mono_ns'] - record['start_mono_ns']) / 1e9:.1f} s", flush=True)
            except KeyboardInterrupt:
                for future in futures:
                    future.cancel()
                ids = docker("ps", "-aq", "--filter", f"name=^{prefix}", "--filter", f"name=^agentprep-{prefix}").split()
                if ids:
                    subprocess.run(["docker", "rm", "-f", *ids], capture_output=True)
                raise
    finally:
        collector.send_signal(signal.SIGTERM)
        collector.wait()

    if subprocess.run([MSWEA_PY, CHARACTERIZATION / "analyze_task_metrics.py", out]).returncode:
        print(f"Task analysis failed; trajectories and samples remain in {out}", file=sys.stderr)
    rows = [compare(t, out) for t in tasks]
    with (out / "replay_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
    faithful = sum(bool(r.get("faithful")) for r in rows)
    print(f"Faithful replays: {faithful}/{len(rows)} (see replay_summary.csv)\nOutput: {out}")


if __name__ == "__main__":
    main()
