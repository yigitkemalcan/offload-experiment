#!/usr/bin/env python3
"""Replay logged mini-SWE-agent tool calls with lazy promotion and measure each call's execution time.

Per task, the tool calls are replayed in order in a fresh container, once per mode:

  regular  nothing is demoted between calls.
  lazy     before every call the container is frozen and demoted the same way as in the offload
           experiment (peak/replay_offload.py), and nothing is promoted: fsync the writable layer
           (dirty cache -> disk), set memory.swappiness to 100 and shrink memory.limit_in_bytes to ~0, so
           the kernel drops the page cache and swaps anonymous/shmem pages out. The limit is restored and the
           container thawed; the call then faults back only what it touches.

slowdown = call time (lazy) / call time (regular); see analyze_lazy.py.

Call times are measured exactly as in the original agent run: the container is started and the call is
executed by mini-swe-agent's own DockerEnvironment (same `docker run` / `docker exec` command, environment,
interpreter and timeout), timed by DefaultAgent._measure_call (monotonic_ns around env.execute). Freezing,
demotion and counter reads all happen outside that window. Both modes freeze and thaw at every boundary, so
the only difference between them is the demotion.

The host page cache is dropped before every (task, mode), as in the offload experiment, so the files a
task reads are charged to its own cgroup. Run as root on the cgroup v1 host with the mini-swe-agent venv
(it has numpy and pandas), with swap enabled (../setup_swap.sh on).
"""

import argparse
import errno
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
sys.path.insert(0, str(ROOT.parent / "peak"))
sys.path.insert(0, str(ROOT.parent.parent / "mini-swe-agent" / "src"))
import replay_offload as ro  # noqa: E402
from replay_offload import CGROUP, PAGE, now  # noqa: E402
from minisweagent.agents.default import DefaultAgent  # noqa: E402
from minisweagent.environments.docker import DockerEnvironment  # noqa: E402
from minisweagent.exceptions import Submitted  # noqa: E402

MODES = ("regular", "lazy")
SCHEMA_VERSION = 1


# ---------------------------------------------------------------- demotion (same steps as peak/replay_offload.py)

def squeeze(cg) -> int:
    """Shrink the limit to 0 until reclaim stops making progress; the rest is unreclaimable (kernel memory)."""
    attempts, last = 0, None
    while True:
        attempts += 1
        try:
            cg.set_limit(0)
            return attempts
        except OSError as e:
            if e.errno != errno.EBUSY:
                raise
        usage = cg.usage()
        if last is not None and usage >= last - PAGE or attempts >= 20:
            return attempts
        last = usage


def demote(cg, upper: Path) -> dict:
    """Container is frozen. Dirty cache -> disk, then reclaim everything charged to the container."""
    before = cg.stat()
    t0 = now()
    flushed = 0
    for path in ro.walk_files(upper):
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOATIME)
            os.fsync(fd)
            os.close(fd)
            flushed += 1
        except OSError:
            pass
    t1 = now()
    attempts = squeeze(cg)
    deadline = time.monotonic() + 60
    while cg.stat()["writeback"] > 0 and time.monotonic() < deadline:
        time.sleep(0.0005)
    t2 = now()
    after = cg.stat()
    return {"demote_flush_s": (t1 - t0) / 1e9, "demote_reclaim_s": (t2 - t1) / 1e9, "demote_s": (t2 - t0) / 1e9,
            "flushed_files": flushed, "demote_limit_attempts": attempts,
            "usage_before_demote_bytes": before["usage_in_bytes"],
            "demoted_bytes": before["usage_in_bytes"] - after["usage_in_bytes"],
            "residual_bytes": after["usage_in_bytes"], "residual_kmem_bytes": after["kmem_usage_in_bytes"],
            "residual_user_bytes": after["usage_in_bytes"] - after["kmem_usage_in_bytes"],
            "swap_bytes": after.get("swap", 0)}


def boundary(cg, upper: Path, limit: int, lazy: bool) -> dict:
    """Freeze, demote (lazy mode only), restore the limit, thaw. Nothing is promoted."""
    t0 = now()
    cg.freeze()
    if not cg.wait_frozen():
        cg.thaw()
        raise RuntimeError("container did not freeze")
    rec = {}
    try:
        if lazy:
            rec = demote(cg, upper)
    finally:
        try:
            if lazy:
                cg.set_limit(limit)  # before thawing: the call must not run under memory pressure
        finally:
            cg.thaw()
    rec["boundary_s"] = (now() - t0) / 1e9
    return rec


# ---------------------------------------------------------------- per-call counters (read outside the timed window)

def blkio_read_bytes(path: Path) -> int:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return -1
    return sum(int(f[2]) for f in (l.split() for l in lines) if len(f) == 3 and f[1] == "Read")


def counters(cg, blkio: Path) -> dict:
    s = cg.stat()
    return {"usage_bytes": s["usage_in_bytes"], "pgmajfault": s.get("pgmajfault", 0),
            "pgpgin": s.get("pgpgin", 0), "blkio_read_bytes": blkio_read_bytes(blkio)}


# ---------------------------------------------------------------- replay

def execute(env, command: str) -> tuple[dict, dict]:
    """One tool call, timed as DefaultAgent.execute_actions times it."""
    timing = {}
    try:
        with DefaultAgent._measure_call(timing):
            out = env.execute({"command": command})
    except Submitted:  # the final call; the agent records outcome "Submitted" (returncode was 0)
        out = {"returncode": 0}
    return timing, out


def replay(env, actions: list[dict], before_call, read_counters, on_row=None) -> list[dict]:
    rows = []
    for i, a in enumerate(actions, 1):
        rec = before_call()
        c0 = read_counters()
        timing, out = execute(env, a["command"])
        c1 = read_counters()
        row = {"call_index": i, "step": a["step"], "action": a["action"],
               "call_s": timing["elapsed_s"], "outcome": timing["outcome"], "returncode": out.get("returncode"),
               "orig_elapsed_s": a["orig_elapsed_s"], "orig_outcome": a["orig_outcome"],
               "orig_returncode": a["orig_returncode"], "usage_at_call_start_bytes": c0["usage_bytes"],
               "max_usage_bytes": None} | rec
        row |= {f"call_{k}": c1[k] - c0[k] for k in ("pgmajfault", "pgpgin", "blkio_read_bytes")}
        rows.append(row)
        if on_row:
            on_row(row)
    return rows


def run_task(task: dict, mode: str, out_dir: Path) -> dict:
    lazy = mode == "lazy"
    rec = {"schema_version": SCHEMA_VERSION, "task_id": task["task_id"], "mode": mode,
           "image": task["env"]["image"], "n_actions": len(task["actions"])}
    env = DockerEnvironment(**task["env"])  # same `docker run` as the original agent
    cg = None
    rows = []
    try:
        cid = rec["container_id"] = env.container_id
        cg = ro.Cgroup(cid)
        info = json.loads(ro.docker("inspect", cid).stdout)[0]
        upper = Path(info["GraphDriver"]["Data"]["UpperDir"])
        limit = int((cg.mem / "memory.limit_in_bytes").read_text())
        if lazy:
            (cg.mem / "memory.swappiness").write_text("100")
        rec["usage_at_start_bytes"] = cg.usage()
        blkio = CGROUP / "blkio" / "docker" / cid / "blkio.throttle.io_service_bytes"
        max_usage = cg.mem / "memory.max_usage_in_bytes"

        def before_call():
            r = boundary(cg, upper, limit, lazy)
            max_usage.write_text("0")
            return r

        def on_row(row):
            row["max_usage_bytes"] = int(max_usage.read_text())
            pd.DataFrame([row]).to_csv(out_dir / "calls.csv", index=False,
                                      mode="w" if row["call_index"] == 1 else "a", header=row["call_index"] == 1)
            print(f"  [{mode}] call {row['call_index']} step {row['step']}: {row['call_s']:.3f}s "
                  f"(orig {row['orig_elapsed_s']:.3f}s) rc={row['returncode']} "
                  f"demoted={row.get('demoted_bytes', 0) / 2**20:.1f}MiB "
                  f"read={row['call_blkio_read_bytes'] / 2**20:.1f}MiB", flush=True)

        rows = replay(env, task["actions"], before_call, lambda: counters(cg, blkio), on_row)
    finally:
        if cg:
            try:
                cg.thaw()
            except OSError:
                pass
            cg.close()
        subprocess.run(["docker", "rm", "-f", env.container_id], capture_output=True)
        env.container_id = None  # cleanup() would stop it again in the background
    if not rows:
        return rec
    df = pd.DataFrame(rows)
    rec |= {"calls": len(df), "total_call_s": float(df.call_s.sum()),
            "orig_total_s": float(df.orig_elapsed_s.sum()),
            "returncode_mismatches": int(sum(str(r) != str(o) for r, o in zip(df.returncode, df.orig_returncode)
                                             if o is not None and o == o))}
    if lazy:
        rec |= {"demote_s_total": float(df.demote_s.sum()), "residual_user_bytes_max": int(df.residual_user_bytes.max()),
                "residual_kmem_bytes_max": int(df.residual_kmem_bytes.max())}
    return rec


# ---------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES),
                    help="run in this order for each task (default: regular lazy)")
    ap.add_argument("--tasks", nargs="*", help="only these task ids")
    args = ap.parse_args()
    if os.geteuid() != 0:
        sys.exit("must run as root (cgroup writes, drop_caches, overlay dirs)")
    if "lazy" in args.modes and not Path("/proc/swaps").read_text().strip().count("\n"):
        sys.exit("no swap enabled: run ../setup_swap.sh on first")
    out_root = args.out / args.run_dir.name
    out_root.mkdir(parents=True, exist_ok=True)
    tasks = ro.load_tasks(args.run_dir, 99, 1.0)  # thresholds are unused here
    if args.tasks:
        tasks = [t for t in tasks if t["task_id"] in args.tasks]
    print(f"{args.run_dir.name}: {len(tasks)} tasks, modes {args.modes}", flush=True)
    for n, task in enumerate(tasks, 1):
        for mode in args.modes:
            out_dir = out_root / task["task_id"] / mode
            if (out_dir / "trial.json").exists():
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{n}/{len(tasks)}] {task['task_id']} {mode}", flush=True)
            try:
                ro.drop_caches()
                rec = run_task(task, mode, out_dir)
            except Exception as e:  # keep going; record the failure
                rec = {"schema_version": SCHEMA_VERSION, "task_id": task["task_id"], "mode": mode, "error": repr(e)}
            (out_dir / "trial.json").write_text(json.dumps(rec, indent=1, default=str))
            print(f"[{n}/{len(tasks)}] {task['task_id']} {mode}: total={rec.get('total_call_s')} "
                  f"orig={rec.get('orig_total_s')} mismatches={rec.get('returncode_mismatches')} "
                  f"err={rec.get('error')}", flush=True)
    trials = [json.loads(p.read_text()) for p in sorted(out_root.glob("*/*/trial.json"))]
    pd.DataFrame(trials).to_csv(out_root / "trials.csv", index=False)
    calls = [pd.read_csv(p).assign(task_id=p.parent.parent.name, mode=p.parent.name)
             for p in sorted(out_root.glob("*/*/calls.csv"))]
    if calls:
        pd.concat(calls).to_csv(out_root / "calls.csv", index=False)
    print(f"wrote {out_root / 'trials.csv'} and {out_root / 'calls.csv'}", flush=True)


if __name__ == "__main__":
    main()
