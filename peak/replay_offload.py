#!/usr/bin/env python3
"""Replay logged mini-SWE-agent tool calls in fresh Docker containers and measure swap-based offload.

Per task: start the task image exactly like the original run, replay every executed tool call in order,
and watch the container's memory cgroup. When memory.usage_in_bytes (page cache included) crosses
80% of the task's p99 from the original run, the container is frozen (cgroup freezer, nothing dies),
then its whole footprint is demoted to disk and promoted back:

  demote  = fsync the container's writable layer (dirty cache -> disk), write every page-cache page the
            container loaded (read-only files included, /dev/shm excluded) to a snapshot file on disk as a
            full VM memory snapshot would, then shrink memory.limit_in_bytes to ~0 so the kernel swaps
            anonymous/shmem pages out and drops the page cache.
  promote = lift the limit, then a helper process placed in the container's memory cgroup (but not its
            freezer cgroup) faults back every page that was resident before demotion: process pages via
            /proc/<pid>/mem (pagemap snapshot) and page cache via file reads (mincore snapshot of every
            cached page under the container's files; no start-of-task baseline is subtracted).

After promotion the container is thawed, the in-flight tool call is allowed to finish (to check it
survived), and the trial ends. The host page cache is dropped before every task, so the files a task
reads are charged to its own cgroup instead of being left cached (and charged elsewhere) by an earlier
trial. Must run as root on the cgroup v1 host. All times are CLOCK_MONOTONIC.
"""

import argparse
import csv
import ctypes
import errno
import json
import mmap
import os
import pickle
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True  # import the existing monitor without touching the mini-swe-agent tree
sys.path.insert(0, str(ROOT.parent.parent / "mini-swe-agent" / "characterization"))
from system_metrics import CONTAINER_COLUMNS, Container  # noqa: E402

CGROUP = Path("/sys/fs/cgroup")
PAGE = os.sysconf("SC_PAGE_SIZE")
libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
MAP_FAILED = ctypes.c_void_p(-1).value
PAGEMAP_CHUNK = 1 << 20


def now() -> int:
    return time.monotonic_ns()


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[(first_index, count)] of consecutive True entries."""
    if not mask.any():
        return []
    d = np.diff(np.r_[0, mask.view(np.int8), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), (ends - starts).tolist()))


# ---------------------------------------------------------------- inputs from the original run

def load_tasks(run_dir: Path, percentile: float, fraction: float) -> list[dict]:
    tasks = pd.read_csv(run_dir / "task_metrics" / "tasks.csv")
    samples = pd.read_csv(run_dir / "system_metrics" / "container_samples.csv",
                          usecols=["container_id", "mem_usage_bytes"])
    p = samples.groupby("container_id").mem_usage_bytes.quantile(percentile / 100)
    out = []
    for row in tasks.itertuples():
        traj_path = run_dir / "results" / row.task_id / f"{row.task_id}.traj.json"
        traj = json.loads(traj_path.read_text())
        out.append({
            "task_id": row.task_id, "orig_exit_status": row.exit_status,
            "orig_p_usage_bytes": int(p[row.container_id]),
            "threshold_bytes": int(fraction * p[row.container_id]),
            "env": traj["info"]["config"]["environment"], "actions": load_actions(traj),
        })
    return out


def load_actions(traj: dict) -> list[dict]:
    """Executed tool calls in order. FormatError steps have no assistant message, so map via step_timings."""
    assistant = [m for m in traj["messages"] if m["role"] == "assistant"]
    steps = [s for s in traj["step_timings"] if s["inference"]["outcome"] != "FormatError"]
    assert len(assistant) == len(steps), "assistant messages do not line up with step_timings"
    results = {m["tool_call_id"]: m.get("extra", {}) for m in traj["messages"] if m["role"] == "tool"}
    actions = []
    for msg, step in zip(assistant, steps):
        for i, (act, timing) in enumerate(zip(msg.get("extra", {}).get("actions", []), step["tools"]), 1):
            orig = results.get(act.get("tool_call_id"), {})
            actions.append({"step": step["step"], "action": i, "command": act["command"],
                            "orig_returncode": orig.get("returncode"), "orig_outcome": timing["outcome"],
                            "orig_elapsed_s": timing["elapsed_s"]})
    return actions


# ---------------------------------------------------------------- cgroup v1 helpers

class Cgroup:
    def __init__(self, cid: str):
        self.mem = CGROUP / "memory" / "docker" / cid
        self.frz = CGROUP / "freezer" / "docker" / cid
        self.usage_fd = os.open(self.mem / "memory.usage_in_bytes", os.O_RDONLY)

    def usage(self) -> int:
        return int(os.pread(self.usage_fd, 64, 0))

    def stat(self) -> dict:
        s = {k: int(v) for k, v in (l.split() for l in (self.mem / "memory.stat").read_text().splitlines())}
        s["usage_in_bytes"] = self.usage()
        s["memsw_usage_in_bytes"] = int((self.mem / "memory.memsw.usage_in_bytes").read_text())
        s["kmem_usage_in_bytes"] = int((self.mem / "memory.kmem.usage_in_bytes").read_text())
        return s

    def pids(self) -> list[int]:
        return [int(x) for x in (self.mem / "cgroup.procs").read_text().split()]

    def freeze(self):
        (self.frz / "freezer.state").write_text("FROZEN")

    def thaw(self):
        (self.frz / "freezer.state").write_text("THAWED")

    def wait_frozen(self, timeout=10.0) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if (self.frz / "freezer.state").read_text().strip() == "FROZEN":
                return True
            time.sleep(0.0002)
        return False

    def set_limit(self, value: int):
        fd = os.open(self.mem / "memory.limit_in_bytes", os.O_WRONLY)
        try:
            os.write(fd, str(value).encode())
        finally:
            os.close(fd)

    def close(self):
        os.close(self.usage_fd)


class ThresholdTrigger(threading.Thread):
    """Kernel memory-threshold notification (cgroup.event_control + eventfd): freeze on first crossing."""

    def __init__(self, cg: Cgroup, threshold: int, on_cross):
        super().__init__(daemon=True)
        self.cg, self.on_cross = cg, on_cross
        self.efd = os.eventfd(0)
        (cg.mem / "cgroup.event_control").write_text(f"{self.efd} {cg.usage_fd} {threshold}")
        self.stopped = False

    def run(self):
        try:
            os.eventfd_read(self.efd)
        except OSError:
            return
        if not self.stopped:
            self.on_cross("eventfd")

    def stop(self):
        self.stopped = True
        os.eventfd_write(self.efd, 1)
        self.join()
        os.close(self.efd)


# ---------------------------------------------------------------- page snapshots (taken while frozen)

def pagemap_snapshot(pids: list[int]) -> dict:
    """Per pid: resident (present or swapped) virtual page runs, taken from /proc/<pid>/pagemap."""
    snap = {}
    for pid in pids:
        try:
            maps = Path(f"/proc/{pid}/maps").read_text().splitlines()
            pm = os.open(f"/proc/{pid}/pagemap", os.O_RDONLY)
        except (FileNotFoundError, ProcessLookupError):
            continue
        ranges, pages = [], 0
        try:
            for line in maps:
                fields = line.split()
                if fields[-1] in ("[vsyscall]", "[vvar]") or fields[1][0] != "r":
                    continue
                lo, hi = (int(x, 16) for x in fields[0].split("-"))
                for chunk in range(lo, hi, PAGEMAP_CHUNK * PAGE):  # bounded reads for huge reservations
                    n = min(PAGEMAP_CHUNK, (hi - chunk) // PAGE)
                    entries = np.frombuffer(os.pread(pm, n * 8, chunk // PAGE * 8), dtype=np.uint64)
                    resident = (entries & np.uint64(3 << 62)) != 0  # bit 63 present, bit 62 swapped
                    for first, count in runs(resident):
                        ranges.append((chunk + first * PAGE, count * PAGE))
                        pages += count
        finally:
            os.close(pm)
        snap[pid] = {"ranges": ranges, "pages": pages}
    return snap


def walk_files(root: Path):
    stack = [str(root)]
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(d)
        except OSError:
            continue
        with it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                    elif e.is_file(follow_symlinks=False):
                        yield e.path
                except OSError:
                    pass


def mincore_file(path: str) -> list[tuple[int, int]] | None:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOATIME | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        size = os.fstat(fd).st_size
        if size == 0:
            return None
        addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if addr in (None, MAP_FAILED):
            return None
        vec = np.zeros((size + PAGE - 1) // PAGE, dtype=np.uint8)
        try:
            if libc.mincore(addr, size, vec.ctypes.data) != 0:
                return None
        finally:
            libc.munmap(addr, size)
        return runs((vec & 1).astype(bool))
    finally:
        os.close(fd)


def cache_snapshot(roots: list[Path]) -> dict[str, set[int]]:
    """Resident page indexes of every file under the container's rootfs (+ /dev/shm)."""
    snap = {}
    for root in roots:
        for path in walk_files(root):
            r = mincore_file(path)
            if r:
                snap[path] = {p for first, count in r for p in range(first, first + count)}
    return snap


def cache_delta(before: dict, after: dict) -> dict[str, list[tuple[int, int]]]:
    """Pages cached at freeze time that were not cached when the container started: the container's cache."""
    out = {}
    for path, pages in after.items():
        new = pages - before.get(path, set())
        if new:
            idx = np.zeros(max(new) + 1, dtype=bool)
            idx[list(new)] = True
            out[path] = [(f * PAGE, c * PAGE) for f, c in runs(idx)]
    return out


def resident_pages(cache: dict[str, list[tuple[int, int]]]) -> int:
    n = 0
    for path, ranges in cache.items():
        r = mincore_file(path)
        if not r:
            continue
        have = {p for f, c in r for p in range(f, f + c)}
        n += sum(len(have.intersection(range(o // PAGE, (o + l) // PAGE))) for o, l in ranges)
    return n


def write_cache_snapshot(cache: dict[str, list[tuple[int, int]]], dest: Path) -> dict:
    """Copy the recorded page-cache ranges into one file and fsync it. The pages are resident, so the
    reads are memory copies; the fsync makes the snapshot reach disk."""
    buf = bytearray(4 << 20)
    view = memoryview(buf)
    written = errors = 0
    out = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        for path, ranges in cache.items():
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NOATIME)
            except OSError:
                errors += 1
                continue
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)  # no readahead: copy only what was cached
            for off0, length in ranges:
                off = 0
                while off < length:
                    n = min(len(buf), length - off)
                    try:
                        got = os.preadv(fd, [view[:n]], off0 + off)
                    except OSError:
                        errors += 1
                        break
                    written += os.write(out, view[:got])
                    if got < n:
                        break
                    off += n
            os.close(fd)
        os.fsync(out)
    finally:
        os.close(out)
    return {"bytes": written, "files": len(cache), "errors": errors}


# ---------------------------------------------------------------- promote helper (runs as child process)

def promote_child(plan_path: str):
    """Join the container's memory cgroup (not its freezer cgroup) and fault every recorded page back."""
    plan = pickle.loads(Path(plan_path).read_bytes())
    buf = bytearray(4 << 20)
    view = memoryview(buf)
    Path(plan["memcg"], "cgroup.procs").write_text(str(os.getpid()))
    t0 = now()
    proc_bytes = proc_errors = 0
    for pid, info in plan["pagemap"].items():
        try:
            fd = os.open(f"/proc/{pid}/mem", os.O_RDONLY)
        except OSError:
            proc_errors += 1
            continue
        for addr, length in info["ranges"]:
            off = 0
            while off < length:
                n = min(len(buf), length - off)
                try:
                    proc_bytes += os.preadv(fd, [view[:n]], addr + off)
                except OSError:
                    proc_errors += 1
                off += n
        os.close(fd)
    t1 = now()
    cache_bytes = cache_errors = 0
    for path, ranges in plan["cache"].items():
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOATIME)
        except OSError:
            cache_errors += 1
            continue
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)  # no readahead: bring back exactly what was there
        for off0, length in ranges:
            off = 0
            while off < length:
                n = min(len(buf), length - off)
                try:
                    got = os.preadv(fd, [view[:n]], off0 + off)
                except OSError:
                    cache_errors += 1
                    break
                cache_bytes += got
                if got < n:
                    break
                off += n
        os.close(fd)
    t2 = now()
    print(json.dumps({"proc_start_ns": t0, "proc_end_ns": t1, "cache_end_ns": t2,
                      "proc_bytes_read": proc_bytes, "proc_read_errors": proc_errors,
                      "cache_bytes_read": cache_bytes, "cache_read_errors": cache_errors}))


# ---------------------------------------------------------------- one trial

class Sampler(threading.Thread):
    """Existing Container sampler (same cgroup files as the original runs) at a finer interval + phase tag."""

    def __init__(self, cid: str, interval: float, out: Path):
        super().__init__(daemon=True)
        self.container, self.interval, self.out = Container(cid), interval, out
        self.phase, self.rows, self.stop_event, self.on_sample = "replay", [], threading.Event(), None

    def run(self):
        nxt = time.monotonic()
        while not self.stop_event.is_set():
            try:
                row = self.container.sample()
            except OSError:
                break
            self.rows.append([self.phase, *row])
            if self.on_sample:
                self.on_sample(row[5])
            nxt += self.interval
            time.sleep(max(0.0, nxt - time.monotonic()))

    def finish(self):
        self.stop_event.set()
        self.join()
        self.container.close()
        with self.out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["phase", *CONTAINER_COLUMNS])
            w.writerows(self.rows)


def drop_caches():
    """Write dirty pages to disk, then evict all clean page cache on the host."""
    subprocess.run(["sync"], check=True)
    Path("/proc/sys/vm/drop_caches").write_text("3")


def docker(*args, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=True, **kw)


def run_trial(task: dict, out_dir: Path, args, measure=None) -> dict:
    """measure(cg, cid, upper, roots, cache_before, sampler, rec, out_dir) runs while frozen (default: measure_offload)."""
    env = task["env"]
    name = f"offload-{task['task_id'].replace('__', '-').lower()[:40]}-{os.getpid()}"
    rec = {k: task[k] for k in ("task_id", "orig_exit_status", "orig_p_usage_bytes", "threshold_bytes")}
    rec |= {"percentile": args.percentile, "fraction": args.fraction, "image": env["image"],
            "n_actions": len(task["actions"]), "triggered": False}
    cid = docker("run", "-d", "--name", name, "-w", env["cwd"], "--rm", env["image"],
                 "sleep", env["container_timeout"]).stdout.strip()
    rec["container_id"] = cid
    cg = Cgroup(cid)
    trig = sampler = None
    try:
        info = json.loads(docker("inspect", cid).stdout)[0]
        init_pid, upper = info["State"]["Pid"], Path(info["GraphDriver"]["Data"]["UpperDir"])
        roots = [Path(info["GraphDriver"]["Data"]["MergedDir"]), Path(f"/proc/{init_pid}/root/dev/shm")]
        # No baseline to subtract: the host cache was dropped before this task, so every page cached under
        # the container's files is the container's, including the files loaded while it started.
        cache_before = {}
        rec["usage_at_start_bytes"] = cg.usage()

        lock, triggered = threading.Lock(), threading.Event()
        state = {"current": None, "last": None}
        log = []

        def on_cross(source: str):
            with lock:
                if triggered.is_set():
                    return
                t_cross = now()
                cg.freeze()
                rec.update(triggered=True, trigger_source=source, trigger_ns=t_cross,
                           usage_at_trigger_bytes=cg.usage())
                cur = state["current"]
                rec["between_tool_calls"] = cur is None
                cur = cur or state["last"]
                if cur:
                    rec.update(step=cur["step"], action=cur["action"], action_index=cur["index"],
                               command=cur["command"], tool_call_running_s=(t_cross - cur["start_ns"]) / 1e9)
                sampler.phase = "frozen"
                triggered.set()

        sampler = Sampler(cid, args.sample_interval, out_dir / "samples.csv")
        sampler.on_sample = lambda usage: usage >= task["threshold_bytes"] and not triggered.is_set() \
            and on_cross("sampler")
        sampler.start()
        trig = ThresholdTrigger(cg, task["threshold_bytes"], on_cross)
        trig.start()
        if cg.usage() >= task["threshold_bytes"]:
            on_cross("already_above_at_start")

        def replay():
            exec_env = [x for k, v in env["env"].items() for x in ("-e", f"{k}={v}")]
            for i, a in enumerate(task["actions"]):
                with lock:
                    if triggered.is_set():
                        return
                    cur = state["current"] = a | {"index": i, "start_ns": now()}
                cmd = ["docker", "exec", "-w", env["cwd"], *exec_env, cid, *env["interpreter"], a["command"]]
                try:
                    r = subprocess.run(cmd, capture_output=True, timeout=env["timeout"])
                    rc = r.returncode
                except subprocess.TimeoutExpired:
                    rc = "timeout"
                end = now()
                with lock:
                    state["current"], state["last"] = None, cur
                log.append({"index": i, "step": a["step"], "action": a["action"], "start_ns": cur["start_ns"],
                            "end_ns": end, "returncode": rc, "orig_returncode": a["orig_returncode"],
                            "frozen_during": triggered.is_set(), "command": a["command"]})

        replayer = threading.Thread(target=replay, daemon=True)
        replayer.start()
        while replayer.is_alive() and not triggered.is_set():
            replayer.join(0.05)
        rec["peak_usage_during_replay_bytes"] = max((r[6] for r in sampler.rows), default=0)

        if triggered.is_set():
            rec["frozen_ok"] = cg.wait_frozen()
            rec["freeze_latency_s"] = (now() - rec["trigger_ns"]) / 1e9
            (measure or measure_offload)(cg, cid, upper, roots, cache_before, sampler, rec, out_dir)
            cg.thaw()
            sampler.phase = "thawed"
            replayer.join(env["timeout"] + 30)  # let the frozen tool call finish: it must have survived
            if rec.get("action_index") is not None:
                done = [l for l in log if l["index"] == rec["action_index"]]
                rec["frozen_call_returncode_after_thaw"] = done[0]["returncode"] if done else None
                rec["frozen_call_orig_returncode"] = task["actions"][rec["action_index"]]["orig_returncode"]
        rec["actions_replayed"] = len(log)
        rec["returncode_mismatches"] = sum(str(l["returncode"]) != str(l["orig_returncode"])
                                           for l in log if l["orig_returncode"] is not None)
        pd.DataFrame(log).to_csv(out_dir / "actions.csv", index=False)
    finally:
        if trig:
            trig.stop()
        try:
            cg.thaw()
        except OSError:
            pass
        if sampler:
            sampler.finish()
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True)
        cg.close()
    return rec


def measure_offload(cg: Cgroup, cid, upper: Path, roots, cache_before, sampler: Sampler, rec: dict, out_dir):
    # ---- what is in memory at the freeze
    before = cg.stat()
    rec["mem_at_freeze"] = before
    for k in ("usage_in_bytes", "rss", "cache", "shmem", "mapped_file", "dirty", "kmem_usage_in_bytes"):
        rec[f"freeze_{k}"] = before.get(k)
    rec["writable_layer_bytes"] = int(subprocess.run(["du", "-sb", str(upper)], capture_output=True,
                                                     text=True).stdout.split()[0])
    t = now()
    pids = cg.pids()
    pagemap = pagemap_snapshot(pids)
    cache = cache_delta(cache_before, cache_snapshot(roots))
    rec["snapshot_s"] = (now() - t) / 1e9
    rec["pids"] = len(pids)
    rec["process_resident_pages"] = sum(v["pages"] for v in pagemap.values())
    rec["container_cache_pages"] = sum(l // PAGE for r in cache.values() for _, l in r)
    rec["container_cache_files"] = len(cache)
    (cg.mem / "memory.swappiness").write_text("100")

    # /dev/shm is shmem: reclaim already writes it to swap, so it is not copied into the cache snapshot
    shm = f"{roots[1]}/"
    disk_cache = {path: r for path, r in cache.items() if not path.startswith(shm)}
    snapshot = out_dir / "cache_snapshot.bin"

    # ---- demote: dirty page cache -> disk, whole container cache -> snapshot file,
    #      then squeeze the cgroup to ~0 (anon/shmem -> swap, page cache dropped)
    sampler.phase = "demote"
    t0 = now()
    dirty_files = 0
    for path in walk_files(upper):
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOATIME)
            os.fsync(fd)
            os.close(fd)
            dirty_files += 1
        except OSError:
            pass
    t1 = now()
    written = write_cache_snapshot(disk_cache, snapshot)
    tw = now()
    attempts, last = 0, None
    while True:
        attempts += 1
        try:
            cg.set_limit(0)
            break
        except OSError as e:
            if e.errno != errno.EBUSY:
                raise
        usage = cg.usage()
        if last is not None and usage >= last - PAGE or attempts >= 20:
            break  # reclaim made no more progress: the rest is unreclaimable (kernel memory)
        last = usage
    deadline = time.monotonic() + 60
    while cg.stat()["writeback"] > 0 and time.monotonic() < deadline:
        time.sleep(0.0005)
    t2 = now()
    snapshot.unlink()
    try:
        cg.set_limit(cg.usage() + (1 << 20))  # hold it out of memory while we look
    except OSError:
        pass
    after_demote = cg.stat()
    rec["mem_after_demote"] = after_demote
    rec.update(demote_flush_s=(t1 - t0) / 1e9, demote_cache_write_s=(tw - t1) / 1e9,
               demote_reclaim_s=(t2 - tw) / 1e9, demote_s=(t2 - t0) / 1e9,
               cache_snapshot_bytes=written["bytes"], cache_snapshot_files=written["files"],
               cache_snapshot_read_errors=written["errors"],
               demote_limit_attempts=attempts, flushed_files=dirty_files,
               demoted_bytes=before["usage_in_bytes"] - after_demote["usage_in_bytes"],
               swapped_bytes=after_demote.get("swap", 0), residual_bytes=after_demote["usage_in_bytes"],
               residual_kmem_bytes=after_demote["kmem_usage_in_bytes"])
    rec["cache_pages_still_resident_after_demote"] = resident_pages(cache)
    sampler.phase = "demoted"

    # ---- promote: lift the limit, fault every recorded page back from swap / disk
    plan = out_dir / "promote_plan.pkl"
    plan.write_bytes(pickle.dumps({"memcg": str(cg.mem), "pagemap": pagemap, "cache": cache}))
    sampler.phase = "promote"
    t0 = now()
    cg.set_limit(-1)
    lift_s = (now() - t0) / 1e9
    child = subprocess.run([sys.executable, __file__, "--promote-child", str(plan)],
                           capture_output=True, text=True, check=True)
    t3 = now()
    plan.unlink()
    c = json.loads(child.stdout)
    after_promote = cg.stat()
    rec["mem_after_promote"] = after_promote
    # promote_s = limit lift + page fault-back; the helper's Python start-up is excluded (reported separately)
    rec.update(promote_s=lift_s + (c["cache_end_ns"] - c["proc_start_ns"]) / 1e9,
               promote_process_s=(c["proc_end_ns"] - c["proc_start_ns"]) / 1e9,
               promote_cache_s=(c["cache_end_ns"] - c["proc_end_ns"]) / 1e9,
               promote_wall_incl_helper_start_s=(t3 - t0) / 1e9,
               promote_bytes_read=c["proc_bytes_read"] + c["cache_bytes_read"],
               promote_read_errors=c["proc_read_errors"] + c["cache_read_errors"],
               usage_after_promote_bytes=after_promote["usage_in_bytes"],
               swap_after_promote_bytes=after_promote.get("swap", 0))
    rec["promote_recovered_fraction"] = after_promote["usage_in_bytes"] / max(1, before["usage_in_bytes"])
    sampler.phase = "promoted"


# ---------------------------------------------------------------- driver

SUMMARY = ["task_id", "triggered", "trigger_source", "step", "action", "command", "between_tool_calls",
           "tool_call_running_s", "orig_p_usage_bytes", "threshold_bytes", "usage_at_trigger_bytes",
           "freeze_usage_in_bytes", "freeze_rss", "freeze_cache", "freeze_shmem", "freeze_mapped_file",
           "freeze_kmem_usage_in_bytes", "writable_layer_bytes", "freeze_latency_s",
           "demote_s", "demote_flush_s", "demote_cache_write_s", "demote_reclaim_s", "cache_snapshot_bytes",
           "demoted_bytes", "swapped_bytes",
           "residual_bytes", "residual_kmem_bytes", "cache_pages_still_resident_after_demote",
           "promote_s", "promote_process_s", "promote_cache_s", "promote_bytes_read", "promote_read_errors",
           "usage_after_promote_bytes", "promote_recovered_fraction", "frozen_call_returncode_after_thaw",
           "frozen_call_orig_returncode", "peak_usage_during_replay_bytes", "actions_replayed", "n_actions",
           "returncode_mismatches", "error"]


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--promote-child":
        return promote_child(sys.argv[2])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--percentile", type=float, default=99)
    ap.add_argument("--fraction", type=float, default=0.8)
    ap.add_argument("--sample-interval", type=float, default=0.005)
    ap.add_argument("--tasks", nargs="*", help="only these task ids")
    args = ap.parse_args()
    if os.geteuid() != 0:
        sys.exit("must run as root (cgroup writes, /proc/<pid>/mem, overlay dirs)")
    if not Path("/proc/swaps").read_text().strip().count("\n"):
        sys.exit("no swap enabled: run setup_swap.sh first")
    sys.setswitchinterval(0.0005)
    out_root = args.out / args.run_dir.name
    out_root.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(args.run_dir, args.percentile, args.fraction)
    if args.tasks:
        tasks = [t for t in tasks if t["task_id"] in args.tasks]
    print(f"{args.run_dir.name}: {len(tasks)} tasks", flush=True)
    for n, task in enumerate(tasks, 1):
        out_dir = out_root / task["task_id"]
        if (out_dir / "trial.json").exists():
            continue
        out_dir.mkdir(exist_ok=True)
        try:
            drop_caches()
            rec = run_trial(task, out_dir, args)
        except Exception as e:  # keep going; record the failure
            rec = {"task_id": task["task_id"], "triggered": False, "error": repr(e)}
        (out_dir / "trial.json").write_text(json.dumps(rec, indent=1, default=str))
        mib = lambda b: f"{b / 2**20:.1f}MiB" if isinstance(b, (int, float)) else "-"
        print(f"[{n}/{len(tasks)}] {task['task_id']}: triggered={rec.get('triggered')} "
              f"step={rec.get('step')} size={mib(rec.get('freeze_usage_in_bytes'))} "
              f"demote={rec.get('demote_s')} promote={rec.get('promote_s')} err={rec.get('error')}", flush=True)
    rows = [json.loads(p.read_text()) for p in sorted(out_root.glob("*/trial.json"))]
    pd.DataFrame(rows).reindex(columns=SUMMARY).to_csv(out_root / "trials.csv", index=False)
    print(f"wrote {out_root / 'trials.csv'}", flush=True)


if __name__ == "__main__":
    main()
