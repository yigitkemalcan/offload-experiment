#!/usr/bin/env python3
"""Two variants of peak's measure_offload() for the end-of-task experiment, one per folder:

  snapshot-promote/  demote unchanged (fsync, cache -> snapshot file, squeeze). The snapshot file is kept and
                     promote reads the page cache back from it, sequentially, instead of from the
                     container's files. Before promote, the snapshot's own page cache is evicted
                     (fadvise DONTNEED), so promote really reads it from disk.
  no-snapshot/       demote = fsync + squeeze, no cache snapshot is written. Promote unchanged: it re-reads
                     the recorded ranges of the container's files.

Everything else (what is recorded at the freeze, the squeeze, process pages faulted back from swap via
/proc/<pid>/mem, which times are measured) is measure_offload() unchanged. /dev/shm is shmem and comes back
from swap in both variants by reading its files, as in measure_offload(); it was never in the snapshot.

In snapshot-promote, the snapshot's pages are read by the helper inside the container's memory cgroup, so they
are charged to it (like a restored VM's memory). They are not the page cache of the container's own files, so
those files and their dentries/inodes are not re-cached: promote_recovered_fraction counts snapshot pages.
"""

import errno
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT.parent / "peak"))
import replay_offload as ro  # noqa: E402
from replay_offload import PAGE, now, walk_files  # noqa: E402

# extra trials.csv columns (after replay_end.SUMMARY)
SUMMARY_EXTRA = ["variant", "promote_files_s", "promote_snapshot_s", "promote_snapshot_bytes_read",
                 "snapshot_pages_resident_before_promote"]


def measure_offload_variant(cg: ro.Cgroup, cid, upper: Path, roots, cache_before, sampler: ro.Sampler, rec: dict,
                            out_dir, *, write_snapshot: bool, promote_from_snapshot: bool):
    assert write_snapshot or not promote_from_snapshot
    rec["variant"] = "snapshot-promote" if promote_from_snapshot else \
        "baseline" if write_snapshot else "no-snapshot"
    # ---- what is in memory at the freeze
    before = cg.stat()
    rec["mem_at_freeze"] = before
    for k in ("usage_in_bytes", "rss", "cache", "shmem", "mapped_file", "dirty", "kmem_usage_in_bytes"):
        rec[f"freeze_{k}"] = before.get(k)
    rec["writable_layer_bytes"] = int(subprocess.run(["du", "-sb", str(upper)], capture_output=True,
                                                     text=True).stdout.split()[0])
    t = now()
    pids = cg.pids()
    pagemap = ro.pagemap_snapshot(pids)
    cache = ro.cache_delta(cache_before, ro.cache_snapshot(roots))
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

    # ---- demote: dirty page cache -> disk, [whole container cache -> snapshot file,]
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
    written = ro.write_cache_snapshot(disk_cache, snapshot) if write_snapshot else \
        {"bytes": 0, "files": 0, "errors": 0}
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
    if promote_from_snapshot:
        # the snapshot was just written, so it is in the host page cache (charged outside the container):
        # evict it (it is fsynced, hence clean) so that promote reads it from disk
        fd = os.open(snapshot, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
        rec["snapshot_pages_resident_before_promote"] = sum(c for _, c in ro.mincore_file(str(snapshot)) or [])
    else:
        snapshot.unlink(missing_ok=True)
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
    rec["cache_pages_still_resident_after_demote"] = ro.resident_pages(cache)
    sampler.phase = "demoted"

    # ---- promote: lift the limit, fault every recorded page back from swap / disk:
    #      process pages from swap, then file ranges (all files, or only /dev/shm), then the snapshot
    files = {path: r for path, r in cache.items() if path.startswith(shm)} if promote_from_snapshot else cache
    plan = out_dir / "promote_plan.pkl"
    plan.write_bytes(pickle.dumps({"memcg": str(cg.mem), "pagemap": pagemap, "cache": files,
                                   "snapshot": str(snapshot) if promote_from_snapshot else None}))
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
    snapshot.unlink(missing_ok=True)
    rec["mem_after_promote"] = after_promote
    # promote_s = limit lift + page fault-back; the helper's Python start-up is excluded (reported separately)
    rec.update(promote_s=lift_s + (c["cache_end_ns"] - c["proc_start_ns"]) / 1e9,
               promote_process_s=(c["proc_end_ns"] - c["proc_start_ns"]) / 1e9,
               promote_cache_s=(c["cache_end_ns"] - c["proc_end_ns"]) / 1e9,  # files + snapshot
               promote_files_s=(c["files_end_ns"] - c["proc_end_ns"]) / 1e9,
               promote_snapshot_s=(c["cache_end_ns"] - c["files_end_ns"]) / 1e9,
               promote_snapshot_bytes_read=c["snapshot_bytes_read"],
               promote_wall_incl_helper_start_s=(t3 - t0) / 1e9,
               promote_bytes_read=c["proc_bytes_read"] + c["cache_bytes_read"] + c["snapshot_bytes_read"],
               promote_read_errors=c["proc_read_errors"] + c["cache_read_errors"] + c["snapshot_read_errors"],
               usage_after_promote_bytes=after_promote["usage_in_bytes"],
               swap_after_promote_bytes=after_promote.get("swap", 0))
    rec["promote_recovered_fraction"] = after_promote["usage_in_bytes"] / max(1, before["usage_in_bytes"])
    sampler.phase = "promoted"


# ---------------------------------------------------------------- promote helper (runs as child process)

def promote_child(plan_path: str):
    """peak's promote_child(), plus a sequential read of the snapshot file when the plan has one."""
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
    snap_bytes = snap_errors = 0
    if plan["snapshot"]:
        try:
            fd = os.open(plan["snapshot"], os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)  # one file, read front to back
            while got := os.readv(fd, [view]):
                snap_bytes += got
            os.close(fd)
        except OSError:
            snap_errors += 1
    t3 = now()
    print(json.dumps({"proc_start_ns": t0, "proc_end_ns": t1, "files_end_ns": t2, "cache_end_ns": t3,
                      "proc_bytes_read": proc_bytes, "proc_read_errors": proc_errors,
                      "cache_bytes_read": cache_bytes, "cache_read_errors": cache_errors,
                      "snapshot_bytes_read": snap_bytes, "snapshot_read_errors": snap_errors}))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--promote-child":
        promote_child(sys.argv[2])
