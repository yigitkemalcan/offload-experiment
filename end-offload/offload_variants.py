#!/usr/bin/env python3
"""Two variants of peak's measure_offload() for the end-of-task experiment, one per folder:

  snapshot-promote/  demote unchanged (fsync, cache -> snapshot file, squeeze). The snapshot file is kept and
                     promote reads the page cache back from it, sequentially, instead of from the
                     container's files. Before promote, the snapshot's own page cache is evicted
                     (fadvise DONTNEED), so promote really reads it from disk.
  no-snapshot/       demote = fsync + squeeze, no cache snapshot is written. Promote unchanged: it re-reads
                     the recorded ranges of the container's files.

Everything else (what is recorded at the freeze, the squeeze, which times are measured) is measure_offload()
unchanged. /dev/shm is shmem and comes back from swap in both variants by reading its files, as in
measure_offload(); it was never in the snapshot.

One change from measure_offload(): promote faults back through /proc/<pid>/mem only the process pages that
belong to no file (anon_pagemap_snapshot()). Pages of mapped files are page cache, and they come back with the
rest of it, from the snapshot or from the files. Faulting them through the mapping would read them from disk
with the device's mmap read-around (4 MiB on md0, so up to 1024x the mapped pages). In snapshot-promote they
would then exist twice, as the files' cache and as the snapshot, and the recovered fraction would exceed 1.

In snapshot-promote, the snapshot's pages are read by the helper inside the container's memory cgroup, so they
are charged to it (like a restored VM's memory). They are not the page cache of the container's own files, so
those files and their dentries/inodes are not re-cached: promote_recovered_fraction counts snapshot pages.

How promote reads the process pages back (promote_mode; the default, serial, is the behaviour above, unchanged):
  serial    one /proc/<pid>/mem read after another, so each swapped page is faulted in synchronously and the
            swap device has one request at a time (plus the kernel's swap readahead around each fault)
  threads   the serial reads split over promote_threads threads, each with its own /proc/<pid>/mem
  threads-all  as threads, and the page-cache re-read from the container's files split over the threads as well

How demote frees the process pages (demote_mode; the default, squeeze, is the behaviour above, unchanged):
  squeeze          shrink memory.limit_in_bytes to ~0: the writer's own thread reclaims everything, one batch after
                   another (page cache dropped, process pages written to swap)
  pageout-threads  first process_madvise(MADV_PAGEOUT) over the recorded process ranges, split over demote_threads
                   threads (each writes its share to swap), then the same squeeze for the page cache and the rest
  pageout-drop-threads  as pageout-threads, and then the recorded files' page cache dropped by the threads
                   (posix_fadvise(DONTNEED) per file: clean after the flush, so nothing is written; pages a process
                   maps and /dev/shm stay for the squeeze), then the squeeze for the rest
demote_pageout_s and demote_drop_s are those parts; demote_reclaim_s still covers them, the squeeze and the
writeback wait.

snapshot_threads (default 1: the snapshot written and read as above, unchanged): with more, demote copies the
recorded page-cache ranges into the snapshot with that many threads (each range has its place in the file fixed in
advance, in the serial copy's order and cut at its file's end, so the snapshot holds the same bytes), and promote
reads the snapshot back with that many threads (4 MiB pieces, round-robin).
"""

import ctypes
import errno
import json
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT.parent / "peak"))
import replay_offload as ro  # noqa: E402
from replay_offload import PAGE, now, walk_files  # noqa: E402

# extra trials.csv columns (after replay_end.SUMMARY)
SUMMARY_EXTRA = ["variant", "promote_files_s", "promote_snapshot_s", "promote_snapshot_bytes_read",
                 "snapshot_pages_resident_before_promote"]

RESIDENT, FILE_OR_SHARED = 3 << 62, 1 << 61  # pagemap: present or swapped; file page or shared anonymous
PROMOTE_MODES = ("serial", "threads", "threads-all")
DEMOTE_MODES = ("squeeze", "pageout-threads", "pageout-drop-threads")
SYS_PIDFD_OPEN, SYS_PROCESS_MADVISE, MADV_PAGEOUT, IOV_MAX = 434, 440, 21, 1024  # x86-64; 5.10+
READ_CHUNK = 4 << 20


def anon_pagemap_snapshot(pids: list[int]) -> dict:
    """peak's pagemap_snapshot() without the pages of mapped files (see the module doc). Kept, as in
    ../compression/replay_compress.py's process_pages(): every resident page of a mapping that has no file in
    the page-cache walk (heap, stacks, shared anonymous memory, deleted files), and the private copy-on-write
    pages of mapped files. file_pages counts the resident pages left to the page-cache restore."""
    snap = {}
    for pid in pids:
        try:
            maps = Path(f"/proc/{pid}/maps").read_text().splitlines()
            pm = os.open(f"/proc/{pid}/pagemap", os.O_RDONLY)
        except (FileNotFoundError, ProcessLookupError):
            continue
        ranges, pages, file_pages = [], 0, 0
        try:
            for line in maps:
                fields = line.split(maxsplit=5)
                path = fields[5] if len(fields) > 5 else ""
                if path in ("[vsyscall]", "[vvar]") or fields[1][0] != "r":
                    continue
                no_file = not path or path.startswith("[") or path.endswith(" (deleted)")
                lo, hi = (int(x, 16) for x in fields[0].split("-"))
                for chunk in range(lo, hi, ro.PAGEMAP_CHUNK * PAGE):  # bounded reads for huge reservations
                    n = min(ro.PAGEMAP_CHUNK, (hi - chunk) // PAGE)
                    entries = np.frombuffer(os.pread(pm, n * 8, chunk // PAGE * 8), dtype=np.uint64)
                    resident = (entries & np.uint64(RESIDENT)) != 0
                    keep = resident if no_file else resident & ((entries & np.uint64(FILE_OR_SHARED)) == 0)
                    file_pages += int(resident.sum() - keep.sum())
                    for first, count in ro.runs(keep):
                        ranges.append((chunk + first * PAGE, count * PAGE))
                        pages += count
        finally:
            os.close(pm)
        snap[pid] = {"ranges": ranges, "pages": pages, "file_pages": file_pages}
    return snap


def measure_offload_variant(cg: ro.Cgroup, cid, upper: Path, roots, cache_before, sampler: ro.Sampler, rec: dict,
                            out_dir, *, write_snapshot: bool, promote_from_snapshot: bool,
                            promote_mode: str = "serial", promote_threads: int = 8,
                            demote_mode: str = "squeeze", demote_threads: int = 8, snapshot_threads: int = 1):
    assert write_snapshot or not promote_from_snapshot
    assert promote_mode in PROMOTE_MODES and demote_mode in DEMOTE_MODES
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
    pagemap = anon_pagemap_snapshot(pids)
    cache = ro.cache_delta(cache_before, ro.cache_snapshot(roots))
    rec["snapshot_s"] = (now() - t) / 1e9
    rec["pids"] = len(pids)
    rec["process_resident_pages"] = sum(v["pages"] + v["file_pages"] for v in pagemap.values())
    rec["process_file_pages"] = sum(v["file_pages"] for v in pagemap.values())  # restored as page cache
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
    written = {"bytes": 0, "files": 0, "errors": 0} if not write_snapshot else \
        ro.write_cache_snapshot(disk_cache, snapshot) if snapshot_threads == 1 else \
        write_snapshot_in_threads(disk_cache, snapshot, snapshot_threads)
    if write_snapshot and snapshot_threads > 1:
        rec["snapshot_threads"] = snapshot_threads
    tw = now()
    if demote_mode != "squeeze":
        paged, pageout_errors = pageout_in_threads(pagemap, demote_threads)
        tp = now()
        rec.update(demote_mode=demote_mode, demote_threads=demote_threads, demote_pageout_s=(tp - tw) / 1e9,
                   demote_pageout_bytes=paged, demote_pageout_errors=pageout_errors)
    if demote_mode == "pageout-drop-threads":
        dropped, drop_errors = drop_cache_in_threads(disk_cache, demote_threads)
        rec.update(demote_drop_s=(now() - tp) / 1e9, demote_drop_files=dropped, demote_drop_errors=drop_errors)
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
                                   "snapshot": str(snapshot) if promote_from_snapshot else None,
                                   "proc_mode": promote_mode, "proc_threads": promote_threads,
                                   "snapshot_threads": snapshot_threads}))
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
    if promote_mode != "serial":
        rec.update(promote_mode=promote_mode, promote_threads=promote_threads)
    rec["promote_recovered_fraction"] = after_promote["usage_in_bytes"] / max(1, before["usage_in_bytes"])
    sampler.phase = "promoted"


class _Iovec(ctypes.Structure):
    _fields_ = [("base", ctypes.c_void_p), ("len", ctypes.c_size_t)]


def pageout_in_threads(pagemap: dict, threads: int) -> tuple[int, int]:
    """process_madvise(MADV_PAGEOUT) over the recorded process ranges (in pieces of at most 4 MiB, dealt
    round-robin to threads, each calling for up to IOV_MAX pieces at a time): each thread reclaims its share,
    writing it to swap. Pages used by more than one process are left alone (the squeeze takes them). Needs
    CAP_SYS_NICE. Returns (bytes advised, failed calls)."""
    libc = ctypes.CDLL(None, use_errno=True)
    pieces = [(pid, addr + off, min(READ_CHUNK, length - off))
              for pid, info in pagemap.items() for addr, length in info["ranges"] for off in range(0, length, READ_CHUNK)]

    def work(share):
        advised = errors = 0
        for pid in dict.fromkeys(p for p, _, _ in share):
            mine = [(addr, n) for p, addr, n in share if p == pid]
            pidfd = libc.syscall(SYS_PIDFD_OPEN, pid, 0)
            if pidfd < 0:
                errors += 1
                continue
            try:
                for i in range(0, len(mine), IOV_MAX):
                    batch = mine[i:i + IOV_MAX]
                    iov = (_Iovec * len(batch))(*(_Iovec(addr, n) for addr, n in batch))
                    got = libc.syscall(SYS_PROCESS_MADVISE, pidfd, iov, len(batch), MADV_PAGEOUT, 0)
                    if got < 0:
                        errors += 1
                    else:
                        advised += got
            finally:
                os.close(pidfd)
        return advised, errors

    with ThreadPoolExecutor(threads) as ex:
        results = list(ex.map(work, [pieces[i::threads] for i in range(threads)]))
    return sum(r[0] for r in results), sum(r[1] for r in results)


def write_snapshot_in_threads(cache: dict, dest: Path, threads: int) -> dict:
    """ro.write_cache_snapshot() with the copying split over threads. Every recorded range, cut at its file's end,
    gets its place in the snapshot in advance (the serial copy's order, so the same bytes in the same places); the
    threads copy 4 MiB pieces round-robin with pwrite, then the file is fsynced once."""
    pieces, size, errors = [], 0, 0
    for path, ranges in cache.items():
        try:
            end = os.stat(path).st_size
        except OSError:
            errors += 1
            continue
        for start, length in ranges:
            length = max(0, min(length, end - start))
            for off in range(0, length, READ_CHUNK):
                n = min(READ_CHUNK, length - off)
                pieces.append((path, start + off, n, size))
                size += n
    out = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)

    def work(share):
        view, fds, written, errs = memoryview(bytearray(READ_CHUNK)), {}, 0, 0
        try:
            for path, src, n, dst in share:
                if path not in fds:
                    try:
                        fds[path] = open_no_readahead(path)
                    except OSError:
                        fds[path] = None
                        errs += 1
                if fds[path] is None:
                    continue
                try:
                    got = os.preadv(fds[path], [view[:n]], src)
                except OSError:
                    errs += 1
                    continue
                done = 0
                while done < got:
                    done += os.pwrite(out, view[done:got], dst + done)
                written += got
        finally:
            for fd in fds.values():
                if fd is not None:
                    os.close(fd)
        return written, errs

    try:
        with ThreadPoolExecutor(threads) as ex:
            results = list(ex.map(work, [pieces[i::threads] for i in range(threads)]))
        os.fsync(out)
    finally:
        os.close(out)
    return {"bytes": sum(r[0] for r in results), "files": len(cache), "errors": errors + sum(r[1] for r in results)}


def drop_cache_in_threads(cache: dict, threads: int) -> tuple[int, int]:
    """posix_fadvise(DONTNEED) on every recorded file, files dealt round-robin to threads: the kernel drops each
    file's clean, unmapped cached pages. Returns (files done, errors)."""
    def work(paths):
        done = errors = 0
        for path in paths:
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NOATIME)
            except OSError:
                errors += 1
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                done += 1
            except OSError:
                errors += 1
            finally:
                os.close(fd)
        return done, errors

    paths = list(cache)
    with ThreadPoolExecutor(threads) as ex:
        results = list(ex.map(work, [paths[i::threads] for i in range(threads)]))
    return sum(r[0] for r in results), sum(r[1] for r in results)


# ---------------------------------------------------------------- promote helper (runs as child process)

def read_in_threads(ranges: dict, open_source, threads: int) -> tuple[int, int]:
    """The serial loops' reads ({source: [(offset, length)]} in pieces of at most 4 MiB), dealt round-robin to
    threads that each open the sources they need with open_source(source). Returns (bytes read, errors)."""
    pieces = [(source, start + off, min(READ_CHUNK, length - off))
              for source, rs in ranges.items() for start, length in rs for off in range(0, length, READ_CHUNK)]

    def work(share):
        view, fds, got, errors = memoryview(bytearray(READ_CHUNK)), {}, 0, 0
        try:
            for source, offset, n in share:
                if source not in fds:
                    try:
                        fds[source] = open_source(source)
                    except OSError:
                        fds[source] = None
                        errors += 1
                if fds[source] is None:
                    continue
                try:
                    got += os.preadv(fds[source], [view[:n]], offset)
                except OSError:
                    errors += 1
        finally:
            for fd in fds.values():
                if fd is not None:
                    os.close(fd)
        return got, errors

    with ThreadPoolExecutor(threads) as ex:
        results = list(ex.map(work, [pieces[i::threads] for i in range(threads)]))
    return sum(r[0] for r in results), sum(r[1] for r in results)


def read_process_pages(pagemap: dict, threads: int) -> tuple[int, int]:
    """Process pages through /proc/<pid>/mem, from swap."""
    return read_in_threads({pid: info["ranges"] for pid, info in pagemap.items()},
                           lambda pid: os.open(f"/proc/{pid}/mem", os.O_RDONLY), threads)


def open_no_readahead(path: str) -> int:
    fd = os.open(path, os.O_RDONLY | os.O_NOATIME)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)  # no readahead: bring back exactly what was there
    return fd


def read_file_pages(cache: dict, threads: int) -> tuple[int, int]:
    """Recorded page-cache ranges re-read from the container's files (reads past a file's end return less)."""
    return read_in_threads(cache, open_no_readahead, threads)


def promote_child(plan_path: str):
    """peak's promote_child(), plus a sequential read of the snapshot file when the plan has one."""
    plan = pickle.loads(Path(plan_path).read_bytes())
    buf = bytearray(4 << 20)
    view = memoryview(buf)
    Path(plan["memcg"], "cgroup.procs").write_text(str(os.getpid()))
    mode = plan.get("proc_mode", "serial")
    t0 = now()
    proc_bytes = proc_errors = 0
    threaded = mode in ("threads", "threads-all")
    if threaded:
        proc_bytes, proc_errors = read_process_pages(plan["pagemap"], plan["proc_threads"])
    for pid, info in plan["pagemap"].items() if not threaded else ():
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
    if mode == "threads-all":
        cache_bytes, cache_errors = read_file_pages(plan["cache"], plan["proc_threads"])
    for path, ranges in plan["cache"].items() if mode != "threads-all" else ():
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
    if plan["snapshot"] and plan.get("snapshot_threads", 1) > 1:
        try:
            snap_bytes, snap_errors = read_in_threads({plan["snapshot"]: [(0, os.stat(plan["snapshot"]).st_size)]},
                                                      lambda path: os.open(path, os.O_RDONLY), plan["snapshot_threads"])
        except OSError:
            snap_errors += 1
    elif plan["snapshot"]:
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
