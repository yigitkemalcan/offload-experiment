#!/usr/bin/env python3
"""Replay each tool call, freeze its container, classify used pages as reused or not, resume.

After every tool call the container is frozen and its pages are checked. A page was
used in the call if it is newly resident (loaded, readahead included) or its idle bit
was cleared since the previous boundary. A used page is reused if the same logical
page (file + offset, or process + virtual page) was used in an earlier call; the
distance to that previous use is recorded in calls and in original agent steps.
Boundary 0, before the first call, is inventory only.

Single workload with ample memory: no eviction or migration is expected, so no
special handling of frame changes. Must run as root on the cgroup v1 x86-64 host.
"""

import argparse
import ctypes
import json
import mmap
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT.parent / "peak"))
import replay_offload as ro  # noqa: E402
from replay_offload import MAP_FAILED, PAGE, PAGEMAP_CHUNK, libc, now  # noqa: E402

PRESENT, FILE_OR_SHARED = 1 << 63, 1 << 61
PFN_MASK = (1 << 55) - 1
KPF_COMPOUND_HEAD, KPF_COMPOUND_TAIL, KPF_ZERO_PAGE = 1 << 15, 1 << 16, 1 << 24
SCHEMA_VERSION = 5
IDLE_BITMAP = "/sys/kernel/mm/page_idle/bitmap"
KPAGEFLAGS = "/proc/kpageflags"
MADV_RANDOM, MADV_POPULATE_READ = 1, 22
AT_EMPTY_PATH, STATX_INO, STATX_BTIME = 0x1000, 0x100, 0x800
SYS_MOVE_PAGES = 279  # x86-64
ANON = np.uint64(1 << 63)  # key = ANON | proc_id << 36 | virtual page; file key = file_id << 32 | page index
KINDS = ("file", "shm", "mapped", "anon")
FILE, SHM, MAPPED, ANON_KIND = range(4)
EMPTY = np.zeros(0, np.uint64)

libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libc.statx.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_uint, ctypes.c_void_p]
libc.syscall.restype = ctypes.c_long
_statx = ctypes.create_string_buffer(256)


# ---------------------------------------------------------------- kernel page interfaces

def read_u64(path: str, index: np.ndarray, gap: int = 512) -> np.ndarray:
    """Entries index[i] of a file of uint64 words (kpageflags, idle bitmap), reading nearby ones together."""
    out = np.zeros(len(index), dtype=np.uint64)
    if not len(index):
        return out
    order = np.argsort(index, kind="stable")
    s = index[order].astype(np.int64)
    cut = np.flatnonzero(np.diff(s) > gap) + 1
    fd = os.open(path, os.O_RDONLY)
    try:
        for a, b in zip(np.r_[0, cut], np.r_[cut, len(s)]):
            lo, hi = int(s[a]), int(s[b - 1]) + 1
            data = bytearray()
            while len(data) < (hi - lo) * 8:  # sysfs returns at most a page per read
                got = os.pread(fd, (hi - lo) * 8 - len(data), lo * 8 + len(data))
                if not got:
                    raise OSError(f"short read from {path}")
                data += got
            out[order[a:b]] = np.frombuffer(data, dtype=np.uint64)[s[a:b] - lo]
    finally:
        os.close(fd)
    return out


def heads(pfns: np.ndarray, flags: np.ndarray) -> np.ndarray:
    """Frames whose idle flag stands for each frame: the head of a transparent huge page, else itself."""
    pfns = pfns.copy()
    tail = (flags & np.uint64(KPF_COMPOUND_TAIL)) != 0
    if tail.any():
        cand = pfns[tail] & ~np.uint64(511)
        ok = (read_u64(KPAGEFLAGS, cand) & np.uint64(KPF_COMPOUND_HEAD)) != 0
        pfns[np.flatnonzero(tail)[ok]] = cand[ok]
    return pfns


def idle(pfns: np.ndarray) -> np.ndarray:
    """True where the frame is still idle: not accessed since it was marked."""
    words = read_u64(IDLE_BITMAP, pfns // np.uint64(64))
    return ((words >> (pfns % np.uint64(64))) & np.uint64(1)) == 1


def mark_idle(pfns: np.ndarray, gap: int = 512):
    if not len(pfns):
        return
    words, inv = np.unique(pfns // np.uint64(64), return_inverse=True)
    masks = np.zeros(len(words), dtype=np.uint64)
    np.bitwise_or.at(masks, inv, np.uint64(1) << (pfns % np.uint64(64)))
    words = words.astype(np.int64)
    cut = np.flatnonzero(np.diff(words) > gap) + 1
    fd = os.open(IDLE_BITMAP, os.O_WRONLY)
    try:
        for a, b in zip(np.r_[0, cut], np.r_[cut, len(words)]):
            lo = int(words[a])
            buf = np.zeros(int(words[b - 1]) - lo + 1, dtype=np.uint64)  # zero words leave frames unchanged
            buf[words[a:b] - lo] = masks[a:b]
            view, off = memoryview(buf.tobytes()), lo * 8
            while view:
                n = os.pwrite(fd, view[:PAGE], off)
                view, off = view[n:], off + n
    finally:
        os.close(fd)


def drain_lru():
    """Put pages still in per-CPU LRU caches on the LRU; the idle flag cannot be set on them otherwise.
    move_pages() with nothing to move does exactly this (lru_cache_disable) and nothing else."""
    status, nodes = (ctypes.c_int * 1)(), (ctypes.c_int * 1)()
    if libc.syscall(SYS_MOVE_PAGES, 0, ctypes.c_ulong(0), None, nodes, status, 0) != 0:
        raise OSError(ctypes.get_errno(), "move_pages")


def file_identity(fd: int) -> tuple[int, int, int]:
    """(device, inode, birth time in ns or 0 where the filesystem keeps none): a recycled inode is a new file."""
    if libc.statx(fd, b"", AT_EMPTY_PATH, STATX_INO | STATX_BTIME, _statx) != 0:
        raise OSError(ctypes.get_errno(), "statx")
    mask, = struct.unpack_from("I", _statx, 0)
    ino, = struct.unpack_from("Q", _statx, 32)
    s, ns = struct.unpack_from("qI", _statx, 80)
    major, minor = struct.unpack_from("II", _statx, 136)
    return os.makedev(major, minor), ino, s * 10**9 + ns if mask & STATX_BTIME else 0


def mapped_identity(pid: int, address_range: str, dev: int, ino: int) -> tuple[int, int, int]:
    """Identity of a mapped file via /proc/pid/map_files (works after unlink)."""
    try:
        fd = os.open(f"/proc/{pid}/map_files/{address_range}", os.O_RDONLY)
    except OSError:
        return dev, ino, 0
    try:
        return file_identity(fd)
    finally:
        os.close(fd)


def lookup(sorted_keys: np.ndarray, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(found, position) of keys in sorted_keys."""
    if not len(sorted_keys):
        return np.zeros(len(keys), bool), np.zeros(len(keys), np.int64)
    pos = np.minimum(np.searchsorted(sorted_keys, keys), len(sorted_keys) - 1)
    return sorted_keys[pos] == keys, pos


# ---------------------------------------------------------------- page identities

class Registry(dict):
    """Small integer ids for files / processes; items[id] describes the object."""

    def __init__(self):
        super().__init__()
        self.items = []

    def id(self, key, info) -> int:
        i = self.get(key)
        if i is None:
            i = self[key] = len(self.items)
            self.items.append(info)
        return i


def scan_files(layers: list[str], shm_root: str | None, files: Registry):
    """Every regular file in the layers and /dev/shm: identity and resident pages (mincore). Touches no page.
    Returns ({(dev, ino)}, [(path, file id, resident mask)] for files with resident pages)."""
    walked, resident = set(), []
    for root in layers + ([shm_root] if shm_root else []):
        for path in ro.walk_files(Path(root)):
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NOATIME | os.O_NONBLOCK)
            except OSError:
                continue
            try:
                ident = file_identity(fd)
                if ident[:2] in walked:  # another hard link
                    continue
                walked.add(ident[:2])
                size = os.fstat(fd).st_size
                if not size:
                    continue
                addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
                if addr in (None, MAP_FAILED):
                    continue
                try:
                    vec = np.zeros((size + PAGE - 1) // PAGE, dtype=np.uint8)
                    ok = libc.mincore(addr, size, vec.ctypes.data) == 0
                finally:
                    libc.munmap(addr, size)
            except OSError:
                continue
            finally:
                os.close(fd)
            mask = (vec & 1).astype(bool)
            if ok and mask.any():
                shm = root == shm_root
                shown = "/dev/shm/" + os.path.relpath(path, root) if shm else "/" + os.path.relpath(path, root)
                info = {"path": shown, "kind": KINDS[SHM if shm else FILE], "layer": root}
                resident.append((path, files.id(ident, info), mask))
    return walked, resident


def file_frames(resident: list) -> tuple[np.ndarray, np.ndarray]:
    """(key, frame) of resident file pages. Mapping them marks them accessed: read idle bits first."""
    keys, pfns = [], []
    pm = os.open("/proc/self/pagemap", os.O_RDONLY)
    try:
        for path, fid, mask in resident:
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NOATIME | os.O_NONBLOCK)
            except OSError:
                continue
            try:
                size = min(os.fstat(fd).st_size, len(mask) * PAGE)
                addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0) if size else None
            finally:
                os.close(fd)
            if addr in (None, MAP_FAILED):
                continue
            try:
                n = (size + PAGE - 1) // PAGE
                if libc.madvise(addr, size, MADV_RANDOM) != 0:
                    raise OSError(ctypes.get_errno(), "MADV_RANDOM")
                for first, count in ro.runs(mask[:n]):
                    if libc.madvise(addr + first * PAGE, count * PAGE, MADV_POPULATE_READ) != 0:
                        raise OSError(ctypes.get_errno(), "MADV_POPULATE_READ")
                e = np.concatenate([np.frombuffer(os.pread(pm, min(PAGEMAP_CHUNK, n - c) * 8, (addr // PAGE + c) * 8),
                                                  dtype=np.uint64) for c in range(0, n, PAGEMAP_CHUNK)])
                idx = np.flatnonzero(mask[:n] & ((e & np.uint64(PRESENT)) != 0))
                pfn = e[idx] & np.uint64(PFN_MASK)
            finally:
                libc.munmap(addr, size)
            keys.append((np.uint64(fid) << np.uint64(32)) | idx.astype(np.uint64))
            pfns.append(pfn)
    finally:
        os.close(pm)
    if not keys:
        return EMPTY, EMPTY
    return np.concatenate(keys), np.concatenate(pfns)


def process_pages(pids: list[int], procs: Registry, files: Registry, walked: set):
    """Present pages of the container processes. Returns anon (key, frame), mapped (key, frame) for files
    outside the walk (walked files come from file_frames), and a row per process."""
    anon_k, anon_p, map_k, map_p, rows = [], [], [], [], []
    for pid in sorted(pids):
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
            maps = Path(f"/proc/{pid}/maps").read_text().splitlines()
            pm = os.open(f"/proc/{pid}/pagemap", os.O_RDONLY)
        except OSError:
            continue
        comm, rest = stat[stat.index("(") + 1:stat.rindex(")")], stat[stat.rindex(")") + 2:].split()
        start = int(rest[19])  # field 22: start time, so a recycled pid is a new process
        pid_id = procs.id((pid, start), {"path": f"pid {pid} ({comm})", "kind": "anon", "layer": ""})
        n_anon = 0
        try:
            for line in maps:
                fields = line.split(maxsplit=5)
                path = fields[5] if len(fields) > 5 else ""
                if path in ("[vsyscall]", "[vvar]", "[vdso]") or fields[1][0] != "r":
                    continue
                lo, hi = (int(x, 16) for x in fields[0].split("-"))
                ino, fid = int(fields[4]), None
                if ino:
                    major, minor = (int(x, 16) for x in fields[3].split(":"))
                    dev, off = os.makedev(major, minor), int(fields[2], 16) // PAGE
                    if (dev, ino) not in walked:
                        fid = files.id(mapped_identity(pid, fields[0], dev, ino),
                                       {"path": path or f"[{fields[3]} {ino}]", "kind": KINDS[MAPPED], "layer": ""})
                for chunk in range(lo, hi, PAGEMAP_CHUNK * PAGE):
                    n = min(PAGEMAP_CHUNK, (hi - chunk) // PAGE)
                    try:
                        e = np.frombuffer(os.pread(pm, n * 8, chunk // PAGE * 8), dtype=np.uint64)
                    except OSError:
                        break
                    present = (e & np.uint64(PRESENT)) != 0
                    shared = present & ((e & np.uint64(FILE_OR_SHARED)) != 0) if ino else np.zeros(n, bool)
                    idx = np.flatnonzero(present & ~shared).astype(np.uint64)  # anonymous or private copies
                    anon_k.append(ANON | (np.uint64(pid_id) << np.uint64(36)) | (np.uint64(chunk // PAGE) + idx))
                    anon_p.append(e[present & ~shared] & np.uint64(PFN_MASK))
                    n_anon += len(idx)
                    if fid is None or not shared.any():  # walked files: file_frames() has these
                        continue
                    page = off + (chunk - lo) // PAGE + np.flatnonzero(shared)
                    map_k.append((np.uint64(fid) << np.uint64(32)) | page.astype(np.uint64))
                    map_p.append(e[shared] & np.uint64(PFN_MASK))
        finally:
            os.close(pm)
        rows.append({"pid": pid, "start_ticks": start, "comm": comm, "cmdline": cmdline, "anon_pages": n_anon})
    cat = lambda xs: np.concatenate(xs) if xs else EMPTY
    return cat(anon_k), cat(anon_p), cat(map_k), cat(map_p), rows


# ---------------------------------------------------------------- per-boundary accounting

CATEGORIES = ("resident", "accessed", "reused", "not_reused")


class ReuseTracker:
    def __init__(self, layers: list[Path], shm_root: Path | None):
        self.layers = [str(r) for r in layers]
        self.shm_root = str(shm_root) if shm_root else None
        self.files, self.procs = Registry(), Registry()
        self.prev = {"key": EMPTY, "pfn": EMPTY, "head": EMPTY}  # previous boundary, sorted by key
        self.last_key, self.last_call = EMPTY, np.zeros(0, np.int64)  # most recent use of each key, sorted
        self.call_steps = []  # original agent step of each call index (0 = startup)
        self.uses = []  # per call: used pages, whether reused, and the reuse distances
        self.processes = []

    def kind_of(self, keys: np.ndarray) -> np.ndarray:
        out = np.full(len(keys), ANON_KIND, dtype=np.int8)
        f = (keys & ANON) == 0
        table = np.array([KINDS.index(it["kind"]) for it in self.files.items] or [0], dtype=np.int8)
        out[f] = table[(keys[f] >> np.uint64(32)).astype(np.int64)]
        return out

    def boundary(self, pids: list[int], step: int, action: int = 0) -> dict:
        """Call with the container frozen. Returns this boundary's page counts."""
        t0 = now()
        call = len(self.call_steps)
        self.call_steps.append(step)
        drain_lru()
        walked, resident_files = scan_files(self.layers, self.shm_root, self.files)
        a_k, a_p, m_k, m_p, procs = process_pages(pids, self.procs, self.files, walked)
        self.processes += [{"call_index": call, "step": step, "action": action} | r for r in procs]

        # Read idle bits of the previous boundary before file_frames() touches the file pages.
        prev_idle = idle(self.prev["head"])
        f_k, f_p = file_frames(resident_files)
        keys, pfns = np.concatenate([f_k, m_k, a_k]), np.concatenate([f_p, m_p, a_p])
        flags = read_u64(KPAGEFLAGS, pfns)
        keep = (flags & np.uint64(KPF_ZERO_PAGE)) == 0
        _, first = np.unique(np.where(keep, pfns, ~np.uint64(0)), return_index=True)  # each frame once
        first = first[keep[first]]
        order = first[np.argsort(keys[first])]
        keys, pfns, flags = keys[order], pfns[order], flags[order]
        head = heads(pfns, flags)

        # Used = newly resident, or resident before and accessed since (idle bit cleared).
        found, pos = lookup(self.prev["key"], keys)
        used = ~found
        used[found] = ~prev_idle[pos[found]]
        # Kernel 5.15 read() does not mark the first page of each 15-page batch after the first as accessed,
        # so it stays idle. Count an idle file page as used when both neighbouring pages were used.
        gap = ~used & ((keys & ANON) == 0) & ((keys & np.uint64(0xFFFFFFFF)) != 0)
        was_used = keys[used]
        used[gap] = lookup(was_used, keys[gap] - np.uint64(1))[0] & lookup(was_used, keys[gap] + np.uint64(1))[0]
        mark_idle(np.unique(head))
        self.prev = {"key": keys, "pfn": pfns, "head": head}

        row = {"call_index": call, "step": step, "action": action,
               "boundary_s": (now() - t0) / 1e9, "pids": len(pids)}
        counts = {name: np.zeros(4, np.int64) for name in CATEGORIES}  # same columns at every boundary
        counts["resident"] = np.bincount(self.kind_of(keys), minlength=4)
        if call:  # boundary 0 is inventory only
            acc = keys[used]
            seen, where = lookup(self.last_key, acc)
            prev_call = np.full(len(acc), -1, np.int64)
            prev_call[seen] = self.last_call[where[seen]]
            call_distance = np.where(seen, call - prev_call, -1)
            step_distance = np.where(seen, step - np.array(self.call_steps)[prev_call], -1)
            self.uses.append({"call_index": call, "step": step, "action": action, "key": acc, "reused": seen,
                              "reuse_distance_calls": call_distance, "reuse_distance_steps": step_distance})
            merged = np.union1d(self.last_key, acc)
            last_call = np.empty(len(merged), np.int64)
            last_call[np.searchsorted(merged, self.last_key)] = self.last_call
            last_call[np.searchsorted(merged, acc)] = call
            self.last_key, self.last_call = merged, last_call
            kind = self.kind_of(acc)
            counts |= {"accessed": np.bincount(kind, minlength=4),
                       "reused": np.bincount(kind[seen], minlength=4),
                       "not_reused": np.bincount(kind[~seen], minlength=4)}
        for name, c in counts.items():
            row |= {f"{name}_{k}_pages": int(n) for k, n in zip(KINDS, c)}
            row[f"{name}_pages"] = int(c.sum())
        return row

    def save(self, out_dir: Path):
        (out_dir / "measurement.json").write_text(json.dumps({
            "schema_version": SCHEMA_VERSION,
            "page_size": PAGE,
            "accessed": "pages newly resident (readahead included) or with idle bit cleared during the call",
            "reused": "accessed page that was also accessed in an earlier call",
            "not_reused": "accessed page with no earlier access; startup inventory excluded",
            "reuse_distance_calls": "call index minus call index of the previous access; -1 if not reused",
            "reuse_distance_steps": "agent step minus agent step of the previous access; -1 if not reused",
        }, indent=2))
        pd.DataFrame(self.processes).to_csv(out_dir / "processes.csv", index=False)
        if not self.uses:
            return
        df = pd.concat([pd.DataFrame(u) for u in self.uses], ignore_index=True)
        np.savez_compressed(out_dir / "accessed.npz", **{c: df[c].to_numpy() for c in df})
        keys = df.key.to_numpy(np.uint64)
        df["kind"] = np.array(KINDS)[self.kind_of(keys)]
        df[df.reused].groupby(["kind", "reuse_distance_calls", "reuse_distance_steps"]).size() \
            .reset_index(name="page_calls").to_csv(out_dir / "reuse_distances.csv", index=False)
        df["anon"] = (keys & ANON) != 0
        df["obj"] = np.where(df.anon, (keys >> np.uint64(36)) & np.uint64((1 << 27) - 1),
                             keys >> np.uint64(32)).astype(np.int64)
        objs = df.groupby(["anon", "obj"]).agg(pages_accessed=("key", "nunique"), page_calls=("key", "size"),
                                               reused_page_calls=("reused", "sum")).reset_index()
        info = [(self.procs if a else self.files).items[o] for a, o in zip(objs.anon, objs.obj)]
        objs.insert(0, "kind", [i["kind"] for i in info])
        objs.insert(1, "object", [i["path"] for i in info])
        objs.insert(2, "layer", [i["layer"] for i in info])
        objs["reuse_fraction"] = objs.reused_page_calls / objs.page_calls
        objs.drop(columns=["anon", "obj"]).sort_values(["reused_page_calls", "page_calls"], ascending=False) \
            .to_csv(out_dir / "objects.csv", index=False)


# ---------------------------------------------------------------- one task

def run_task(task: dict, out_dir: Path) -> tuple[dict, list[dict]]:
    env = task["env"]
    name = f"reuse-{task['task_id'].replace('__', '-').lower()[:40]}-{os.getpid()}"
    rec = {"schema_version": SCHEMA_VERSION, "task_id": task["task_id"], "image": env["image"],
           "n_actions": len(task["actions"])}
    cid = ro.docker("run", "-d", "--name", name, "-w", env["cwd"], "--rm", env["image"],
                    "sleep", env["container_timeout"]).stdout.strip()
    cg = ro.Cgroup(cid)
    rows, log, tracker = [], [], None
    try:
        info = json.loads(ro.docker("inspect", cid).stdout)[0]
        data = info["GraphDriver"]["Data"]
        layers = [Path(data["UpperDir"]), *(Path(p) for p in data["LowerDir"].split(":"))]
        keepalive = info["State"]["Pid"]  # the container's `sleep`: harness, not agent; freeze/thaw touches its stack
        tracker = ReuseTracker(layers, Path(f"/proc/{keepalive}/root/dev/shm"))
        max_usage = cg.mem / "memory.max_usage_in_bytes"

        def boundary(step: int, action: int, elapsed: float):
            peak = int(max_usage.read_text())
            cg.freeze()
            if not cg.wait_frozen():
                raise RuntimeError("container did not freeze")
            try:
                row = tracker.boundary([p for p in cg.pids() if p != keepalive], step, action)
                row |= {"call_s": elapsed, "usage_bytes": cg.usage(),
                        "max_usage_during_call_bytes": peak}
            finally:
                cg.thaw()
            max_usage.write_text("0")
            rows.append(row)
            pd.DataFrame([row]).to_csv(out_dir / "calls.csv", index=False,
                                      mode="w" if len(rows) == 1 else "a", header=len(rows) == 1)
            print(f"  call {row['call_index']} step {step} action {action}: resident={row['resident_pages']} accessed={row.get('accessed_pages')} "
                  f"reused={row.get('reused_pages')} not_reused={row.get('not_reused_pages')} "
                  f"({row['boundary_s']:.1f}s)", flush=True)

        boundary(0, 0, 0.0)
        exec_env = [x for k, v in env["env"].items() for x in ("-e", f"{k}={v}")]
        for a in task["actions"]:
            t0 = time.monotonic()
            cmd = ["docker", "exec", "-w", env["cwd"], *exec_env, cid, *env["interpreter"], a["command"]]
            try:
                rc = subprocess.run(cmd, capture_output=True, timeout=env["timeout"]).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
            log.append({"call_index": len(log) + 1, "step": a["step"], "action": a["action"],
                        "returncode": rc, "orig_returncode": a["orig_returncode"], "command": a["command"]})
            boundary(a["step"], a["action"], time.monotonic() - t0)
    finally:
        try:
            cg.thaw()
        except OSError:
            pass
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True)
        cg.close()
        pd.DataFrame(log).to_csv(out_dir / "actions.csv", index=False)
        if tracker:
            tracker.save(out_dir)

    steps = pd.DataFrame(rows[1:])
    rec |= {"steps": len({r["step"] for r in rows[1:]}), "calls": len(steps), "actions_replayed": len(log),
            "returncode_mismatches": sum(str(l["returncode"]) != str(l["orig_returncode"])
                                         for l in log if l["orig_returncode"] is not None),
            "start_resident_pages": rows[0]["resident_pages"]}
    if len(steps):
        rec["surviving_processes_max"] = int(steps.pids.max())
        rec["distinct_pages_accessed"] = int(np.unique(np.concatenate([u["key"] for u in tracker.uses])).size)
        for k in ("", *(f"{k}_" for k in KINDS)):
            acc = steps[f"accessed_{k}pages"].sum()
            rec[f"accessed_{k}page_calls"] = int(acc)
            for m in ("reused", "not_reused"):
                n = int(steps[f"{m}_{k}pages"].sum())
                rec[f"{m}_{k}page_calls"] = n
                rec[f"{m}_{k}fraction"] = n / acc if acc else None
    return rec, rows


# ---------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "results-v5")
    ap.add_argument("--tasks", nargs="*", help="only these task ids")
    args = ap.parse_args()
    if os.geteuid() != 0:
        sys.exit("must run as root (idle page bitmap, kpageflags, pagemap frames, cgroup freezer)")
    if not os.path.exists(IDLE_BITMAP):
        sys.exit(f"{IDLE_BITMAP} missing: kernel needs CONFIG_IDLE_PAGE_TRACKING")
    out_root = args.out / args.run_dir.name
    out_root.mkdir(parents=True, exist_ok=True)
    for trial in out_root.glob("*/trial.json"):
        if json.loads(trial.read_text()).get("schema_version") != SCHEMA_VERSION:
            sys.exit(f"{trial}: incompatible results; choose a fresh --out directory")
    tasks = ro.load_tasks(args.run_dir, 99, 1.0)  # thresholds are unused here
    if args.tasks:
        tasks = [t for t in tasks if t["task_id"] in args.tasks]
    print(f"{args.run_dir.name}: {len(tasks)} tasks", flush=True)
    for n, task in enumerate(tasks, 1):
        out_dir = out_root / task["task_id"]
        if (out_dir / "trial.json").exists():
            continue
        out_dir.mkdir(exist_ok=True)
        print(f"[{n}/{len(tasks)}] {task['task_id']}", flush=True)
        rows = []
        try:
            ro.drop_caches()
            rec, rows = run_task(task, out_dir)
        except Exception as e:  # keep going; record the failure
            rec = {"schema_version": SCHEMA_VERSION, "task_id": task["task_id"], "error": repr(e)}
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / "calls.csv", index=False)
        (out_dir / "trial.json").write_text(json.dumps(rec, indent=1, default=str))
        print(f"[{n}/{len(tasks)}] {task['task_id']}: steps={rec.get('steps')} "
              f"reused={rec.get('reused_fraction')} not_reused={rec.get('not_reused_fraction')} "
              f"err={rec.get('error')}", flush=True)
    trials = [json.loads(p.read_text()) for p in sorted(out_root.glob("*/trial.json"))]
    pd.DataFrame(trials).to_csv(out_root / "trials.csv", index=False)
    steps = [pd.read_csv(p).assign(task_id=p.parent.name) for p in sorted(out_root.glob("*/calls.csv"))]
    if steps:
        pd.concat(steps).to_csv(out_root / "calls.csv", index=False)
    distances = [pd.read_csv(p).assign(task_id=p.parent.name)
                 for p in sorted(out_root.glob("*/reuse_distances.csv"))]
    if distances:
        pd.concat(distances).to_csv(out_root / "reuse_distances.csv", index=False)
    print(f"wrote {out_root / 'trials.csv'} and {out_root / 'calls.csv'}", flush=True)


if __name__ == "__main__":
    main()
