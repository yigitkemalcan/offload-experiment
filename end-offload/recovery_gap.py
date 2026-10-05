#!/usr/bin/env python3
"""What is the memory that end-of-task promote does not bring back?

replay_end.py recovers ~76% of the frozen footprint (median). The missing part is kernel memory plus page
cache charged to the container that the file scan (mincore over the container's rootfs) never sees. This
script runs small synthetic workloads, one fresh container each, and attributes every page-cache page
charged to the container to its owner, at the freeze and again after promote:

  container  pages of files under the container's rootfs and /dev/shm (what promote restores)
  runtime    pages of host files run inside the container's cgroup by `docker exec` (runc and its libs)
  metadata   pages of the block device under Docker's storage: ext4 directory blocks, inode tables, ...
  other      the rest of memory.stat cache

A page is counted when /proc/kpagecgroup says it is charged to the container's memory cgroup (via
/proc/self/pagemap -> PFN), so each page is counted once and hardlinked files are not double counted.
Kernel memory is reported as the cgroup's kmem usage plus host-wide /proc/slabinfo deltas (the per-cgroup
slab breakdown is empty on this kernel; other activity on the host adds noise to these deltas).

Workloads (each preceded by a host cache drop, as in replay_end.py):
  idle        start the container, nothing else                       baseline
  exec40      40 x `true` through docker exec                         runtime files
  find_names  find / -xdev (reads directories, does not stat files)   directory blocks
  find_stat   find / -xdev -printf %s (stats every inode)             + inode tables
  cat_testbed cat every file under /testbed                           file contents (+ their metadata)
  copyup      read 50 tracked .py files, then sed -i them             originals hidden by the overlay copy-up
  replay      the task's own tool calls, as in replay_end.py          the real thing

Each container is then demoted and promoted with peak's measure_offload(), unchanged.

Usage (from this directory; root, swap on, ~2-4 min per workload):
  sudo ../.venv/bin/python recovery_gap.py ../../swebench-runs/<run> <task_id>
  sudo ../.venv/bin/python recovery_gap.py ../../swebench-runs/<run> <task_id> --workloads idle find_stat
  sudo ../.venv/bin/python recovery_gap.py ... --skip-metadata   # skip the block-device scan (slow: 3.8 TB)
Writes results-gap/<task_id>/<workload>.json and prints one table per task.
"""

import argparse
import ctypes
import json
import mmap
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT.parent / "peak"))
import replay_offload as ro  # noqa: E402
from replay_end import replay  # noqa: E402

PAGE, MIB = ro.PAGE, 2**20
PFN_MASK = (1 << 55) - 1
PRESENT = np.uint64(1 << 63)
MAP_CHUNK = 1 << 36  # map large files / the block device 64 GiB at a time
MADV_RANDOM = 1
ro.libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
SLABS = ["dentry", "inode_cache", "ovl_inode", "ext4_inode_cache", "proc_inode_cache", "shmem_inode_cache",
         "buffer_head", "radix_tree_node", "kernfs_node_cache", "filp", "vm_area_struct", "anon_vma"]

WORKLOADS = {
    "idle": [],
    "exec40": ["true"] * 40,
    "find_names": ["find / -xdev > /dev/null"],
    "find_stat": ["find / -xdev -printf '%s\\n' > /dev/null"],
    "cat_testbed": ["find /testbed -type f -exec cat {} + > /dev/null"],
    "copyup": ["git ls-files -z '*.py' | head -z -n 50 | xargs -0 cat > /dev/null",
               "git ls-files -z '*.py' | head -z -n 50 | xargs -0 du -cb | tail -1",  # size of the originals
               "git ls-files -z '*.py' | head -z -n 50 | xargs -0 sed -i '1s/^/# /'"],
    "replay": None,  # the task's own tool calls
}


# ---------------------------------------------------------------- page ownership

class PageOwner:
    """PFNs of the resident pages of a file that are charged to one memory cgroup."""

    def __init__(self, memcg: Path):
        self.ino = os.stat(memcg).st_ino  # kpagecgroup reports the memcg directory's inode number
        self.pagemap = os.open("/proc/self/pagemap", os.O_RDONLY)
        self.kpc = os.open("/proc/kpagecgroup", os.O_RDONLY)
        self.failed_chunks = 0  # mmap/mincore failures: a silently skipped range would undercount
        self.others = Counter()  # memcg inode -> resident pages of the scanned files charged elsewhere

    def close(self):
        os.close(self.pagemap)
        os.close(self.kpc)

    def charged(self, path: str) -> tuple[int, np.ndarray]:
        """(resident pages, PFNs charged to the cgroup). Only resident pages are touched, so nothing is read
        from disk and no page changes owner."""
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError:
            return 0, np.empty(0, np.uint64)
        resident, mine = 0, []
        try:
            size = os.lseek(fd, 0, os.SEEK_END)  # works for block devices too
            for off in range(0, size, MAP_CHUNK):
                n = min(MAP_CHUNK, size - off)
                addr = ro.libc.mmap(None, n, mmap.PROT_READ, mmap.MAP_SHARED, fd, off)
                if addr in (None, ro.MAP_FAILED):
                    self.failed_chunks += 1
                    continue
                try:
                    vec = np.zeros((n + PAGE - 1) // PAGE, dtype=np.uint8)
                    if ro.libc.mincore(addr, n, vec.ctypes.data) != 0:
                        self.failed_chunks += 1
                        continue
                    idx = np.flatnonzero(vec & 1)
                    resident += len(idx)
                    # no fault readahead: touching a cached page must not pull its uncached neighbours in,
                    # or the measurement after this would see (and promote) pages the container never had
                    ro.libc.madvise(addr, n, MADV_RANDOM)
                    for first, count in ro.runs((vec & 1).astype(bool)):
                        for i in range(first, first + count):
                            ctypes.c_char.from_address(addr + i * PAGE).value  # map it: fault, no I/O
                        e = np.frombuffer(os.pread(self.pagemap, count * 8, (addr // PAGE + first) * 8), np.uint64)
                        for pfn in (e[(e & PRESENT) != 0] & np.uint64(PFN_MASK)).tolist():
                            cg = int.from_bytes(os.pread(self.kpc, 8, pfn * 8), "little")
                            if cg == self.ino:
                                mine.append(pfn)
                            else:
                                self.others[cg] += 1
                finally:
                    ro.libc.munmap(addr, n)
        finally:
            os.close(fd)
        return resident, np.array(mine, np.uint64)

    def charged_many(self, paths) -> tuple[int, np.ndarray]:
        resident, mine = 0, []
        for p in paths:
            r, m = self.charged(p)
            resident += r
            mine.append(m)
        return resident, np.unique(np.concatenate(mine)) if mine else np.empty(0, np.uint64)


def runtime_files() -> list[str]:
    """Host files a `docker exec` runs inside the container's cgroup: runc and its shared libraries."""
    files = [shutil.which("runc"), shutil.which("containerd-shim-runc-v2"), "/etc/ld.so.cache"]
    ldd = subprocess.run(["ldd", files[0]], capture_output=True, text=True).stdout
    files += re.findall(r"(/\S+) \(0x", ldd)
    return [os.path.realpath(f) for f in files if f]


def storage_device() -> str:
    root = ro.docker("info", "-f", "{{.DockerRootDir}}").stdout.strip()
    return subprocess.run(["findmnt", "-no", "SOURCE", "-T", root], capture_output=True, text=True,
                          check=True).stdout.strip()


def memcg_paths() -> dict[int, str]:
    """memcg directory inode -> path; kpagecgroup reports 0 for pages charged to no cgroup."""
    base = ro.CGROUP / "memory"
    return {os.stat(d).st_ino: "/" + os.path.relpath(d, base) for d, _, _ in os.walk(base)}


def slab_bytes() -> dict[str, int]:
    out = {}
    for line in Path("/proc/slabinfo").read_text().splitlines()[2:]:
        f = line.split()
        if f[0] in SLABS:
            out[f[0]] = int(f[1]) * int(f[3])  # active objects x object size
    return out


def attribute(cg: ro.Cgroup, owner: PageOwner, roots, runtime, device, slab0) -> dict:
    t = time.monotonic()
    s = cg.stat()
    slab = slab_bytes()  # before the walk below, which itself fills dentry/inode caches
    container_files = [p for r in roots for p in ro.walk_files(r)]
    owner.failed_chunks, owner.others = 0, Counter()
    c_res, c = owner.charged_many(container_files)
    names = memcg_paths()
    elsewhere = {names.get(k, "removed cgroup" if k else "no cgroup"): v * PAGE / MIB
                 for k, v in owner.others.most_common(8)}
    r_res, r = owner.charged_many(runtime)
    failed_files = owner.failed_chunks
    owner.failed_chunks = 0
    m_res, m = owner.charged(device) if device else (0, np.empty(0, np.uint64))
    explained = len(np.union1d(np.union1d(c, r), m))
    cache_pages = s["cache"] // PAGE
    return {
        "usage_mib": s["usage_in_bytes"] / MIB, "cache_mib": s["cache"] / MIB, "rss_mib": s["rss"] / MIB,
        "swap_mib": s.get("swap", 0) / MIB, "kmem_mib": s["kmem_usage_in_bytes"] / MIB,
        "container_files_resident_mib": c_res * PAGE / MIB,  # what the file scan counts (any owner, per path)
        "container_mib": len(c) * PAGE / MIB,
        "container_files_charged_elsewhere_mib": elsewhere,  # owner cgroup -> MiB (top 8)
        "failed_chunks_files": failed_files, "failed_chunks_device": owner.failed_chunks,
        "runtime_mib": len(r) * PAGE / MIB,
        "metadata_mib": len(m) * PAGE / MIB if device else None,
        "metadata_device_resident_mib": m_res * PAGE / MIB if device else None,  # whole host, any owner
        "other_mib": (cache_pages - explained) * PAGE / MIB,
        "slab_delta_mib": {k: (slab.get(k, 0) - slab0.get(k, 0)) / MIB for k in SLABS},
        "attribution_s": time.monotonic() - t,
    }


# ---------------------------------------------------------------- one workload

def run_workload(task: dict, workload: str, out: Path, args, runtime, device) -> dict:
    env = task["env"]
    ro.drop_caches()
    slab0 = slab_bytes()
    name = f"gap-{workload}-{task['task_id'].replace('__', '-').lower()[:30]}-{os.getpid()}"
    cid = ro.docker("run", "-d", "--name", name, "-w", env["cwd"], "--rm", env["image"],
                    "sleep", env["container_timeout"]).stdout.strip()
    cg = ro.Cgroup(cid)
    owner = PageOwner(cg.mem)
    rec = {"task_id": task["task_id"], "workload": workload, "image": env["image"]}
    sampler = None
    try:
        info = json.loads(ro.docker("inspect", cid).stdout)[0]
        upper = Path(info["GraphDriver"]["Data"]["UpperDir"])
        roots = [Path(info["GraphDriver"]["Data"]["MergedDir"]), Path(f"/proc/{info['State']['Pid']}/root/dev/shm")]
        commands = WORKLOADS[workload]
        if commands is None:
            log = replay(task, cid)
            rec["commands"] = len(log)
        else:
            rec["commands"], rec["outputs"] = len(commands), []
            exec_env = [x for k, v in env["env"].items() for x in ("-e", f"{k}={v}")]
            for c in commands:
                r = subprocess.run(["docker", "exec", "-w", env["cwd"], *exec_env, cid, *env["interpreter"], c],
                                   capture_output=True, text=True, timeout=600)
                rec["outputs"].append({"command": c, "returncode": r.returncode, "stdout": r.stdout[-200:],
                                       "stderr": r.stderr[-200:]})
        cg.freeze()
        rec["frozen_ok"] = cg.wait_frozen()
        rec["at_freeze"] = attribute(cg, owner, roots, runtime, device, slab0)
        sampler = ro.Sampler(cid, 0.005, out / f"{workload}.samples.csv")
        sampler.phase = "frozen"
        sampler.start()
        m = {}
        ro.measure_offload(cg, cid, upper, roots, {}, sampler, m, out)
        rec["offload"] = {k: m[k] for k in ("freeze_usage_in_bytes", "cache_snapshot_bytes", "swapped_bytes",
                                            "demote_s", "promote_s", "promote_bytes_read",
                                            "usage_after_promote_bytes", "promote_recovered_fraction")}
        rec["after_promote"] = attribute(cg, owner, roots, runtime, device, slab0)
    finally:
        try:
            cg.thaw()
        except OSError:
            pass
        if sampler:
            sampler.finish()
        owner.close()
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True)
        cg.close()
    return rec


def table(recs: list[dict]) -> str:
    cols = ["usage", "kmem", "cache", "container", "runtime", "metadata", "other", "scan"]
    keys = ["usage_mib", "kmem_mib", "cache_mib", "container_mib", "runtime_mib", "metadata_mib", "other_mib",
            "container_files_resident_mib"]
    lines = [f"{'MiB':24}" + "".join(f"{c:>10}" for c in cols)]
    for r in recs:
        for stage in ("at_freeze", "after_promote"):
            a = r.get(stage)
            if a:
                lines.append(f"{r['workload'] + ' ' + stage.split('_')[0]:24}" +
                             "".join(f"{'-' if a[k] is None else f'{a[k]:.1f}':>10}" for k in keys))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("task_id")
    ap.add_argument("--workloads", nargs="*", default=list(WORKLOADS), choices=list(WORKLOADS))
    ap.add_argument("--skip-metadata", action="store_true", help="skip the block-device scan")
    ap.add_argument("--out", type=Path, default=ROOT / "results-gap")
    args = ap.parse_args()
    if os.geteuid() != 0:
        sys.exit("must run as root (cgroup writes, pagemap PFNs, /proc/kpagecgroup)")
    if not Path("/proc/swaps").read_text().strip().count("\n"):
        sys.exit("no swap enabled: run setup_swap.sh first")
    sys.setswitchinterval(0.0005)
    task = next((t for t in ro.load_tasks(args.run_dir, 99, 0.8) if t["task_id"] == args.task_id), None)
    if task is None:
        sys.exit(f"{args.task_id} not in {args.run_dir}")
    out = args.out / args.task_id
    out.mkdir(parents=True, exist_ok=True)
    runtime = runtime_files()
    device = None if args.skip_metadata else storage_device()
    print(f"{args.task_id}: runtime files {runtime}; metadata device {device}", flush=True)
    recs = []
    for w in args.workloads:
        try:
            rec = run_workload(task, w, out, args, runtime, device)
        except Exception as e:  # keep going; record the failure
            rec = {"task_id": args.task_id, "workload": w, "error": repr(e)}
        (out / f"{w}.json").write_text(json.dumps(rec, indent=1, default=str))
        recs.append(rec)
        a = rec.get("at_freeze", {})
        print(f"{w}: usage={a.get('usage_mib', 0):.1f}MiB attribution={a.get('attribution_s', 0):.0f}s "
              f"recovered={rec.get('offload', {}).get('promote_recovered_fraction')} err={rec.get('error')}",
              flush=True)
    print(table(recs), flush=True)


if __name__ == "__main__":
    main()
