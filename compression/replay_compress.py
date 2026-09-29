#!/usr/bin/env python3
"""Replay logged mini-SWE-agent tool calls in fresh Docker containers and time compressing their memory.

Per task the replay and the trigger are exactly those of ../peak/replay_offload.py: the container is
started like the original run, its tool calls are replayed, and it is frozen (cgroup freezer) when
memory.usage_in_bytes crosses 80% of the task's p99 from the original run. While frozen, its memory is
dumped to one dense image, the container equivalent of the populated-page VM capture in
mini-swe-agent-jovan-main/compression_experiment.py:

  process pages = every present or swapped page of the container's processes that belongs to no file
                  under the rootfs: anonymous memory (heap, stacks, copy-on-write copies) and shared
                  anonymous memory (SysV, memfd, MAP_SHARED anon). Pages shared between processes
                  (fork) are copied once, keyed by page frame / swap slot.
  page cache    = every cached page of every file under the container's rootfs and /dev/shm (mincore).
                  File pages mapped by processes are copied here, not above, so no page is copied twice.

Zero pages are dropped, as densify() does for guest memory. The container is then thawed, the in-flight
tool call finishes and the container is removed, so the codecs run on an otherwise idle host (as phase B
of compression_experiment.py). Each codec compresses the image file to a file and decompresses it back,
with the same command lines as compression_experiment.py; each is repeated and the median kept, and the
start-up cost of the codec binary (run on an empty input) is reported so it can be subtracted, since
container images are tens of MiB rather than GiB. Must run as root on the cgroup v1 host.
"""

import argparse
import csv
import filecmp
import json
import os
import shutil
import statistics
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
from replay_offload import PAGE, PAGEMAP_CHUNK, cache_delta, cache_snapshot, now  # noqa: E402

PRESENT, SWAPPED, FILE_OR_SHARED = 1 << 63, 1 << 62, 1 << 61
PFN_MASK = (1 << 55) - 1  # page frame (present) or swap type+offset (swapped); zero unless root
CODECS = "lz4,zstd-1,zstd-3,zstd-9,zstd-19"
MIB = 2**20


# ---------------------------------------------------------------- capture (taken while frozen)

def process_pages(pids: list[int]) -> tuple[dict[int, list[tuple[int, int]]], dict]:
    """Per pid: (vaddr, length) runs of the process pages that go into the image (see module doc)."""
    seen, out = set(), {}
    stats = {"pages": 0, "swapped_pages": 0, "shared_duplicate_pages": 0}
    for pid in pids:
        try:
            maps = Path(f"/proc/{pid}/maps").read_text().splitlines()
            pm = os.open(f"/proc/{pid}/pagemap", os.O_RDONLY)
        except OSError:
            continue
        ranges = []
        try:
            for line in maps:
                fields = line.split(maxsplit=5)
                path = fields[5] if len(fields) > 5 else ""
                if path in ("[vsyscall]", "[vvar]") or fields[1][0] != "r":
                    continue
                # A named file's pages are in the page-cache walk; anything else has only this copy.
                no_file = not path or path.startswith("[") or path.endswith(" (deleted)")
                lo, hi = (int(x, 16) for x in fields[0].split("-"))
                for chunk in range(lo, hi, PAGEMAP_CHUNK * PAGE):
                    n = min(PAGEMAP_CHUNK, (hi - chunk) // PAGE)
                    e = np.frombuffer(os.pread(pm, n * 8, chunk // PAGE * 8), dtype=np.uint64)
                    keep = (e & np.uint64(PRESENT | SWAPPED)) != 0
                    if not no_file:
                        keep &= (e & np.uint64(FILE_OR_SHARED)) == 0
                    for i in np.flatnonzero(keep).tolist():
                        key = int(e[i]) & (SWAPPED | PFN_MASK)
                        if key & PFN_MASK:  # frame known (root): keep a page shared by several pids once
                            if key in seen:
                                keep[i] = False
                                stats["shared_duplicate_pages"] += 1
                                continue
                            seen.add(key)
                        stats["swapped_pages"] += bool(int(e[i]) & SWAPPED)
                    for first, count in ro.runs(keep):
                        ranges.append((chunk + first * PAGE, count * PAGE))
                        stats["pages"] += count
        finally:
            os.close(pm)
        out[pid] = ranges
    return out, stats


class DenseImage:
    """Takes memory in page-sized pieces and writes only the populated pages to one file."""

    def __init__(self, path: Path):
        self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        self.counts = {}  # source -> [kept pages, zero pages]

    def add(self, source: str, data: memoryview):
        if len(data) % PAGE:  # last page of a file: the rest of the page in memory is zero
            data = memoryview(bytes(data) + bytes(PAGE - len(data) % PAGE))
        pages = np.frombuffer(data, dtype=np.uint8).reshape(-1, PAGE)
        populated = pages.any(axis=1)
        kept = int(populated.sum())
        c = self.counts.setdefault(source, [0, 0])
        c[0] += kept
        c[1] += len(pages) - kept
        out = memoryview((pages[populated] if kept < len(pages) else pages).reshape(-1))
        while out:
            out = out[os.write(self.fd, out):]

    def close(self):
        os.fsync(self.fd)  # on disk and still in the host page cache: the codecs read it hot
        os.close(self.fd)


def copy_ranges(path: str, ranges, image: DenseImage, source: str, buf: memoryview) -> int:
    """Read the (offset, length) ranges of path into the image. Returns the number of read errors."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOATIME)
    except OSError:
        return 1
    errors = 0
    mem = path.startswith("/proc/")
    try:
        if not mem:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)  # no readahead: copy only what was cached
        for off0, length in ranges:
            off = 0
            while off < length:
                n = min(len(buf), length - off)
                try:
                    got = os.preadv(fd, [buf[:n]], off0 + off)
                except OSError:
                    got = -1
                if got > 0:  # /proc/<pid>/mem returns short at an unreadable page; the next read hits it
                    image.add(source, buf[:got])
                    off += got
                elif got == 0 and not mem:  # end of file
                    break
                else:  # unreadable page: skip it and go on with the rest of the range
                    errors += 1
                    off += PAGE
    finally:
        os.close(fd)
    return errors


def capture(cg: ro.Cgroup, cid, upper, roots, cache_before, sampler, rec: dict, out_dir: Path):
    """run_trial() measure hook: dump the frozen container's memory to out_dir/capture.bin."""
    before = cg.stat()
    rec["mem_at_freeze"] = before
    for k in ("usage_in_bytes", "rss", "cache", "shmem", "mapped_file", "dirty", "kmem_usage_in_bytes"):
        rec[f"freeze_{k}"] = before.get(k)
    sampler.phase = "capture"
    t0 = now()
    pids = cg.pids()
    procs, pstats = process_pages(pids)
    cache = cache_delta(cache_before, cache_snapshot(roots))
    t1 = now()
    shm = f"{roots[1]}/" if len(roots) > 1 else None
    image = DenseImage(out_dir / "capture.bin")
    buf = memoryview(bytearray(4 << 20))
    proc_errors = cache_errors = 0
    try:
        for pid, ranges in procs.items():
            proc_errors += copy_ranges(f"/proc/{pid}/mem", ranges, image, "process", buf)
        for path, ranges in cache.items():
            source = "shm" if shm and path.startswith(shm) else "cache"
            cache_errors += copy_ranges(path, ranges, image, source, buf)
    finally:
        image.close()
    t2 = now()
    sampler.phase = "captured"
    counts = {s: image.counts.get(s, [0, 0]) for s in ("process", "cache", "shm")}
    rec.update(pids=len(pids), capture_snapshot_s=(t1 - t0) / 1e9, capture_write_s=(t2 - t1) / 1e9,
               capture_bytes=(out_dir / "capture.bin").stat().st_size,
               capture_zero_pages=sum(c[1] for c in counts.values()),
               capture_process_read_errors=proc_errors, capture_cache_read_errors=cache_errors,
               process_swapped_pages=pstats["swapped_pages"],
               process_shared_duplicate_pages=pstats["shared_duplicate_pages"],
               container_cache_files=len(cache))
    for s, (kept, zero) in counts.items():
        rec[f"capture_{s}_bytes"] = kept * PAGE
        rec[f"capture_{s}_zero_pages"] = zero


# ---------------------------------------------------------------- codecs (container already removed)

def codec_commands(name: str) -> tuple[list[str], list[str]]:
    """Same command lines as compression_experiment.py."""
    if name == "lz4":
        return ["lz4", "-1", "-c"], ["lz4", "-d", "-c"]
    level = name.split("-")[1]
    return ["zstd", f"-{level}", "-T1", "-c"], ["zstd", "-d", "-T1", "-c"]


def time_codec(command: list[str], source: Path, target: Path) -> float:
    """Wall time for one compressor invocation, reading and writing real files."""
    with source.open("rb") as stdin, target.open("wb") as stdout:
        started = time.monotonic()
        subprocess.run(command, stdin=stdin, stdout=stdout, check=True)
        return time.monotonic() - started


def time_codecs(image: Path, codecs: list[str], repeats: int) -> list[dict]:
    work = image.parent
    packed, restored, empty, empty_packed = (work / n for n in ("tmp.z", "tmp.raw", "empty.bin", "empty.z"))
    empty.touch()
    size = image.stat().st_size
    rows = []
    try:
        for name in codecs:
            compress_cmd, decompress_cmd = codec_commands(name)
            c = [time_codec(compress_cmd, image, packed) for _ in range(repeats)]
            d = [time_codec(decompress_cmd, packed, restored) for _ in range(repeats)]
            small = packed.stat().st_size
            ok = filecmp.cmp(image, restored, shallow=False)
            c0 = statistics.median(time_codec(compress_cmd, empty, empty_packed) for _ in range(repeats))
            d0 = statistics.median(time_codec(decompress_cmd, empty_packed, restored) for _ in range(repeats))
            cs, ds = statistics.median(c), statistics.median(d)
            rows.append({
                "codec": name, "dense_bytes": size, "compressed_bytes": small,
                "ratio": round(size / max(small, 1), 3),
                "compress_s": cs, "decompress_s": ds,
                "compress_startup_s": c0, "decompress_startup_s": d0,
                "compress_net_s": max(cs - c0, 0.0), "decompress_net_s": max(ds - d0, 0.0),
                "compress_mibps": size / MIB / max(cs, 1e-9), "decompress_mibps": size / MIB / max(ds, 1e-9),
                "compress_runs_s": c, "decompress_runs_s": d, "roundtrip_ok": ok,
            })
    finally:
        for p in (packed, restored, empty, empty_packed):
            p.unlink(missing_ok=True)
    return rows


# ---------------------------------------------------------------- driver

SUMMARY = ["task_id", "triggered", "trigger_source", "step", "action", "command", "between_tool_calls",
           "orig_p_usage_bytes", "threshold_bytes", "usage_at_trigger_bytes", "freeze_usage_in_bytes",
           "freeze_rss", "freeze_cache", "freeze_shmem", "freeze_mapped_file", "freeze_kmem_usage_in_bytes",
           "freeze_latency_s", "pids", "capture_bytes", "capture_process_bytes", "capture_cache_bytes",
           "capture_shm_bytes", "capture_zero_pages", "process_swapped_pages", "process_shared_duplicate_pages",
           "capture_process_read_errors", "capture_cache_read_errors", "capture_snapshot_s", "capture_write_s",
           "frozen_call_returncode_after_thaw", "frozen_call_orig_returncode", "actions_replayed", "n_actions",
           "returncode_mismatches", "error"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--percentile", type=float, default=99)
    ap.add_argument("--fraction", type=float, default=0.8)
    ap.add_argument("--sample-interval", type=float, default=0.005)
    ap.add_argument("--codecs", default=CODECS, help=f"comma separated (default {CODECS})")
    ap.add_argument("--repeats", type=int, default=3, help="runs per codec and direction; the median is kept")
    ap.add_argument("--keep-captures", action="store_true", help="keep <task>/capture.bin")
    ap.add_argument("--tasks", nargs="*", help="only these task ids")
    args = ap.parse_args()
    codecs = args.codecs.split(",")
    if os.geteuid() != 0:
        sys.exit("must run as root (cgroup writes, /proc/<pid>/mem, pagemap frames, overlay dirs)")
    for tool in {codec_commands(c)[0][0] for c in codecs}:
        if not shutil.which(tool):
            sys.exit(f"{tool} not found: sudo apt install {tool}")
    sys.setswitchinterval(0.0005)
    out_root = args.out / args.run_dir.name
    out_root.mkdir(parents=True, exist_ok=True)
    tasks = ro.load_tasks(args.run_dir, args.percentile, args.fraction)
    if args.tasks:
        tasks = [t for t in tasks if t["task_id"] in args.tasks]
    print(f"{args.run_dir.name}: {len(tasks)} tasks, codecs {codecs}", flush=True)
    for n, task in enumerate(tasks, 1):
        out_dir = out_root / task["task_id"]
        if (out_dir / "trial.json").exists():
            continue
        out_dir.mkdir(exist_ok=True)
        image = out_dir / "capture.bin"
        try:
            ro.drop_caches()
            rec = ro.run_trial(task, out_dir, args, measure=capture)
        except Exception as e:  # keep going; record the failure
            rec = {"task_id": task["task_id"], "triggered": False, "error": repr(e)}
        if rec.get("capture_bytes"):
            try:
                rec["codecs"] = time_codecs(image, codecs, args.repeats)
            except Exception as e:
                rec["error"] = repr(e)
        if not args.keep_captures:
            image.unlink(missing_ok=True)
        (out_dir / "trial.json").write_text(json.dumps(rec, indent=1, default=str))
        timings = " ".join(f"{c['codec']}={c['compress_s']:.3f}/{c['decompress_s']:.3f}s"
                           for c in rec.get("codecs", []))
        print(f"[{n}/{len(tasks)}] {task['task_id']}: triggered={rec.get('triggered')} "
              f"image={rec.get('capture_bytes', 0) / MIB:.1f}MiB {timings} err={rec.get('error')}", flush=True)
    trials = [json.loads(p.read_text()) for p in sorted(out_root.glob("*/trial.json"))]
    pd.DataFrame(trials).reindex(columns=SUMMARY).to_csv(out_root / "trials.csv", index=False)
    rows = [{"task_id": t["task_id"]} | {k: v for k, v in c.items() if not k.endswith("_runs_s")}
            for t in trials for c in t.get("codecs", [])]
    with (out_root / "compression.csv").open("w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"wrote {out_root / 'trials.csv'} and {out_root / 'compression.csv'}", flush=True)


if __name__ == "__main__":
    main()
