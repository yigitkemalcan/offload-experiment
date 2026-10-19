#!/usr/bin/env python3
"""Share-nothing task containers: each container runs on its own, uncached copy of every file it can read.

Containers started from the same image (or from images with common layers) read the same files on disk,
and the kernel keeps one cached copy of each file page. It is charged to the first container that read it;
every other container, at the same time or later, reads it for free. To give each task its own memory, as
in its own VM but on the same kernel, each task container gets a private copy of its root filesystem:

  prepare  a helper container from the task's image copies, as root with `cp -a` (owners, modes, timestamps
           and hard links kept), every top-level directory of the image (usr, opt, testbed, etc, root, var,
           home, ...: all but proc, sys, dev, run, tmp) into the task's private directory on the host. The
           agent's files (runtime, agent.py, trace, Docker's init) are mounted under /opt in the helper, so
           they are copied too, and so is a second, untouched copy of any --pristine directory (an agent that
           runs its task repeatedly restores /testbed and /tmp from it). It then writes the copy to disk
           (sync -f), evicts every copied file from the
           page cache (posix_fadvise DONTNEED) and reports how much of the copy is still cached (mincore,
           hard links counted once).
  run      mount_args(): the task container bind-mounts each copied directory over the image's
           (-v <private>/usr:/usr ...). Every file it reads (libraries, Python, the repository, the agent) is
           then its own file, read from disk and charged to it alone. Files it writes are its own anyway
           (the overlay's upper layer, or the private copy).
  remove   a helper container deletes the copy (root-owned) after the task.

Still shared: the kernel and its dentry/inode caches (filled by the copy), Docker's own processes, files in
/ itself (e.g. /miniconda.sh, never read by the tasks), /run and /tmp. A copy costs the image's size plus the
runtime's (about 3.6 GB for a SWE-bench task; ext4 has no reflinks) and some seconds, outside the task's
timed window. Everything privileged runs in containers, so no root is needed on the host.

Run inside the helper as: python private_root.py /private [--pristine SRC=DST ...]   (prints one JSON line)
"""
import argparse
import ctypes
import json
import mmap
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

EXCLUDE = {"proc", "sys", "dev", "run", "tmp", "private"}
PAGE = os.sysconf("SC_PAGE_SIZE")
HELPER_PYTHON = "/opt/mswea-agent/python/bin/python3"  # the agent runtime, mounted in the helper


# ---------------------------------------------------------------- host side

def _docker(*args: str) -> str:
    return subprocess.run(["docker", *args], check=True, capture_output=True, text=True).stdout.strip()


def prepare(image: str, private: Path, name: str, files: dict[str, Path], pristine: dict[str, str] | None = None) -> dict:
    """Copy image + files (container path under /opt -> host path) into private/, evicted from the cache.
    pristine: image directory -> container path inside a copied directory, for a second, untouched copy of it
    (e.g. {"/testbed": "/opt/mswea-replay/pristine/testbed"}), to restore it from later."""
    private.mkdir(parents=True)
    mounts = [x for dst, src in files.items() for x in ("-v", f"{Path(src).resolve()}:{dst}:ro")]
    extra = [x for src, dst in (pristine or {}).items() for x in ("--pristine", f"{src}={dst}")]
    out = _docker("run", "--rm", "--name", name, "--network", "none", *mounts,
                  "-v", f"{Path(__file__).resolve()}:/private_root.py:ro", "-v", f"{private.resolve()}:/private",
                  image, HELPER_PYTHON, "-I", "-B", "/private_root.py", "/private", *extra)
    return json.loads(out.splitlines()[-1])


def mount_args(private: Path) -> list[str]:
    return [x for d in sorted(os.listdir(private)) for x in ("-v", f"{(private / d).resolve()}:/{d}")]


def remove(image: str, private: Path, name: str):
    _docker("run", "--rm", "--name", name, "--network", "none", "-v", f"{private.resolve()}:/private",
            image, "find", "/private", "-mindepth", "1", "-delete")
    private.rmdir()


# ---------------------------------------------------------------- helper container side

libc = ctypes.CDLL(None, use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]


def resident_bytes(fd: int, size: int) -> int:
    """Cached bytes of an open file (mincore on a mapping that is never touched, so nothing is read)."""
    if not size:
        return 0
    addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
    if addr in (None, ctypes.c_void_p(-1).value):
        raise OSError(ctypes.get_errno(), "mmap failed")
    try:
        vec = ctypes.create_string_buffer((size + PAGE - 1) // PAGE)
        if libc.mincore(addr, size, vec):
            raise OSError(ctypes.get_errno(), "mincore failed")
        return (len(vec.raw) - vec.raw.count(0)) * PAGE
    finally:
        libc.munmap(addr, size)


def evict(root: Path) -> dict:
    """Write root's files to disk, drop them from the page cache, and measure what is still cached."""
    subprocess.run(["sync", "-f", str(root)], check=True)
    seen, files, size, resident = set(), 0, 0, 0
    for dirpath, _, names in os.walk(root):
        for name in names:
            path = os.path.join(dirpath, name)
            st = os.lstat(path)
            if not stat.S_ISREG(st.st_mode) or (st.st_dev, st.st_ino) in seen:
                continue
            seen.add((st.st_dev, st.st_ino))
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NOATIME)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                resident += resident_bytes(fd, st.st_size)
            finally:
                os.close(fd)
            files, size = files + 1, size + st.st_size
    return {"files": files, "bytes": size, "resident_bytes": resident}


def copy_root(dest: Path) -> list[str]:
    dirs = sorted(d for d in os.listdir("/") if d not in EXCLUDE and os.path.isdir(f"/{d}") and not os.path.islink(f"/{d}"))
    for d in dirs:
        subprocess.run(["cp", "-a", f"/{d}", str(dest / d)], check=True)
    return dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dest", type=Path)
    ap.add_argument("--pristine", action="append", default=[], metavar="SRC=DST")
    args = ap.parse_args()
    t0 = time.monotonic()
    dirs = copy_root(args.dest)
    for src, dst in (x.split("=", 1) for x in args.pristine):
        (args.dest / dst.lstrip("/")).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["cp", "-a", src, str(args.dest / dst.lstrip("/"))], check=True)
    t1 = time.monotonic()
    stats = evict(args.dest)
    print(json.dumps({"dirs": dirs, "copy_s": round(t1 - t0, 3), "evict_s": round(time.monotonic() - t1, 3), **stats}))


if __name__ == "__main__":
    main()
