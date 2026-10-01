"""Page access tracking on a toy process driven step by step; no Docker. Needs root (idle bitmap, frames):

    sudo env REUSE_KERNEL_TESTS=1 ../.venv/bin/python -m unittest test_replay_reuse -v       # from this directory

Works in a directory next to this file (a disk filesystem: eviction and readahead behave as in the layers).
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

import replay_reuse as rr

PAGE = rr.PAGE
MIB = 1 << 20

# Holds a 4 MiB small-page anonymous buffer, 4 MiB meant for huge pages, a 1 MiB memfd and a 1 MiB mapping of a
# deleted file outside the scanned roots. Then runs one line of commands per step and answers "ok":
#   a/B/H read a.bin / b2.bin / h.bin     R rename b.bin -> b2.bin     L hard link h.bin -> b2.bin
#   P replace a.bin (new file, os.replace)                            T truncate and rewrite a.bin in place
#   m write the first half of the small buffer   t touch one byte of the first huge page   f write the memfd
#   s read the shm file   c create c.bin   e read e.bin, then evict it   r read the first page of r.bin
#   q read all of r.bin   W overwrite one page of b2.bin in place (pwrite at 1 MiB)
#   x run a short-lived process that allocates 16 MiB and reads a.bin   z start a background sleep
WORKLOAD = r'''
import ctypes, mmap, os, subprocess, sys
root, shm, outside = sys.argv[1:]
small = mmap.mmap(-1, 4 << 20, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS); small.madvise(mmap.MADV_NOHUGEPAGE); small.write(b"x" * (4 << 20))
raw = mmap.mmap(-1, 6 << 20, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS); base = ctypes.addressof(ctypes.c_char.from_buffer(raw))
off = -base % (2 << 20); raw.madvise(mmap.MADV_HUGEPAGE, off, 4 << 20); raw[off:off + (4 << 20)] = b"y" * (4 << 20)
mfd = os.memfd_create("toy"); os.ftruncate(mfd, 1 << 20); mem = mmap.mmap(mfd, 1 << 20); mem.write(b"m" * (1 << 20))
with open(f"{outside}/d.bin", "wb") as f: f.write(os.urandom(1 << 20))
d = open(f"{outside}/d.bin", "rb"); dmap = mmap.mmap(d.fileno(), 0, prot=mmap.PROT_READ); dmap.read(); os.unlink(f"{outside}/d.bin")
print(ctypes.addressof(ctypes.c_char.from_buffer(small)), base + off, flush=True)
def read(name): open(f"{root}/{name}", "rb").read()
for line in sys.stdin:
    for c in line.strip():
        if c == "a": read("a.bin")
        elif c == "B": read("b2.bin")
        elif c == "H": read("h.bin")
        elif c == "R": os.rename(f"{root}/b.bin", f"{root}/b2.bin")
        elif c == "L": os.link(f"{root}/b2.bin", f"{root}/h.bin")
        elif c == "P":
            open(f"{root}/a.tmp", "wb").write(os.urandom(4 << 20)); os.replace(f"{root}/a.tmp", f"{root}/a.bin")
        elif c == "T": open(f"{root}/a.bin", "wb").write(os.urandom(4 << 20))
        elif c == "m": small[:2 << 20] = b"z" * (2 << 20)
        elif c == "t": raw[off] = 1
        elif c == "f": mem[:] = b"n" * (1 << 20)
        elif c == "s": open(f"{shm}/s.bin", "rb").read()
        elif c == "c": open(f"{root}/c.bin", "wb").write(b"c" * (1 << 20))
        elif c == "e":
            fd = os.open(f"{root}/e.bin", os.O_RDONLY); os.read(fd, 1 << 20)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
        elif c == "r":
            fd = os.open(f"{root}/r.bin", os.O_RDONLY); os.read(fd, 4096); os.close(fd)
        elif c == "q": open(f"{root}/r.bin", "rb").read()
        elif c == "W":
            fd = os.open(f"{root}/b2.bin", os.O_WRONLY); os.pwrite(fd, b"w" * 4096, 1 << 20); os.close(fd)
        elif c == "z": sleeper = subprocess.Popen(["sleep", "600"])
        elif c == "x":
            subprocess.run([sys.executable, "-c", f"b = bytearray(16 << 20); open('{root}/a.bin', 'rb').read()"], check=True)
    print("ok", flush=True)
'''


def ident(path: Path):
    fd = os.open(path, os.O_RDONLY)
    try:
        return rr.file_identity(fd)
    finally:
        os.close(fd)


class AccountingTest(unittest.TestCase):
    """Exercise full boundary accounting without touching kernel tracking state."""

    def setUp(self):
        self.tracker = rr.ReuseTracker([], None)
        self.identity = (1, 2, 3)
        self.tracker.files.id(self.identity, {"path": "/a", "kind": "file", "layer": "test"})
        self.tracker.procs.id((10, 20), {"path": "process", "kind": "anon", "layer": ""})
        self.keys = np.array([0], np.uint64)
        self.pfns = np.array([100], np.uint64)
        self.anon = False
        self.is_idle = False
        patches = [
            patch.object(rr, "drain_lru"),
            patch.object(rr, "scan_files", side_effect=lambda *a: (set(), [])),
            patch.object(rr, "process_pages", side_effect=self.process_pages),
            patch.object(rr, "file_frames", side_effect=self.file_frames),
            patch.object(rr, "read_u64", side_effect=lambda path, xs: np.zeros(len(xs), np.uint64)),
            patch.object(rr, "heads", side_effect=lambda pfns, flags: pfns),
            patch.object(rr, "idle", side_effect=lambda xs: np.full(len(xs), self.is_idle)),
            patch.object(rr, "mark_idle", side_effect=self.mark),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.step_number = 0

    def process_pages(self, *args):
        a_k, a_p = (self.keys, self.pfns) if self.anon else (rr.EMPTY, rr.EMPTY)
        return a_k, a_p, rr.EMPTY, rr.EMPTY, []

    def file_frames(self, *args):
        return (rr.EMPTY, rr.EMPTY) if self.anon else (self.keys, self.pfns)

    def mark(self, *args):
        self.is_idle = True

    def distances(self):
        u = self.tracker.uses[-1]
        return u["reuse_distance_steps"].tolist(), u["reuse_distance_calls"].tolist()

    def boundary(self, accessed=True):
        self.is_idle = not accessed
        row = self.tracker.boundary([], self.step_number)
        self.step_number += 1
        if self.step_number > 1:
            self.assertEqual(row["accessed_pages"], row["reused_pages"] + row["not_reused_pages"])
        return row

    def test_startup_and_idle_pages(self):
        self.boundary()
        row = self.boundary()
        self.assertEqual((row["reused_pages"], row["not_reused_pages"]), (0, 1))
        self.assertEqual(self.boundary(False)["accessed_pages"], 0)
        self.assertEqual(self.boundary()["reused_pages"], 1)
        self.assertEqual(self.distances(), ([2], [2]))

    def test_load_then_two_reuses_and_export(self):
        self.keys = self.pfns = rr.EMPTY
        self.boundary()
        self.step_number = 3
        self.keys, self.pfns = np.array([0], np.uint64), np.array([100], np.uint64)
        self.assertEqual(self.boundary()["not_reused_pages"], 1)
        self.boundary(False)
        self.assertEqual(self.boundary()["reused_pages"], 1)
        self.boundary(False)
        self.assertEqual(self.boundary()["reused_pages"], 1)
        with tempfile.TemporaryDirectory() as path:
            self.tracker.save(Path(path))
            with np.load(Path(path) / "accessed.npz") as data:
                self.assertEqual(data["step"].tolist(), [3, 5, 7])
                self.assertEqual(data["reused"].tolist(), [False, True, True])
                self.assertEqual(data["reuse_distance_steps"].tolist(), [-1, 2, 2])
                self.assertEqual(data["reuse_distance_calls"].tolist(), [-1, 2, 2])
            hist = pd.read_csv(Path(path) / "reuse_distances.csv")
            self.assertEqual(hist.page_calls.tolist(), [2])
            objects = pd.read_csv(Path(path) / "objects.csv")
            self.assertAlmostEqual(objects.reuse_fraction.iloc[0], 2 / 3)

    def test_two_calls_in_same_step(self):
        self.boundary()
        self.step_number = 3
        self.boundary()
        self.step_number = 3
        row = self.boundary()
        self.assertEqual(row["reused_pages"], 1)
        self.assertEqual(self.distances(), ([0], [1]))

    def test_single_idle_file_page_between_used_pages_counts(self):
        # pages 0-4 of one file; per-page idle state after the call
        self.keys = self.pfns = np.arange(5, dtype=np.uint64)
        self.boundary()
        pattern = np.array([False, True, False, True, True])  # 1 is a gap, 3-4 is not
        with patch.object(rr, "idle", side_effect=lambda xs: pattern[:len(xs)] if len(xs) == 5 else np.ones(len(xs), bool)):
            self.tracker.boundary([], 1)
        self.assertEqual(self.tracker.uses[-1]["key"].tolist(), [0, 1, 2])

    def test_startup_row_has_all_columns(self):
        self.assertEqual(self.boundary().keys(), self.boundary().keys())

    def test_no_diagnostic_categories(self):
        self.boundary()
        row = self.boundary()
        self.assertFalse(any(word in k for k in row for word in
                            ("unknown", "frame_changed", "untracked", "mark_failed", "reused_prev")))
        page_names = {k for k in row if k.endswith("_pages")}
        expected = {f"{name}_pages" for name in rr.CATEGORIES}
        expected |= {f"{name}_{kind}_pages" for name in rr.CATEGORIES for kind in rr.KINDS}
        self.assertEqual(page_names, expected)

    def test_file_reappearance(self):
        self.boundary()
        self.boundary()
        self.keys = self.pfns = rr.EMPTY
        self.boundary()
        self.keys, self.pfns = np.array([0], np.uint64), np.array([200], np.uint64)
        self.assertEqual(self.boundary()["reused_pages"], 1)

    def test_anon_page_reuse(self):
        self.anon = True
        self.keys = np.array([int(rr.ANON) | 123], np.uint64)
        self.boundary()
        self.boundary()
        row = self.boundary()
        self.assertEqual((row["reused_anon_pages"], row["not_reused_pages"]), (1, 0))


class ReplayTest(unittest.TestCase):
    def test_boundary_after_every_tool_call(self):
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        import json
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            (root / "memory.max_usage_in_bytes").write_text("100")
            cg = MagicMock()
            cg.mem = root
            cg.wait_frozen.return_value = True
            cg.usage.return_value = 100
            cg.pids.return_value = [10]
            tracker = MagicMock()
            tracker.uses = [{"key": np.array([0], np.uint64)}] * 3
            def boundary(pids, step, action):
                row = {"call_index": tracker.boundary.call_count - 1, "step": step,
                       "action": action, "pids": 1, "boundary_s": 0.0}
                for name in rr.CATEGORIES:
                    row[f"{name}_pages"] = 1
                    for kind in rr.KINDS:
                        row[f"{name}_{kind}_pages"] = int(kind == "file")
                return row
            tracker.boundary.side_effect = boundary
            env = {"image": "test", "cwd": "/", "container_timeout": "2h", "env": {},
                   "interpreter": ["bash", "-c"], "timeout": 10}
            actions = [{"step": step, "action": action, "command": "true", "orig_returncode": 0}
                       for step, action in [(1, 1), (1, 2), (2, 1)]]
            inspect = [{"GraphDriver": {"Data": {"UpperDir": "/upper", "LowerDir": "/lower"}},
                        "State": {"Pid": 10}}]
            with patch.object(rr.ro, "docker", side_effect=[SimpleNamespace(stdout="cid"),
                       SimpleNamespace(stdout=json.dumps(inspect))]), \
                 patch.object(rr.ro, "Cgroup", return_value=cg), \
                 patch.object(rr, "ReuseTracker", return_value=tracker), \
                 patch.object(rr.subprocess, "run", return_value=SimpleNamespace(returncode=0)):
                rec, rows = rr.run_task({"task_id": "test", "env": env, "actions": actions}, root)
            self.assertEqual([(r["step"], r["action"]) for r in rows], [(0, 0), (1, 1), (1, 2), (2, 1)])
            self.assertEqual(cg.freeze.call_count, 4)
            self.assertEqual((rec["calls"], rec["steps"]), (3, 2))


class IdentityTest(unittest.TestCase):
    def test_live_mapping_keeps_walked_identity_after_unlink(self):
        with tempfile.TemporaryDirectory() as path:
            p = Path(path) / "file"
            p.write_bytes(b"content")
            fd = os.open(p, os.O_RDONLY)
            try:
                identity = rr.file_identity(fd)
                p.unlink()
                # A duplicate descriptor simulates the live /proc/pid/map_files reference.
                with patch.object(rr.os, "open", side_effect=lambda *a: os.dup(fd)):
                    self.assertEqual(rr.mapped_identity(123, "1000-2000", *identity[:2]), identity)
            finally:
                os.close(fd)

    def test_replacement_and_rename_identity(self):
        with tempfile.TemporaryDirectory() as path:
            a, b = Path(path) / "a", Path(path) / "b"
            a.write_bytes(b"a")
            original = ident(a)
            a.rename(b)
            self.assertEqual(ident(b), original)
            a.write_bytes(b"replacement")
            a.replace(b)
            self.assertNotEqual(ident(b), original)

    def test_private_and_shared_mapping_classification(self):
        import ctypes
        import mmap
        with mmap.mmap(-1, PAGE, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS) as private, \
                mmap.mmap(-1, PAGE) as shared:
            private[0] = shared[0] = 1
            private_vpn = ctypes.addressof(ctypes.c_char.from_buffer(private)) // PAGE
            shared_vpn = ctypes.addressof(ctypes.c_char.from_buffer(shared)) // PAGE
            files = rr.Registry()
            a_k, *rest = rr.process_pages([os.getpid()], rr.Registry(), files, set())
            virtual = a_k & np.uint64((1 << 36) - 1)
            self.assertTrue(np.any(virtual == private_vpn))
            self.assertFalse(np.any(virtual == shared_vpn))
            self.assertTrue(any("/dev/zero" in it["path"] for it in files.items))


@unittest.skipUnless(os.geteuid() == 0 and os.getenv("REUSE_KERNEL_TESTS") == "1",
                     "kernel tests require root and REUSE_KERNEL_TESTS=1")
class ReuseTest(unittest.TestCase):
    """Toy process in its own memory cgroup; each assertion is about one object (file, memfd, buffer region)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=Path(__file__).parent, prefix="test-tmp-"))
        self.addCleanup(shutil.rmtree, self.tmp)
        if not any(line.split()[4] == "/dev/shm" and " - tmpfs " in line
                   for line in Path("/proc/self/mountinfo").read_text().splitlines()):
            self.skipTest("requires tmpfs mounted at /dev/shm")
        self.shm = Path(tempfile.mkdtemp(dir="/dev/shm", prefix="reuse-test-"))
        self.addCleanup(shutil.rmtree, self.shm)
        self.root, self.outside = self.tmp / "rootfs", self.tmp / "outside"
        for d in (self.root, self.outside):
            d.mkdir()
        for name, n in (("a.bin", 4), ("b.bin", 4), ("e.bin", 1), ("r.bin", 8)):
            (self.root / name).write_bytes(os.urandom(n * MIB))
        (self.shm / "s.bin").write_bytes(os.urandom(MIB))
        subprocess.run(["sync"], check=True)
        fd = os.open(self.root / "r.bin", os.O_RDONLY)  # r.bin starts out of the cache
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
        # Processes are discovered from a memory cgroup, as the replay does for the container.
        self.cgroup = Path(f"/sys/fs/cgroup/memory/reuse-test-{os.getpid()}")
        self.cgroup.mkdir()
        self.proc = subprocess.Popen([sys.executable, "-c", WORKLOAD, *map(str, (self.root, self.shm, self.outside))],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.addCleanup(self.remove_cgroup)
        (self.cgroup / "cgroup.procs").write_text(str(self.proc.pid))
        self.small, self.huge = (int(x) // PAGE for x in self.proc.stdout.readline().split())
        self.tracker = rr.ReuseTracker([self.root], self.shm)
        self.rows = [self.tracker.boundary(self.pids(), 0)]

    def pids(self) -> list[int]:
        return [int(x) for x in (self.cgroup / "cgroup.procs").read_text().split()]

    def remove_cgroup(self):
        for pid in self.pids():
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass
        self.proc.kill()  # in case it never joined the cgroup
        self.proc.wait()
        for _ in range(100):
            try:
                self.cgroup.rmdir()
                return
            except OSError:
                time.sleep(0.05)

    def step(self, commands: str) -> dict:
        self.proc.stdin.write(commands + "\n")
        self.proc.stdin.flush()
        self.assertEqual(self.proc.stdout.readline().strip(), "ok")
        self.rows.append(self.tracker.boundary(self.pids(), len(self.rows)))
        self.check_row(self.rows[-1])
        return self.rows[-1]

    def fid(self, name: str) -> int:
        return self.tracker.files[ident(self.root / name)]

    def memfd(self) -> int:
        return next(i for i, it in enumerate(self.tracker.files.items) if it["path"].startswith("/memfd:toy"))

    def pages(self, keys: np.ndarray, obj) -> int:
        """Keys of a file id, or of a region of the small / huge buffer ("small", "small2", "huge1", "huge2")."""
        if isinstance(obj, int):
            f = keys[(keys & rr.ANON) == 0]
            return int(((f >> np.uint64(32)) == obj).sum())
        vpn = (keys[(keys & rr.ANON) != 0] & np.uint64((1 << 36) - 1)).astype(np.int64)
        lo = {"small": self.small, "small2": self.small + 512, "huge1": self.huge, "huge2": self.huge + 512}[obj]
        return int(((vpn >= lo) & (vpn < lo + 512)).sum())

    def acc(self, obj) -> int:
        return self.pages(self.tracker.uses[-1]["key"], obj)

    def reused(self, obj) -> int:
        u = self.tracker.uses[-1]
        return self.pages(u["key"][u["reused"]], obj)

    def frames(self, fid: int) -> dict[int, int]:
        """key -> frame of this file's resident pages at the last boundary."""
        keys, pfns = self.tracker.prev["key"], self.tracker.prev["pfn"]
        mine = ((keys & rr.ANON) == 0) & ((keys >> np.uint64(32)) == fid)
        return dict(zip(keys[mine].tolist(), pfns[mine].tolist()))

    def is_thp(self) -> bool:
        pm = os.open(f"/proc/{self.proc.pid}/pagemap", os.O_RDONLY)
        e = np.frombuffer(os.pread(pm, 8, self.huge * 8), np.uint64)
        os.close(pm)
        return bool(rr.read_u64(rr.KPAGEFLAGS, e & np.uint64(rr.PFN_MASK))[0] & np.uint64(1 << 22))

    def check_row(self, r: dict):
        for k in rr.KINDS:
            self.assertEqual(r[f"accessed_{k}_pages"], r[f"reused_{k}_pages"] + r[f"not_reused_{k}_pages"])

    def test_steps(self):
        start = self.rows[0]
        self.assertEqual(start["resident_file_pages"], 2048 + 256)  # a, b, e (r was evicted)
        self.assertEqual(start["resident_shm_pages"], 256)
        self.assertEqual(start["accessed_pages"], 0)  # boundary 0 is inventory only
        a0, b0, mfd = self.fid("a.bin"), self.fid("b.bin"), self.memfd()
        self.assertEqual(self.tracker.files.items[mfd]["kind"], "mapped")

        r = self.step("amcx")  # first task step: nothing is reuse
        self.assertEqual(r["reused_pages"], 0)
        self.assertEqual(self.acc(a0), 1024)
        self.assertEqual(self.acc(self.fid("c.bin")), 256)  # first loading
        self.assertEqual((self.acc("small"), self.acc("small2")), (512, 0))
        self.assertEqual(self.acc("huge1") + self.acc("huge2"), 0)
        # the short-lived child of "x" has exited: only the toy process is in the cgroup
        self.assertEqual({p["pid"] for p in self.tracker.processes if p["step"] == 1}, {self.proc.pid})

        r = self.step("tsfz")  # c.bin was fresh at the last boundary; it must have been marked idle anyway
        self.assertEqual(self.acc(self.fid("c.bin")), 0)
        self.assertEqual(self.acc(a0), 0)
        self.assertEqual(r["accessed_shm_pages"], 256)
        self.assertEqual((self.acc(mfd), self.reused(mfd)), (256, 0))
        self.assertEqual(self.acc("huge1"), 512 if self.is_thp() else 1)  # one flag per huge page
        self.assertEqual(self.acc("huge2"), 0)
        self.assertEqual(len({p["pid"] for p in self.tracker.processes if p["step"] == 2}), 2)  # + background sleep

        r = self.step("amf")  # repeated read, persistent anonymous buffer, memfd: reuse from steps 1-2
        self.assertEqual((self.acc(a0), self.reused(a0)), (1024, 1024))
        self.assertEqual((self.acc("small"), self.reused("small")), (512, 512))
        self.assertEqual((self.acc(mfd), self.reused(mfd)), (256, 256))

        r = self.step("RB")  # rename keeps the file; b stayed idle since startup: no reuse
        self.assertEqual(self.fid("b2.bin"), b0)
        self.assertEqual((self.acc(b0), self.reused(b0)), (1024, 0))

        resident = r["resident_file_pages"]
        r = self.step("LH")  # hard link: same file, counted once, reuse from the previous step
        self.assertEqual(self.fid("h.bin"), b0)
        self.assertEqual(r["resident_file_pages"], resident)
        self.assertEqual((self.acc(b0), self.reused(b0)), (1024, 1024))

        key = (b0 << 32) | 256
        before = self.frames(b0)[key]
        r = self.step("W")  # in-place overwrite of one page: same frame, observed, reuse; file annotated
        self.assertEqual(self.frames(b0)[key], before)
        self.assertEqual((self.acc(b0), self.reused(b0)), (1, 1))

        r = self.step("P")  # replaced at the same path: a new file
        a1 = self.fid("a.bin")
        self.assertNotEqual(a1, a0)
        self.assertEqual((self.acc(a1), self.reused(a1)), (1024, 0))

        r = self.step("a")  # loaded in one step, read in the next: reuse
        self.assertEqual((self.acc(a1), self.reused(a1)), (1024, 1024))

        r = self.step("T")  # same logical file offsets are reused, regardless of frames
        self.assertEqual((self.acc(a1), self.reused(a1)), (1024, 1024))

        r = self.step("e")  # read then evicted within the step: the blind spot, seen only as disappeared
        self.assertEqual(self.acc(self.fid("e.bin")), 0)

        r = self.step("r")  # one page read, neighbours read ahead: all loaded pages count
        rf = self.fid("r.bin")
        loaded = len(self.frames(rf))
        self.assertEqual((self.acc(rf), self.reused(rf)), (loaded, 0))
        self.assertGreater(loaded, 1)

        r = self.step("q")  # the whole file read: the pages loaded last step are reuse
        self.assertEqual((self.acc(rf), self.reused(rf)), (2048, loaded))

        out = self.tmp / "out"
        out.mkdir()
        self.tracker.save(out)
        objs = pd.read_csv(out / "objects.csv")
        b = objs[objs.object == "/b.bin"].iloc[0]
        self.assertEqual(b.reused_page_calls, 1025)
        procs = pd.read_csv(out / "processes.csv")
        self.assertIn(self.proc.pid, set(procs.pid))


if __name__ == "__main__":
    unittest.main()
