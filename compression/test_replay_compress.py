"""Capture and codec checks on a toy process tree; no Docker, no root needed.

    ../.venv/bin/python -m unittest test_replay_compress -v          # from this directory
    sudo ../.venv/bin/python -m unittest test_replay_compress -v     # also checks fork de-duplication

Without root the pagemap hides page frames, so pages shared between forked processes are copied once
per process; as root they must be copied once.
"""
import collections
import ctypes
import hashlib
import mmap
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import replay_compress as rc

PAGE = rc.PAGE
MIB = 1 << 20

# Parent: 8 MiB random private anonymous, 1 MiB random shared anonymous, 4 MiB of written zeros, a 2 MiB file mmapped and read, a 1 MiB file in
# the fake /dev/shm; then forks (the child shares the random pages copy-on-write) and both wait.
WORKLOAD = r'''
import mmap, os, sys, time
root, shm, ready = sys.argv[1:]
anon = mmap.mmap(-1, 8 << 20, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS); anon.write(os.urandom(8 << 20))
shared = mmap.mmap(-1, 1 << 20, flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS); shared.write(os.urandom(1 << 20))
zeros = mmap.mmap(-1, 4 << 20, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS); zeros.write(bytes(4 << 20))
with open(f"{root}/../shared.bin", "wb") as g: g.write(shared)
with open(f"{root}/mapped.bin", "wb") as f: f.write(os.urandom(2 << 20))
f = open(f"{root}/mapped.bin", "rb"); m = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ); m.read()
with open(f"{shm}/segment", "wb") as g: g.write(os.urandom(1 << 20))
with open(f"{root}/../anon.bin", "wb") as g: g.write(anon)  # outside the rootfs: tells the test the bytes
child = os.fork()
if child:
    open(ready, "w").write(str(child))
time.sleep(600)
'''


def page_hashes(data: bytes) -> list[bytes]:
    return [hashlib.blake2b(data[i:i + PAGE], digest_size=16).digest() for i in range(0, len(data), PAGE)]


class CaptureTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root, self.shm, self.out = self.tmp / "rootfs", self.tmp / "shm", self.tmp / "out"
        for d in (self.root, self.shm, self.out):
            d.mkdir()
        ready = self.tmp / "ready"
        self.proc = subprocess.Popen([sys.executable, "-c", WORKLOAD, str(self.root), str(self.shm), str(ready)])
        deadline = time.monotonic() + 30
        while not (ready.exists() and ready.read_text()):
            self.assertLess(time.monotonic(), deadline, "workload did not start")
            time.sleep(0.05)
        self.pids = [self.proc.pid, int(ready.read_text())]
        for pid in self.pids:
            os.kill(pid, signal.SIGSTOP)  # stands in for the cgroup freezer

    def tearDown(self):
        for pid in self.pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.proc.wait()
        shutil.rmtree(self.tmp)

    def test_capture(self):
        cg = SimpleNamespace(stat=lambda: {"usage_in_bytes": 0}, pids=lambda: self.pids)
        rec = {}
        rc.capture(cg, "cid", None, [self.root, self.shm], {}, SimpleNamespace(phase=None), rec, self.out)
        image = (self.out / "capture.bin").read_bytes()
        self.assertEqual(len(image), rec["capture_bytes"])
        self.assertEqual(len(image) % PAGE, 0)
        counts = collections.Counter(page_hashes(image))
        self.assertNotIn(hashlib.blake2b(bytes(PAGE), digest_size=16).digest(), counts, "zero pages kept")
        self.assertGreaterEqual(rec["capture_process_zero_pages"], (4 << 20) // PAGE)

        # the random anonymous buffer: in the image, once per process without root, once in total with root
        anon = page_hashes((self.tmp / "anon.bin").read_bytes())
        expected = 1 if os.geteuid() == 0 else 2
        self.assertEqual({counts[h] for h in anon}, {expected}, "anonymous pages missing or duplicated")
        # shared anonymous memory has no file for the cache walk, so it comes from the processes; fork does
        # not copy page-table entries of shared mappings, so only the parent has them present
        shared = page_hashes((self.tmp / "shared.bin").read_bytes())
        self.assertEqual({counts[h] for h in shared}, {1}, "shared anonymous pages missing or duplicated")
        # the mmapped file: copied once, from the page cache, although both processes map it
        mapped = page_hashes((self.root / "mapped.bin").read_bytes())
        self.assertEqual({counts[h] for h in mapped}, {1}, "mapped file pages missing or duplicated")
        self.assertGreaterEqual(rec["capture_cache_bytes"], 2 * MIB)
        self.assertGreaterEqual(rec["capture_shm_bytes"], 1 * MIB)
        self.assertGreaterEqual(rec["capture_process_bytes"], expected * 8 * MIB + MIB)
        self.assertEqual(rec["capture_process_read_errors"], 0)
        self.assertEqual(rec["capture_cache_read_errors"], 0)
        print(f"\n  image {len(image) / MIB:.1f} MiB: process {rec['capture_process_bytes'] / MIB:.1f}, "
              f"cache {rec['capture_cache_bytes'] / MIB:.1f}, shm {rec['capture_shm_bytes'] / MIB:.1f}, "
              f"zero pages dropped {rec['capture_zero_pages']}, "
              f"fork duplicates dropped {rec['process_shared_duplicate_pages']}")


class DenseImageTest(unittest.TestCase):
    def test_partial_last_page_is_padded(self):
        with tempfile.TemporaryDirectory() as d:
            img = rc.DenseImage(Path(d) / "x")
            img.add("cache", memoryview(b"\1" * (PAGE + 10)))
            img.add("cache", memoryview(bytes(3 * PAGE)))
            img.close()
            data = (Path(d) / "x").read_bytes()
        self.assertEqual(len(data), 2 * PAGE)
        self.assertEqual(data[PAGE:PAGE + 10], b"\1" * 10)
        self.assertEqual(data[PAGE + 10:], bytes(PAGE - 10))
        self.assertEqual(img.counts["cache"], [2, 3])


class CopyRangesTest(unittest.TestCase):
    def test_unreadable_page_is_skipped_not_the_rest(self):
        """A hole in the middle of a /proc/<pid>/mem range costs one page, not the rest of the range."""
        m = mmap.mmap(-1, 3 * PAGE)
        for i in range(3):
            m[i * PAGE:(i + 1) * PAGE] = bytes([i + 1]) * PAGE
        ref = ctypes.c_char.from_buffer(m)
        addr = ctypes.addressof(ref)
        del ref
        libc = ctypes.CDLL(None, use_errno=True)
        libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.assertEqual(libc.munmap(addr + PAGE, PAGE), 0)
        try:
            with tempfile.TemporaryDirectory() as d:
                img = rc.DenseImage(Path(d) / "x")
                errors = rc.copy_ranges(f"/proc/{os.getpid()}/mem", [(addr, 3 * PAGE)], img, "process",
                                        memoryview(bytearray(4 << 20)))
                img.close()
                data = (Path(d) / "x").read_bytes()
        finally:
            m.close()  # munmap of the whole range; the hole is already unmapped
        self.assertEqual(errors, 1)
        self.assertEqual(data, b"\1" * PAGE + b"\3" * PAGE)


class CodecTest(unittest.TestCase):
    def test_roundtrip(self):
        codecs = [c for c in ("lz4", "zstd-1", "zstd-19") if shutil.which(rc.codec_commands(c)[0][0])]
        self.assertTrue(codecs, "neither lz4 nor zstd on PATH")
        with tempfile.TemporaryDirectory() as d:
            image = Path(d) / "capture.bin"
            image.write_bytes(os.urandom(2 * MIB) + b"text " * (1 * MIB))
            rows = rc.time_codecs(image, codecs, repeats=2)
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()), ["capture.bin"], "temp files left")
        for r in rows:
            self.assertTrue(r["roundtrip_ok"], r["codec"])
            self.assertGreater(r["ratio"], 1.2, r["codec"])
            self.assertEqual(len(r["compress_runs_s"]), 2)
            self.assertLessEqual(r["compress_net_s"], r["compress_s"])
            print(f"\n  {r['codec']}: ratio {r['ratio']:.2f} compress {r['compress_s'] * 1e3:.1f} ms "
                  f"(start-up {r['compress_startup_s'] * 1e3:.1f}) decompress {r['decompress_s'] * 1e3:.1f} ms "
                  f"(start-up {r['decompress_startup_s'] * 1e3:.1f})", end="")


if __name__ == "__main__":
    unittest.main()
