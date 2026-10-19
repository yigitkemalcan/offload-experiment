"""private_root.py with real containers (about a minute and 7 GB of temporary copies; no root). From this directory:

    ../.venv/bin/python -B -m unittest test_private -v

Two containers from the same image read the same large file. Shared (the image's files): the page cache
is charged once in total, to whichever container read it first. Private: the copy is uncached after
prepare, and each container is charged the whole file, also when both read it at the same time.
"""
import json
import subprocess
import time
import unittest
from pathlib import Path

import private_root as pr
import replay_in_container as ric

IMAGE = "docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"
MIB = 1 << 20


def cache_bytes(cid: str) -> int:
    stat = Path(f"/sys/fs/cgroup/memory/docker/{cid}/memory.stat").read_text()
    return int(dict(line.split() for line in stat.splitlines())["cache"])


def largest_file() -> tuple[str, int]:
    out = ric.docker("run", "--rm", "--network", "none", IMAGE, "bash", "-c",
                     "find /opt/miniconda3 -type f -size +40M -printf '%s %p\\n' | sort -n | tail -1")
    size, path = out.split(" ", 1)
    return path, int(size)


def start_reader(name: str, path: str, *run_args: str) -> str:
    return ric.docker("run", "-d", "--name", name, "--network", "none", *run_args, IMAGE,
                      "bash", "-c", f"cat {path} > /dev/null && touch /tmp/read && sleep 60")


def wait_read(cid: str):
    deadline = time.monotonic() + 60
    while subprocess.run(["docker", "exec", cid, "test", "-e", "/tmp/read"]).returncode:
        assert time.monotonic() < deadline, "reader did not finish"
        time.sleep(0.2)


class PrivateRootTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path, cls.size = largest_file()
        cls.containers, cls.privates = [], []

    @classmethod
    def tearDownClass(cls):
        if cls.containers:
            subprocess.run(["docker", "rm", "-f", *cls.containers], capture_output=True)
        for i, private in enumerate(cls.privates):
            if private.exists():
                pr.remove(IMAGE, private, f"agentrm-test-{i}")

    def test_shared_files_are_charged_once(self):
        a = start_reader("agentin-test-shared-a", self.path)
        self.containers.append(a)
        wait_read(a)
        b = start_reader("agentin-test-shared-b", self.path)
        self.containers.append(b)
        wait_read(b)
        print(f"\nshared: {self.path} {self.size / MIB:.0f} MiB, charged A {cache_bytes(a) / MIB:.0f} MiB, B {cache_bytes(b) / MIB:.0f} MiB")
        self.assertLess(cache_bytes(b), 0.1 * self.size)
        self.assertLess(cache_bytes(a) + cache_bytes(b), 1.2 * self.size)

    def test_private_copies_are_cold_and_charged_to_each(self):
        files = {f"/opt/mswea-replay/{k}": v for k, v in [("agent.py", ric.ROOT / "agent.py"), ("docker-init", ric.DOCKER_INIT)]}
        files["/opt/mswea-agent"] = ric.RUNTIME
        cids = []
        for i in range(2):
            private = ric.PRIVATE / f"test-{i}"
            self.privates.append(private)
            stats = pr.prepare(IMAGE, private, f"agentprep-test-{i}", files)
            self.assertEqual(stats["resident_bytes"], 0, json.dumps(stats))
            self.assertIn("testbed", stats["dirs"])
        for i, private in enumerate(self.privates):  # both read at the same time
            cids.append(start_reader(f"agentin-test-private-{i}", self.path, *pr.mount_args(private)))
            self.containers.append(cids[-1])
        for cid in cids:
            wait_read(cid)
        charged = [cache_bytes(cid) for cid in cids]
        print(f"\nprivate: {self.path} {self.size / MIB:.0f} MiB, charged {[round(c / MIB) for c in charged]} MiB")
        for c in charged:
            self.assertGreater(c, 0.95 * self.size)


if __name__ == "__main__":
    unittest.main()
