"""Unprivileged tests: timing, boundary ordering, demotion loop and slowdown pairing; no Docker, no root.

    MSWEA_SILENT_STARTUP=1 PYTHONDONTWRITEBYTECODE=1 ../../mini-swe-agent/.venv/bin/python -m unittest test_replay_lazy -v
"""
import errno
import json
import tempfile
import time
import unittest
from pathlib import Path

import pandas as pd

import analyze_lazy as al
import replay_lazy as rl
from minisweagent.exceptions import Submitted


class FakeEnv:
    def __init__(self, delay=0.05, submit_on=None):
        self.delay, self.submit_on, self.commands = delay, submit_on, []

    def execute(self, action):
        self.commands.append(action["command"])
        time.sleep(self.delay)
        if action["command"] == self.submit_on:
            raise Submitted({"role": "exit"})
        return {"output": "", "returncode": 3}


class FakeCgroup:
    """Usage shrinks by `step` per limit write until `floor`; writing 0 always fails with EBUSY."""

    def __init__(self, usage=100 << 20, floor=1 << 20, step=40 << 20):
        self.usage_, self.floor, self.step, self.log = usage, floor, step, []

    def usage(self):
        return self.usage_

    def stat(self):
        return {"usage_in_bytes": self.usage_, "kmem_usage_in_bytes": self.floor, "writeback": 0, "swap": 0}

    def set_limit(self, v):
        self.log.append(("limit", v))
        if v == 0:
            self.usage_ = max(self.floor, self.usage_ - self.step)
            raise OSError(errno.EBUSY, "busy")

    def freeze(self):
        self.log.append("freeze")

    def wait_frozen(self):
        return True

    def thaw(self):
        self.log.append("thaw")


ACTIONS = [{"step": 1, "action": 1, "command": "a", "orig_elapsed_s": 0.04, "orig_outcome": "returned",
            "orig_returncode": 3},
           {"step": 2, "action": 1, "command": "b", "orig_elapsed_s": 0.04, "orig_outcome": "Submitted",
            "orig_returncode": 0}]


class ReplayTest(unittest.TestCase):
    def test_times_only_the_call_and_runs_boundary_before_each(self):
        events = []
        before = lambda: events.append("boundary") or time.sleep(0.2) or {"demote_s": 0.2}
        reads = lambda: events.append("counters") or {"usage_bytes": 0, "pgmajfault": 0, "pgpgin": 0,
                                                       "blkio_read_bytes": 0}
        rows = rl.replay(FakeEnv(0.05, submit_on="b"), ACTIONS, before, reads)
        self.assertEqual(events, ["boundary", "counters", "counters"] * 2)
        for r in rows:  # the 0.2 s boundary is not in the call time
            self.assertGreaterEqual(r["call_s"], 0.05)
            self.assertLess(r["call_s"], 0.15)
        self.assertEqual([r["outcome"] for r in rows], ["returned", "Submitted"])
        self.assertEqual([r["returncode"] for r in rows], [3, 0])
        self.assertEqual(rows[0]["demote_s"], 0.2)

    def test_lazy_boundary_restores_limit_before_thaw(self):
        cg = FakeCgroup()
        with tempfile.TemporaryDirectory() as d:
            rec = rl.boundary(cg, Path(d), limit=12345, lazy=True)
        self.assertEqual(cg.log[0], "freeze")
        self.assertEqual(cg.log[-2:], [("limit", 12345), "thaw"])
        self.assertEqual(rec["residual_bytes"], 1 << 20)
        self.assertEqual(rec["residual_user_bytes"], 0)
        self.assertEqual(rec["demoted_bytes"], 99 << 20)

    def test_regular_boundary_does_not_touch_limit(self):
        cg = FakeCgroup()
        rl.boundary(cg, Path("/nonexistent"), limit=12345, lazy=False)
        self.assertEqual(cg.log, ["freeze", "thaw"])

    def test_limit_restored_and_thawed_when_demotion_fails(self):
        cg = FakeCgroup()
        cg.stat = lambda: (_ for _ in ()).throw(OSError("gone"))
        with self.assertRaises(OSError):
            rl.boundary(cg, Path("/nonexistent"), limit=7, lazy=True)
        self.assertEqual(cg.log[-2:], [("limit", 7), "thaw"])

    def test_squeeze_stops_when_reclaim_stalls(self):
        cg = FakeCgroup()
        attempts = rl.squeeze(cg)
        self.assertEqual(cg.usage(), cg.floor)
        self.assertLess(attempts, 20)


class AnalyzeTest(unittest.TestCase):
    def write(self, root, mode, call_s, rc):
        d = root / "run" / "task" / mode
        d.mkdir(parents=True)
        (d / "trial.json").write_text(json.dumps({"mode": mode}))
        pd.DataFrame({"call_index": [1, 2, 3], "step": [1, 2, 3], "action": 1, "call_s": call_s,
                      "outcome": "returned", "returncode": rc, "orig_elapsed_s": [0.1, 0.2, 0.4],
                      "orig_returncode": 0, "call_blkio_read_bytes": 0, "demote_s": 0.01,
                      "residual_user_bytes": 0, "residual_kmem_bytes": 1, "demoted_bytes": 5, "swap_bytes": 0,
                      "call_pgmajfault": 0}).to_csv(d / "calls.csv", index=False)

    def test_slowdown_pairs_calls_and_skips_changed_behavior(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.write(root, "regular", [0.1, 0.2, 0.4], [0, 0, 0])
            self.write(root, "lazy", [0.3, 0.4, 0.8], [0, 0, 1])
            calls = al.pair(al.load(root))
        self.assertEqual(calls.slowdown.round(6).tolist(), [3.0, 2.0, 2.0])
        self.assertEqual(calls.same_behavior.tolist(), [True, True, False])


if __name__ == "__main__":
    unittest.main()
