"""The promote modes' helper, the page-cache walk's hard links, and the trial plan and summary; no Docker, no root,
no swap needed. DABStep's test_promote_modes.py without its parallel-all parts (threaded snapshot and cache drop),
plus one_name_per_file().

    ../../.venv/bin/python -m unittest test_promote_modes -v      # from this directory

The helper (offload_variants.promote_child) runs in this process on this process's own memory: both modes must
read every recorded byte.
"""
import contextlib
import io
import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import replay_promote_modes as pm
from offload_variants import (DEMOTE_MODES, PROMOTE_MODES, anon_pagemap_snapshot, one_name_per_file,
                              pageout_in_threads, promote_child, read_process_pages, ro)

HELD = bytearray(os.urandom(8 << 20))  # 8 MiB of anonymous memory that is certainly resident


def run_helper(mode: str, pagemap: dict, memcg: Path, cache: dict) -> dict:
    plan = memcg / f"plan-{mode}.pkl"
    plan.write_bytes(pickle.dumps({"memcg": str(memcg), "pagemap": pagemap, "cache": cache, "snapshot": None,
                                   "proc_mode": mode, "proc_threads": 4}))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        promote_child(str(plan))
    return json.loads(out.getvalue())


class Helper(unittest.TestCase):
    def test_every_mode_reads_every_recorded_byte(self):
        pagemap = anon_pagemap_snapshot([os.getpid()])
        recorded = sum(length for _, length in pagemap[os.getpid()]["ranges"])
        self.assertGreater(recorded, len(HELD))
        with tempfile.TemporaryDirectory() as d:
            files = {str(Path(d, f"f{i}")): size for i, size in enumerate((10_000, 6 << 20, 4096))}
            for path, size in files.items():
                Path(path).write_bytes(os.urandom(size))
            # whole files, a range past the end of one (as the last page of a file), and a vanished file
            cache = {path: [(0, size)] for path, size in files.items()}
            cache[str(Path(d, "f0"))] = [(0, 8192), (8192, 4096)]
            cache[str(Path(d, "gone"))] = [(0, 4096)]
            results = {mode: run_helper(mode, pagemap, Path(d), cache) for mode in ("serial", "threads")}
            self.assertEqual(Path(d, "cgroup.procs").read_text(), str(os.getpid()))  # joins the plan's cgroup
        for mode, r in results.items():
            self.assertEqual((r["proc_bytes_read"], r["proc_read_errors"]), (recorded, 0), mode)
            self.assertEqual((r["cache_bytes_read"], r["cache_read_errors"]), (sum(files.values()), 1), mode)
            self.assertLessEqual(r["proc_start_ns"], r["proc_end_ns"])

    def test_threads_split_the_reads_and_count_a_vanished_process(self):
        ranges = anon_pagemap_snapshot([os.getpid()])[os.getpid()]["ranges"][:20]
        total = sum(length for _, length in ranges)
        for threads in (1, 3, 8):
            self.assertEqual(read_process_pages({os.getpid(): {"ranges": ranges}}, threads), (total, 0))
        self.assertEqual(read_process_pages({2**22 + 12345: {"ranges": ranges}}, 3)[0], 0)


    def test_pageout_is_asked_for_every_range_and_failures_are_counted(self):
        pagemap = anon_pagemap_snapshot([os.getpid()])
        recorded = sum(length for _, length in pagemap[os.getpid()]["ranges"])
        advised, errors = pageout_in_threads(pagemap, 4)
        if os.geteuid() == 0:  # pages this process's memory out to swap (it comes back on use)
            self.assertEqual((advised, errors), (recorded, 0))
        else:  # refused: it needs CAP_SYS_NICE
            self.assertEqual(advised, 0)
            self.assertGreaterEqual(errors, 1)
        self.assertEqual(pageout_in_threads({2**22 + 12345: pagemap[os.getpid()]}, 3)[0], 0)


class HardLinks(unittest.TestCase):
    def test_a_linked_file_is_walked_once_and_its_other_names_counted(self):
        with tempfile.TemporaryDirectory() as d:
            pkg, env, alone = Path(d, "pkgs-f"), Path(d, "env-f"), Path(d, "alone")
            pkg.write_bytes(os.urandom(3 * 4096))
            alone.write_bytes(os.urandom(4096))
            os.link(pkg, env)  # conda's pkgs/ <-> the environment
            pkg.read_bytes(), env.read_bytes(), alone.read_bytes()  # all cached
            walked = ro.cache_delta({}, ro.cache_snapshot([Path(d)]))
            self.assertEqual(len(walked), 3)  # by path: the linked file twice
            kept, duplicate_pages = one_name_per_file(walked | {str(Path(d, "gone")): [(0, 4096)]})
        self.assertEqual(duplicate_pages, 3)
        self.assertEqual(len([p for p in kept if p in (str(pkg), str(env))]), 1)
        self.assertIn(str(alone), kept)
        self.assertIn(str(Path(d, "gone")), kept)  # gone since the walk: kept, its read fails as before


class Plan(unittest.TestCase):
    def test_modes_set_the_measurement_of_each_variant(self):
        self.assertEqual(list(pm.MODES), ["serial", "parallel-swap"])
        for variant in pm.VARIANTS:
            for mode in pm.MODES:
                s = pm.settings(variant, mode, 8)
                self.assertIn(s["demote_mode"], DEMOTE_MODES)
                self.assertIn(s["promote_mode"], PROMOTE_MODES)
                self.assertEqual((s["write_snapshot"], s["promote_from_snapshot"]), (variant == "snapshot",) * 2)
                self.assertNotIn("snapshot_threads", s)  # the page cache moves with one thread in every mode
            self.assertEqual({k: pm.settings(variant, "serial", 8)[k] for k in ("demote_mode", "promote_mode")},
                             {"demote_mode": "squeeze", "promote_mode": "serial"})
            self.assertEqual({k: pm.settings(variant, "parallel-swap", 8)[k] for k in ("demote_mode", "promote_mode",
                                                                                     "demote_threads", "promote_threads")},
                             {"demote_mode": "pageout-threads", "promote_mode": "threads", "demote_threads": 8,
                              "promote_threads": 8})

    def test_modes_are_interleaved_within_each_repeat(self):
        self.assertEqual(pm.trials(["1300"], ["serial", "threads"], 2),
                         [("1300", "serial", 1), ("1300", "threads", 1), ("1300", "serial", 2), ("1300", "threads", 2)])

    def test_summary_takes_medians_of_good_trials_and_adds_the_reference(self):
        mib = 2**20
        rows = pd.DataFrame([
            {"task_id": "1300", "mode": "serial", "repeat": k, "faithful": True, "error": None, "demote_s": 1.5,
             "swapped_bytes": 200 * mib, "promote_process_s": s, "promote_s": s + 1, "promote_recovered_fraction": 0.7}
            for k, s in ((1, 2.0), (2, 2.2), (3, 1.8))] + [
            {"task_id": "1300", "mode": "threads", "repeat": 1, "faithful": True, "error": None, "demote_s": 1.5,
             "swapped_bytes": 200 * mib, "promote_process_s": 0.5, "promote_s": 1.5, "promote_recovered_fraction": 0.7},
            {"task_id": "1300", "mode": "threads", "repeat": 2, "faithful": False, "error": None, "demote_s": 9.0,
             "swapped_bytes": 200 * mib, "promote_process_s": 9.0, "promote_s": 9.0, "promote_recovered_fraction": 0.7}])
        ref = pd.DataFrame({"task_id": [1300, 38], "swapped_bytes": [210 * mib, 1], "promote_process_s": [2.1, 1],
                            "promote_s": [3.0, 1], "promote_recovered_fraction": [0.72, 1],
                            "demote_s": [1.4, 1]})
        s = pm.summarize(rows, ref).set_index("mode")
        self.assertEqual((s.loc["serial", "trials"], s.loc["serial", "process_s"], s.loc["serial", "swap_in_mib_s"]),
                         (3, 2.0, 100.0))
        self.assertEqual((s.loc["threads", "trials"], s.loc["threads", "swap_in_mib_s"]), (1, 400.0))
        self.assertEqual(s.loc["serial (100-task run)", "process_s"], 2.1)
        self.assertEqual((s.loc["threads", "demote_s"], s.loc["serial (100-task run)", "demote_s"]), (1.5, 1.4))
        self.assertEqual(len(s), 3)


if __name__ == "__main__":
    unittest.main()
