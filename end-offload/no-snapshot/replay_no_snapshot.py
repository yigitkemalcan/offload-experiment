#!/usr/bin/env python3
"""End-of-task offload, variant: demote writes no cache snapshot.

Same replay and freeze as ../replay_end.py. Demote = fsync the writable layer, then squeeze to ~0 (no copy of
the page cache to a snapshot file). Promote is unchanged: process pages from swap, page cache re-read from the
container's own files. See ../offload_variants.py.

Usage (from this directory):
  sudo ../../.venv/bin/python replay_no_snapshot.py ../../../swebench-runs/<run>     # -> results/<run>/
  sudo ../../.venv/bin/python ../summarize_end.py results
  ../../.venv/bin/python ../plot_cdf.py --summary results/summary.json --out plots
"""

import functools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent))
import replay_end  # noqa: E402
from offload_variants import SUMMARY_EXTRA, measure_offload_variant  # noqa: E402

if __name__ == "__main__":
    replay_end.main(functools.partial(measure_offload_variant, write_snapshot=False, promote_from_snapshot=False),
                    out=HERE / "results", doc=__doc__, summary=replay_end.SUMMARY + SUMMARY_EXTRA)
