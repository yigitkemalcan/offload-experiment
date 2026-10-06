#!/usr/bin/env python3
"""End-of-task offload, variant: promote reads the page cache back from the demote-side snapshot file.

Same replay and freeze as ../replay_end.py. Demote is unchanged (fsync the writable layer, copy the
container's page cache to a snapshot file, squeeze to ~0); the snapshot is kept, its own page cache is
evicted, and promote reads it sequentially instead of re-reading the container's files. Process pages and
/dev/shm come back from swap as before. See ../offload_variants.py.

Usage (from this directory):
  sudo ../../.venv/bin/python replay_snapshot_promote.py ../../../swebench-runs/<run>     # -> results/<run>/
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
    replay_end.main(functools.partial(measure_offload_variant, write_snapshot=True, promote_from_snapshot=True),
                    out=HERE / "results", doc=__doc__, summary=replay_end.SUMMARY + SUMMARY_EXTRA)
