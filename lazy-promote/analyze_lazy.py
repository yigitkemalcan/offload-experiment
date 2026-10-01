#!/usr/bin/env python3
"""Slowdown of lazy promotion per tool call: call time (lazy) / call time (regular).

Pairs the two replays of each call (same task, same call index). The primary denominator is the regular
replay, measured by the same harness right before the lazy one. The original agent run's time is kept as a
second reference (slowdown_vs_orig): it was measured the same way, but with the agent and model server
running, on a host cache the replay does not reproduce.

A call is compared only if it behaved the same in both replays (same return code and outcome). Writes
<results>/analysis/{calls,tasks}.csv and summary.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["run", "task_id", "call_index"]


def load(results: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p).assign(run=p.parents[2].name, task_id=p.parents[1].name, mode=p.parent.name)
              for p in sorted(results.glob("*/*/*/calls.csv"))
              if (p.parent / "trial.json").exists() and "error" not in json.loads((p.parent / "trial.json").read_text())]
    if not frames:
        raise SystemExit(f"no finished trials under {results}")
    return pd.concat(frames, ignore_index=True)


def pair(calls: pd.DataFrame) -> pd.DataFrame:
    reg = calls[calls["mode"] == "regular"].set_index(KEY)
    lazy = calls[calls["mode"] == "lazy"].set_index(KEY)
    both = reg.index.intersection(lazy.index)
    reg, lazy = reg.loc[both], lazy.loc[both]
    out = pd.DataFrame({"step": lazy.step, "action": lazy.action,
                        "regular_s": reg.call_s, "lazy_s": lazy.call_s, "orig_s": lazy.orig_elapsed_s,
                        "regular_returncode": reg.returncode, "lazy_returncode": lazy.returncode,
                        "orig_returncode": lazy.orig_returncode, "outcome": lazy.outcome})
    for c in ("demote_s", "demoted_bytes", "residual_user_bytes", "residual_kmem_bytes", "swap_bytes",
              "call_pgmajfault", "call_blkio_read_bytes"):
        if c in lazy:
            out[c] = lazy[c]
    out["regular_blkio_read_bytes"] = reg.call_blkio_read_bytes
    out["same_behavior"] = (reg.returncode.astype(str) == lazy.returncode.astype(str)) & (reg.outcome == lazy.outcome)
    out["slowdown"] = out.lazy_s / out.regular_s
    out["slowdown_vs_orig"] = out.lazy_s / out.orig_s
    out["added_s"] = out.lazy_s - out.regular_s
    return out.reset_index()


def stats(x: pd.Series) -> dict:
    x = x.dropna()
    if not len(x):
        return {}
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(x.median()),
            "p10": float(x.quantile(0.1)), "p90": float(x.quantile(0.9)), "p99": float(x.quantile(0.99)),
            "geomean": float(np.exp(np.log(x[x > 0]).mean())), "max": float(x.max())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path, nargs="?", default=Path(__file__).resolve().parent / "results")
    args = ap.parse_args()
    out_dir = args.results / "analysis"
    out_dir.mkdir(exist_ok=True)

    calls = pair(load(args.results))
    calls.to_csv(out_dir / "calls.csv", index=False)
    ok = calls[calls.same_behavior]
    tasks = ok.groupby(["run", "task_id"]).agg(calls=("call_index", "size"), regular_s=("regular_s", "sum"),
                                               lazy_s=("lazy_s", "sum"), orig_s=("orig_s", "sum")).reset_index()
    tasks["slowdown"] = tasks.lazy_s / tasks.regular_s
    tasks["slowdown_vs_orig"] = tasks.lazy_s / tasks.orig_s
    tasks.to_csv(out_dir / "tasks.csv", index=False)

    summary = {
        "calls_paired": int(len(calls)), "calls_compared": int(len(ok)),
        "calls_excluded_behavior_differs": int((~calls.same_behavior).sum()),
        "tasks": int(len(tasks)),
        "per_call_slowdown": stats(ok.slowdown),
        "per_call_added_s": stats(ok.added_s),
        "per_task_slowdown (sum lazy / sum regular)": stats(tasks.slowdown),
        "overall_slowdown (all calls, time-weighted)": float(ok.lazy_s.sum() / ok.regular_s.sum()),
        "per_call_slowdown_vs_orig": stats(ok.slowdown_vs_orig),
        "regular_vs_orig (replay noise check)": stats(ok.regular_s / ok.orig_s),
    }
    if "residual_user_bytes" in ok:
        summary["demotion"] = {"residual_user_bytes": stats(ok.residual_user_bytes),
                               "residual_kmem_bytes": stats(ok.residual_kmem_bytes),
                               "demote_s": stats(ok.demote_s),
                               "lazy_read_bytes_per_call": stats(ok.call_blkio_read_bytes),
                               "regular_read_bytes_per_call": stats(ok.regular_blkio_read_bytes)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
