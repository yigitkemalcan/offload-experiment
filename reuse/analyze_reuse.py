#!/usr/bin/env python3
"""Summarize replay_reuse.py results by memory kind (file cache, shm, mapped, anon) and page source.

    ../.venv/bin/python analyze_reuse.py results-v5-full            # every task under the directory
    ../.venv/bin/python analyze_reuse.py results-v5-full --out analysis

Writes CSVs to --out (default <results>/analysis) and prints the main tables:
  kinds.csv      per kind: page-uses, reused / not_reused, reuse fraction, share of all page-uses
  sources.csv    per source (task repo, environment, /tmp, process memory, ...) and kind
  distances.csv  per kind: reuse distance statistics in calls and steps
  histogram.csv  per kind: reused page-uses at each call / step distance
  memory_types.csv  page cache (file + shm + mapped) vs anonymous: page-uses, reuse fraction, share
  tasks.csv      per task and kind: page-weighted and per-call-mean reuse fraction
  summary.json   the main numbers of all of the above in one file
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KINDS = ("file", "shm", "mapped", "anon")
TYPES = {"cache": ("file", "shm", "mapped"), "anon": ("anon",)}  # page cache vs anonymous memory
ENVIRONMENT = ("/opt/", "/usr/", "/lib", "/bin/", "/sbin/", "/etc/", "/root/", "/var/")


def keepalive(task_dir: Path) -> int | None:
    """pid of the container's keep-alive `sleep` if this task measured it (results before it was excluded).
    It is the only process at boundary 0. Its one anonymous stack page is touched by our freeze/thaw."""
    p = pd.read_csv(task_dir / "processes.csv") if (task_dir / "processes.csv").stat().st_size > 1 else pd.DataFrame()
    first = p[p.call_index == 0] if len(p) else p
    return int(first.pid.iloc[0]) if len(first) == 1 else None


def drop_keepalive(task_dir: Path, calls: pd.DataFrame, objs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove the keep-alive's anonymous page-uses from calls.csv counts and objects.csv (process id 0 in keys)."""
    pid = keepalive(task_dir)
    if pid is None:
        return calls, objs
    z = np.load(task_dir / "accessed.npz")
    key, reused = z["key"].astype(np.uint64), z["reused"]
    mine = keepalive_keys(task_dir, key)
    per_call = pd.DataFrame({"call_index": z["call_index"][mine], "reused": reused[mine]}).groupby(
        "call_index").reused.agg(["size", "sum"]).reindex(calls.call_index, fill_value=0)
    calls = calls.copy()
    n, r = per_call["size"].to_numpy(), per_call["sum"].to_numpy()
    for k in ("anon_", ""):
        calls[f"accessed_{k}pages"] -= n
        calls[f"reused_{k}pages"] -= r
        calls[f"not_reused_{k}pages"] -= n - r
    return calls, keepalive_objects(task_dir, objs)


def keepalive_objects(task_dir: Path, objs: pd.DataFrame) -> pd.DataFrame:
    """objects.csv without the keep-alive process row."""
    pid = keepalive(task_dir)
    if pid is None:
        return objs
    return objs[~((objs.kind == "anon") & objs.object.astype(str).str.startswith(f"pid {pid} ("))].copy()


def source(kind: str, path: str) -> str:
    if kind == "anon":
        return "process memory"
    if kind == "shm":
        return "/dev/shm"
    if path.startswith("/testbed"):
        return "task repo (/testbed)"
    if path.startswith("/tmp"):
        return "/tmp"
    if path.startswith(ENVIRONMENT):
        return "environment (runtime, libraries, config)"
    return "other"


def fraction(num, den):
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    return float(v[np.searchsorted(np.cumsum(w), q * w.sum())])


def load(results: Path):
    calls, hist, objs = [], [], []
    for trial in sorted(results.glob("**/trial.json")):
        d = trial.parent
        if not (d / "calls.csv").exists():  # failed task
            continue
        ids = {"run": d.parent.name, "task_id": d.name}
        c = pd.read_csv(d / "calls.csv")
        o = pd.read_csv(d / "objects.csv") if (d / "objects.csv").exists() else pd.DataFrame(columns=["kind", "object"])
        c, o = drop_keepalive(d, c, o)
        calls.append(c.assign(**ids))
        objs.append(o.assign(**ids))
        if (d / "accessed.npz").exists():
            hist.append(distance_histogram(d).assign(**ids))
    if not calls:
        raise SystemExit(f"no task results under {results}")
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()
    calls = cat(calls)
    for m in ("resident", "accessed", "reused", "not_reused"):
        calls[f"{m}_cache_pages"] = sum(calls[f"{m}_{k}_pages"] for k in TYPES["cache"])
    return calls[calls.call_index > 0], cat(hist), cat(objs)  # call 0 is the startup inventory


def keepalive_keys(task_dir: Path, key: np.ndarray) -> np.ndarray:
    """Mask of the keep-alive's anonymous page keys (process id 0) in results that measured it."""
    if keepalive(task_dir) is None:
        return np.zeros(len(key), bool)
    return ((key >> np.uint64(63)) == 1) & (((key >> np.uint64(36)) & np.uint64((1 << 27) - 1)) == 0)


def distance_histogram(task_dir: Path) -> pd.DataFrame:
    """reuse_distances.csv with the anon rows rebuilt from accessed.npz without the keep-alive's pages."""
    path = task_dir / "reuse_distances.csv"
    h = pd.read_csv(path) if path.exists() and path.stat().st_size > 1 else pd.DataFrame(
        columns=["kind", "reuse_distance_calls", "reuse_distance_steps", "page_calls"])
    z = np.load(task_dir / "accessed.npz")
    key = z["key"].astype(np.uint64)
    anon = z["reused"] & ((key >> np.uint64(63)) == 1) & ~keepalive_keys(task_dir, key)
    a = pd.DataFrame({"reuse_distance_calls": z["reuse_distance_calls"][anon],
                      "reuse_distance_steps": z["reuse_distance_steps"][anon]})
    a = a.groupby(["reuse_distance_calls", "reuse_distance_steps"]).size().reset_index(name="page_calls")
    return pd.concat([h[h.kind != "anon"], a.assign(kind="anon")], ignore_index=True)


def by_kind(calls: pd.DataFrame, kinds=KINDS) -> pd.DataFrame:
    rows = []
    for k in kinds:
        acc, reu = calls[f"accessed_{k}_pages"].sum(), calls[f"reused_{k}_pages"].sum()
        rows.append({"kind": k, "page_uses": acc, "reused": reu, "not_reused": acc - reu})
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"kind": "all", "page_uses": df.page_uses.sum(), "reused": df.reused.sum(),
                       "not_reused": df.not_reused.sum()}
    df["reuse_fraction"] = fraction(df.reused, df.page_uses)
    df["share_of_page_uses"] = df.page_uses / df.page_uses.iloc[-1]
    return df


def by_type(calls: pd.DataFrame) -> pd.DataFrame:
    df = by_kind(calls, ("cache", "anon"))
    df.insert(1, "includes", [", ".join(TYPES.get(t, ("all",))) for t in df.kind])
    return df.rename(columns={"kind": "memory_type"})


def by_source(objs: pd.DataFrame) -> pd.DataFrame:
    objs = objs.assign(source=[source(k, str(p)) for k, p in zip(objs.kind, objs.object)])
    df = objs.groupby(["source", "kind"], as_index=False).agg(
        objects=("object", "size"), page_uses=("page_calls", "sum"), reused=("reused_page_calls", "sum"))
    df["not_reused"] = df.page_uses - df.reused
    df["reuse_fraction"] = fraction(df.reused, df.page_uses)
    df["share_of_page_uses"] = df.page_uses / df.page_uses.sum()
    return df.sort_values("page_uses", ascending=False)


def distances(hist: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if hist.empty:
        return pd.DataFrame(), pd.DataFrame()
    h = hist.groupby(["kind", "reuse_distance_calls", "reuse_distance_steps"], as_index=False).page_calls.sum()
    rows = []
    for k, g in list(h.groupby("kind")) + [("all", h)]:
        w = g.page_calls.to_numpy()
        row = {"kind": k, "reused_page_uses": int(w.sum())}
        for unit in ("calls", "steps"):
            v = g[f"reuse_distance_{unit}"].to_numpy()
            row |= {f"{unit}_mean": float(np.average(v, weights=w)),
                    f"{unit}_median": weighted_quantile(v, w, 0.5),
                    f"{unit}_p90": weighted_quantile(v, w, 0.9),
                    f"{unit}_max": int(v.max()),
                    f"{unit}_share_at_1": float(w[v == 1].sum() / w.sum())}
        rows.append(row)
    return pd.DataFrame(rows), h


def by_task(calls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run, task), g in calls.groupby(["run", "task_id"]):
        row = {"run": run, "task_id": task, "calls": len(g)}
        for k in ("", "cache_", *(f"{k}_" for k in KINDS)):
            acc, reu = g[f"accessed_{k}pages"], g[f"reused_{k}pages"]
            name = k.rstrip("_") or "all"
            row[f"{name}_page_uses"] = int(acc.sum())
            row[f"{name}_reuse_fraction"] = reu.sum() / acc.sum() if acc.sum() else np.nan
            row[f"{name}_per_call_mean"] = np.nanmean(fraction(reu, acc)) if (acc > 0).any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def records(df: pd.DataFrame) -> list[dict]:
    """Rows as JSON-ready dicts (NaN -> null, numpy -> Python numbers)."""
    return json.loads(df.to_json(orient="records"))


def summary(calls, kinds, types, sources, stats, tasks) -> dict:
    per_task = {}
    for col in ("all_reuse_fraction", "all_per_call_mean"):
        v = tasks[col].dropna()
        per_task[col] = {"mean": v.mean(), "median": v.median(), "std": v.std(), "min": v.min(), "max": v.max(),
                         "min_task": tasks.task_id[v.idxmin()], "max_task": tasks.task_id[v.idxmax()]}
    all_row = kinds.set_index("kind").loc["all"]
    return {
        "tasks": int(tasks.task_id.nunique()),
        "calls": int(len(calls)),
        "runs": sorted(calls.run.unique().tolist()),
        "overall": {"page_uses": int(all_row.page_uses), "reused": int(all_row.reused),
                    "not_reused": int(all_row.not_reused), "reuse_fraction": float(all_row.reuse_fraction)},
        "by_memory_type": records(types[types.memory_type != "all"]),
        "by_kind": records(kinds[kinds.kind != "all"]),
        "by_source": records(sources),
        "reuse_distance": records(stats),
        "per_task_distribution": json.loads(json.dumps(per_task, default=float)),
        "per_task": records(tasks[["run", "task_id", "calls", "all_page_uses", "all_reuse_fraction",
                                   "all_per_call_mean", "cache_page_uses", "cache_reuse_fraction",
                                   "anon_page_uses", "anon_reuse_fraction"]]),
        "definitions": {
            "page_use": "one page used in one tool call (newly resident or idle bit cleared)",
            "reuse_fraction": "reused page-uses / all page-uses (page-weighted)",
            "per_call_mean": "mean over calls of each call's reuse fraction (every call weighted equally)",
            "reuse_distance_calls": "tool calls since the previous use of the same page",
            "reuse_distance_steps": "agent steps since the previous use of the same page",
            "memory_types": "cache = page cache of files (file, mapped) and shared memory (shm); "
                            "anon = private anonymous memory of processes alive at the boundary",
            "sources": "task repo = /testbed; environment = /opt, /usr, /lib, /bin, /sbin, /etc, /root, /var; "
                       "process memory = anonymous pages",
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    out = args.out or args.results / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    calls, hist, objs = load(args.results)
    kinds, types, sources, tasks = by_kind(calls), by_type(calls), by_source(objs), by_task(calls)
    stats, histogram = distances(hist)
    for name, df in (("kinds", kinds), ("memory_types", types), ("sources", sources), ("distances", stats),
                     ("histogram", histogram), ("tasks", tasks)):
        df.to_csv(out / f"{name}.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary(calls, kinds, types, sources, stats, tasks), indent=1))

    pd.set_option("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.3f}".format)
    print(f"{tasks.task_id.nunique()} tasks, {len(calls)} calls\n")
    print("== page cache vs anonymous (page-weighted)\n", types.to_string(index=False), "\n")
    print("== by memory kind (page-weighted)\n", kinds.to_string(index=False), "\n")
    print("== by source\n", sources.to_string(index=False), "\n")
    if not stats.empty:
        print("== reuse distance\n", stats.to_string(index=False), "\n")
    cols = ["all_reuse_fraction", "all_per_call_mean", "cache_reuse_fraction", "anon_reuse_fraction"]
    print("== per task (distribution across tasks)\n", tasks[cols].describe().loc[["mean", "50%", "min", "max"]]
          .to_string(), "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
