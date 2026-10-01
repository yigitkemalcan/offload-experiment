# Page reuse — schema 5

For the single-workload, ample-memory CPU experiment, the loop is:

1. Replay one recorded tool call through `docker exec`.
2. Freeze the container.
3. Check its pages (cached files in the layers and /dev/shm, pages of its processes).
4. Classify each page used in the call as **reused** or **not_reused**; record the distance for reused ones.
5. Resume the container and run the next tool call.

Every tool call gets a boundary, even if several belong to one agent step.
The original step and action IDs are preserved.

## Classification

A page counts as used in a call if it is newly resident (loaded, readahead
included) or it was resident before and its idle bit was cleared. Pages are
identified logically: file + page offset, or process + virtual page.

- **reused:** used in this call and in some earlier call.
- **not_reused:** first use.

`accessed_pages = reused_pages + not_reused_pages`. Boundary 0 is inventory
only; container startup is not counted as use. Memory is ample, so there is no
special handling of evictions or frame changes.

Kernel 5.15 `read()` never marks the first page of each 15-page batch after the
first as accessed, so it stays idle. An idle file page whose previous and next
pages were both used is therefore counted as used too.

## Distances

For each reused page, relative to its most recent previous use:

- `reuse_distance_calls`: current call index minus previous call index.
- `reuse_distance_steps`: current agent step minus previous agent step (0 for two calls in one step).

Both are `-1` for not-reused pages. Example: loaded in step 3, read in step 5,
read again in step 7 gives `not_reused, reused, reused` with step distances `-1, 2, 2`.

## Files

Results default to `results-v5`.

- `calls.csv`: one row per call plus startup; resident/accessed/reused/not-reused
  counts split into file/shm/mapped/anon, timings and memory peak.
- `accessed.npz`: every used page per call: call index, step, action, key, `reused`, both distances.
- `reuse_distances.csv`: histogram by memory kind, call distance and step distance.
- `objects.csv`: per file/process page uses and reuse fraction.
- `processes.csv`: processes alive at each boundary.
- `actions.csv`, `trial.json`, `measurement.json`: commands, summary and definitions.

The run directory combines per-task calls, summaries and distance histograms.

## Analysis and plots

See `INTERPRETATION.md` for what the memory kinds and sources mean and how to read the numbers.

Both read every finished task (with `trial.json`) under the results directory, across runs:

```sh
sudo ../.venv/bin/python analyze_reuse.py results-v5-full   # tables -> results-v5-full/analysis/
sudo ../.venv/bin/python plot_reuse.py results-v5-full      # plots  -> results-v5-full/plots/
```

## Tests

Unprivileged accounting, identity and replay-loop tests:

```sh
PYTHONDONTWRITEBYTECODE=1 ../.venv/bin/python -m unittest test_replay_reuse -v
```

Opt-in kernel/cgroup integration test (run by the experiment operator):

```sh
sudo env REUSE_KERNEL_TESTS=1 ../.venv/bin/python -m unittest test_replay_reuse.ReuseTest -v
```
