# Lazy promotion slowdown

This experiment measures how much longer recorded tool calls take when container memory is
reclaimed before every call and pages are brought back on demand. It replays existing commands;
it does not run the model again or generate new actions.

```text
per-call slowdown = lazy call time / regular call time
added time       = lazy call time - regular call time
```

A slowdown of 1.4 means 40% longer execution. The primary comparison is against a regular replay
using the same harness, not against the original agent recording. Demotion time is measured
separately and excluded from call slowdown.

## Execution order

Tasks are processed sequentially. Each task runs completely in regular mode and then completely
in lazy mode, in separate fresh containers:

```text
Task 1
  sync + clear host page cache
  fresh container -> all regular calls -> remove container
  sync + clear host page cache
  fresh container -> all lazy calls -> remove container
Task 2
  sync + clear host page cache
  fresh container -> all regular calls -> remove container
  ...
```

The host cache is cleared before each `(task, mode)`, not after every call. There is no extra clear
at the end of the experiment. This affects the whole host, so run while other measurements are idle.

Every executed tool call gets a boundary, including multiple actions within one agent step:

- **Regular:** freeze -> thaw -> execute the call. Cache can accumulate between calls.
- **Lazy:** freeze -> flush writable files -> reclaim memory -> restore the original memory limit
  -> thaw -> execute the call. This also happens before the first call.

Both modes start with a cold host cache. Lazy mode additionally reclaims container memory between
calls. There is no explicit promotion pass: execution loads pages as needed, including any kernel
readahead. Foreground command processes normally exit between calls, so the memory affected is
largely file cache for executables, libraries, and repository files. This is not a guarantee that
commands cannot leave background processes alive; the lazy harness does not inventory them.

## Code and measurements

### `replay_lazy.py`: replay and demotion

The script imports trajectory loading and cgroup helpers from `../peak/replay_offload.py`.
`load_tasks` reads task metrics, container samples, trajectory environment configuration, and
executed actions. It currently requires the original resource samples even though the computed
peak-memory threshold is unused in this experiment.

`DockerEnvironment(**config)` uses the recorded image, command environment, working directory,
interpreter and timeout. `execute` wraps `env.execute` in `DefaultAgent._measure_call`, the same
`monotonic_ns` timing mechanism as the original agent. Timing includes the Docker execution path,
not just CPU time inside the command. A final `Submitted` action is handled explicitly.

`run_task` enables cgroup swappiness 100 for lazy mode. `boundary` freezes the container; in lazy
mode it calls `demote`, restores the original memory limit, and thaws. The restoration and thaw
are attempted even if demotion raises an exception.

`demote` performs:

1. Record cgroup memory usage.
2. Walk the writable layer and open, `fsync`, and close its files.
3. Repeatedly try setting `memory.limit_in_bytes` to zero. This creates reclaim pressure: clean
   cache can be discarded and anonymous/shared memory can be swapped.
4. Stop on a successful limit write, when a retry reduces usage by no more than one page, or at
   20 attempts. An `EBUSY` result triggers this retry logic; other errors propagate.
5. Wait for reported writeback to finish, with a 60-second deadline, then record residual usage.

The current code does **not** retry reclaim after the writeback wait. A stalled attempt is not
proof that all remaining pages are permanently unreclaimable. The flush loop also ignores file
operation errors, and the writeback deadline does not itself raise an error.

`demote_flush_s` includes directory traversal and file operations, not only time writing dirty
bytes. `flushed_files` counts successfully processed files, including clean files.
`demote_reclaim_s` combines limit-write attempts and the subsequent writeback wait.
`demote_s` is the sum of these two phases. Unlike the peak experiment, this code does **not**
write a snapshot of clean cached file contents.

Freeze/thaw, demotion, and counter reads are outside `call_s`. The script records major-fault
and block-read counter differences around each call, as well as `pgpgin` (a cgroup charging-event
counter, not a direct disk-byte count). It resets the memory high-water counter before each call.

### `analyze_lazy.py`: comparison

Pairs regular and lazy rows by `(run, task_id, call_index)`. Summary statistics use pairs with
matching return codes and outcomes. The paired CSV retains rows with different behavior and marks
them using `same_behavior`; those rows are excluded from aggregate comparisons.

- Per-call statistics give every call equal weight.
- Task slowdown is `sum(lazy call times) / sum(regular call times)` within a task.
- Overall slowdown uses those sums across all comparable calls; it is not the mean call ratio.
- `slowdown_vs_orig` and `regular_vs_orig` are secondary references to the recorded agent times.

The original times used the same timing mechanism but different runtime/cache conditions.
Neither task slowdown nor overall slowdown includes model inference, container setup or demotion.

### `plot_lazy.py`: validation, plots and command lookup

Rebuilds paired data from completed trials, checks expected call indices and durations against the
original trajectories, and joins full command strings. Writes PNG plots, ranked CSVs, validation
and a report. It does not generate PDFs. The demoted-memory plot uses every completed lazy boundary,
independent of behavior filtering. Default output is `<results>/plots`.

## Running

From `offload-experiment/lazy-promote`, enable swap if necessary and run a smoke test:

```sh
sudo ../setup_swap.sh on
sudo ../../mini-swe-agent/.venv/bin/python replay_lazy.py \
  ../../swebench-runs/qwen-run-20260923T053154Z-PF12hU \
  --tasks astropy__astropy-14369
```

The host must support the harness's cgroup v1 memory/freezer paths and Docker writable-layer
inspection. Root is required for cgroup writes, cache clearing and access to overlay files.

For the full 100 tasks, run the following inside an existing tmux session. One sudo invocation
runs both batches and analysis, so sudo credential expiration cannot cause a second prompt:

```sh
sudo bash -c '
set -e
exec > >(tee -a full-run.log) 2>&1
PY=../../mini-swe-agent/.venv/bin/python
"$PY" -u replay_lazy.py ../../swebench-runs/qwen-run-20260923T053154Z-PF12hU
"$PY" -u replay_lazy.py ../../swebench-runs/qwen-run-20260923T073021Z-LtPzSb
"$PY" -u analyze_lazy.py results
'
```

After entering the password and seeing progress, detach with Ctrl-b, then d. Closing the client
terminal after detaching leaves the tmux job running; do not kill its session.

Any `(task, mode)` with `trial.json` is skipped, **including recorded failures**. Interrupted pairs
without that file restart from a fresh container. Use `--out results-retry` for a separate retry
set. Individual task exceptions are recorded and the batch continues, so process completion or
`set -e` alone does not establish that every task succeeded.

Generate analysis/plots separately (use sudo only if needed for directory permissions):

```sh
../../mini-swe-agent/.venv/bin/python analyze_lazy.py results
PYTHONDONTWRITEBYTECODE=1 ../../mini-swe-agent/.venv/bin/python plot_lazy.py results
```

`plot_lazy.py --out <directory>` overrides the plot destination. Plotting requires matplotlib.

## Output files

| Location | Contents |
|---|---|
| `results/<run>/<task>/<mode>/calls.csv` | Call timing, outcomes, original references, memory/counter deltas; lazy rows also include demotion phases and residuals |
| `results/<run>/<task>/<mode>/trial.json` | Task/mode totals, mismatch count or error |
| `results/<run>/{calls,trials}.csv` | Combined per-run records |
| `results/analysis/calls.csv` | Paired calls, slowdown, added seconds, behavior-match flag |
| `results/analysis/tasks.csv` | Task-level sums and ratios |
| `results/analysis/summary.json` | Aggregate slowdown, baseline checks and demotion statistics |
| `results/plots/ranked_calls.csv` | Comparable calls ranked by slowdown, with full commands |
| `results/plots/ranked_added_time.csv` | Calls ranked by absolute added time |
| `results/plots/{validation.json,original_mismatches.csv,report.md}` | Validation and worst-call details |
| `results/plots/{demoted_memory.csv,demoted_memory_summary.json}` | Per-boundary memory sizes, residual fractions and quantiles |

## Results: completed 100-task run (2026-10-01)

The two input runs contain 42 and 58 tasks. All 200 task/mode trials completed, with all 4,487
expected call pairs present. Return codes and outcomes match between regular and lazy for every
pair. Five calls differ from their original recorded return codes, identically in both replays:
`django__django-11740` call 49 and `django__django-12754` calls 34, 35, 36 and 42. Original return
codes are unavailable for 83 paired calls; missing values are not counted as mismatches.

Task lengths range from 4 to 249 calls (median 32, mean 44.87). The four-call smoke-test task
`astropy__astropy-14369` is genuinely short: its original eight inference steps executed commands
only at steps 1, 2, 3 and 5, then ended with `RepeatedFormatError`. Replay completion does not mean
that the original agent solved the task. These 100 tasks include unsuccessful agent trajectories.

| Metric | Value |
|---|---:|
| Median per-call slowdown | 1.412x |
| p90 / p99 per-call slowdown | 1.466x / 2.645x |
| Maximum per-call slowdown | 3.485x |
| Overall time-weighted slowdown | 1.399x |
| Median / maximum task slowdown | 1.371x / 1.866x |
| Median / p99 added call time | 77.6 ms / 447.3 ms |
| Total regular / lazy call time | 1,149.30 s / 1,608.38 s |
| Total added call time | 459.08 s |
| Total demotion time (separate) | 325.74 s |

The typical call took about 41% longer, usually adding roughly 78 ms. The worst call was
`grep -r "simplify_regexp" .` in `django__django-11728`, call 5: 0.452 -> 1.575 seconds
(3.485x, +1.123 seconds). Recursive repository searches dominate the highest ratios, consistent
with reloading files that regular execution can read from cache. This association alone does not
isolate all causes of the latency.

## How to read the plots

A CDF's y coordinate is the percentage of observations at or below its x coordinate. For example,
90% at 1.466x means 90% of calls experienced at most that slowdown.

### `slowdown_cdf.png`

Left: the full per-call slowdown CDF. Its steep rise near 1.4x shows a common proportional penalty;
the right tail contains rarer, larger penalties. Right: the same CDF zoomed to the slowest 10%
(y = 90–100%). Every call has equal weight; this is not a time-weighted distribution.

### `worst_calls.png`

The 15 largest slowdown ratios, with regular and lazy duration bars, command previews, ratio and
added milliseconds. Full commands are in the ranked CSV/report. The ranking is by ratio, which
is different from ranking by absolute latency; use `ranked_added_time.csv` for the latter.

### `latency_diagnostics.png`

Only two panels are generated:

- **Call durations:** each point is a regular/lazy pair, on logarithmic axes. Above the diagonal
  means lazy was slower. Long calls tend to lie closer to the diagonal, so the extra cost is a
  smaller fraction of their execution time.
- **Disk reads and added latency:** lazy disk-read volume versus lazy minus regular seconds,
  shown in milliseconds. Many calls cluster near 45–50 MiB and 70–100 ms added. Larger reads
  sometimes accompany much greater penalties, but there is no single linear relationship.
  The x axis is total lazy reads, not additional reads relative to regular. Read pattern, regular
  I/O, execution work and noise can also matter.

The former third panel showing added-time and demotion-time distributions was removed. The
recorded demotion times remain available in the CSVs and analysis summary.

### `task_and_demotion.png`

Left: one ratio of summed call durations per task. Individual outliers are combined with other
calls in that task. This is tool-execution slowdown, not full agent completion-time slowdown.

Right: estimated user memory remaining after demotion, in MiB on a logarithmic axis. Median
residual is 0.51 MiB; p99 is 1.13 MiB; maximum is 11.71 MiB. Kernel memory is excluded from this
curve. It shows absolute residual size, not the fraction reclaimed or proof of unevictable pages.

### `demoted_memory_cdf.png`

Left: CDFs of memory reclaimed and total memory before demotion. Demoted memory is the reduction
in cgroup resident usage, **not bytes written to disk**: clean file cache can simply be discarded.
Median demoted size is 24.13 MiB, p90 53.34 MiB, p99 94.50 MiB, maximum 201.21 MiB.

Right: per-boundary residual fractions. Both curves divide by total pre-demotion memory,
including kernel memory in the denominator:

```text
estimated user residual = total residual - kernel residual
user fraction          = estimated user residual / total usage before demotion
total fraction         = total residual / total usage before demotion
```

Estimated user fraction has median 1.72%, p99 4.24%, maximum 11.25%. Total residual including
kernel has median 9.01%. The user residual is generally small by size, but a small set of hot
pages could still affect the next call materially. Neither this plot nor the byte fraction measures
how much residual pages bias slowdown. The curves are separate distributions, not paired vertical
readouts of the same event at each percentile.

## Why some demotions take longer than the peak experiment

Eleven lazy demotions exceeded 500 ms; only one exceeded one second. Rows refer to demotion
**before** the indicated call, so the preceding command produced the state being demoted.

| Task / next call | Memory before | Reclaimed | Flush phase | Reclaim + writeback wait | Total |
|---|---:|---:|---:|---:|---:|
| `django-11740` / 116 | 183.75 MiB | 166.57 MiB | 1,211 ms | 329 ms | 1,540 ms |
| `django-11740` / 141 | 98.26 MiB | 88.04 MiB | 163 ms | 823 ms | 986 ms |
| `django-11276` / 79 | 78.69 MiB | 68.69 MiB | 2 ms | 863 ms | 865 ms |

Before boundary 116, call 115 executed `git reset --hard` followed by an edit. Successfully
processed writable files jumped from 242 to 6,375 and flush time rose from 5.7 ms to 1,211 ms.
Thus the largest outlier is primarily traversal/open/fsync/close work over the enlarged writable
file set, not unusually slow reclaim. Per-file timings and dirty-byte counts were not saved, so
we cannot attribute the flush phase more precisely.

The existing peak experiment's maximum demotion is 559 ms, at about 126 MiB resident usage:
5 ms flushing, 186 ms writing a cache snapshot, 367 ms reclaiming. It is not an equivalent state:

- The peak harness triggers at 80% of the original p99 usage, not necessarily the true peak.
- 85 of its 100 trials triggered during the first call, and its flush loop handled at most 49 files.
- For `django-11740`, it stopped during call 15, before the later reset and large writable file set.
- It inspects resident pages before the demotion timer and writes a cache snapshot before reclaim.
- It observes 100 demotions; lazy replay observes 4,487, including later modifications.

The 98 MiB lazy case takes longer mainly because its reclaim phase takes 823 ms, versus 367 ms
in the peak maximum. The logs locate that delay but **do not establish its underlying cause**.
Limit-write latency, writeback waiting, page state, I/O contention and scheduling were not
separately measured. Total resident bytes are not a sequential disk-transfer size, so size alone
cannot predict demotion time or establish comparable disk throughput.

## Why memory remains, and what we can conclude

Remaining memory means RAM accounting, not specifically CPU caches. A live frozen container
still needs kernel structures, and reclaim need not remove every charge. The plotted user
residual is a subtraction of counters, not an inventory of particular files or pages.

Linux documents `memory.usage_in_bytes` as an approximate counter and recommends detailed
`memory.stat` values for more exact accounting. That limits interpretation of small residuals
and changing user/kernel splits. See the [cgroup v1 memory documentation](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v1/memory.html#usage-in-bytes).

The largest residuals illustrate the uncertainty:

| Boundary before call in `django-11740` | Estimated user residual | Kernel residual |
|---|---:|---:|
| 116 | 11.46 MiB | 5.71 MiB |
| 117 | 11.71 MiB | 2.46 MiB |
| 118 | 0.47 MiB | 12.38 MiB |

The changing split makes accounting effects worth investigating; it does not prove the same
pages changed categories, because another command ran between boundaries. The first two
boundaries used 3 and 6 limit-write attempts, not the 20-attempt ceiling.

Some residual may be avoidable: the progress heuristic can stop before delayed reclaim finishes,
and no additional reclaim is attempted after waiting for writeback. The current data cannot
distinguish that from remaining file/anonymous pages, accounting effects or truly unevictable pages.
Calling all residual memory "unreclaimable kernel memory" would be incorrect, even though the
reclaim-loop comment uses that shorthand.

A targeted follow-up would replay `django-11740` around boundaries 116–118, retain full
`memory.stat` snapshots (cache, rss, dirty, writeback, unevictable and kernel/total usage), time each
limit write and writeback wait separately, then retry after a short wait while frozen. Several
stable readings and a bounded timeout would test whether more effort reduces the residual.
This is a proposed diagnostic, **not a change made to the completed experiment**. More thorough
reclaim may cost more time; both residual size and demotion latency should be reported.

## Interpretation limits

- Each task has one measurement per mode, always regular first. There are no repeated-trial
  confidence intervals or randomized-order controls.
- Matching return codes and outcomes do not prove identical outputs or filesystem states.
- Five original-return-code differences remain in the primary comparison because both replays
  agree. Results describe the replayed workload, not guaranteed reproduction of every original effect.
- Residual memory can affect cache warmth; its performance effect cannot be inferred from size alone.
- Tool-time slowdown excludes demotion, boundary work, model inference and setup. It is not an
  end-to-end overhead estimate for a deployed offloading policy.
- Demotion at every call boundary differs from freezing a running command near a memory threshold.

## Tests

Unprivileged tests cover timing boundaries, limit restoration/thaw on failure, reclaim stopping,
and pairing/behavior filtering. They do not substitute for kernel/Docker integration checks.

```sh
MSWEA_SILENT_STARTUP=1 PYTHONDONTWRITEBYTECODE=1 ../../mini-swe-agent/.venv/bin/python -m unittest test_replay_lazy -v
```
