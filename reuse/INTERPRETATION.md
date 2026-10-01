# Interpreting the reuse results

Notes on what the categories mean and how to read the numbers. Example figures are from the
first 69 tasks (42 PF12hU + 27 LtPzSb, 2,879 tool calls). See README.md for how the measurement works.

## Memory kinds: how the kernel backs a page

| kind | what it is | group |
|---|---|---|
| `file` | page cache of regular files in the container's filesystem (all overlay layers) | cache |
| `shm` | files in the container's `/dev/shm`: RAM-only (tmpfs), never written to disk | cache |
| `mapped` | shared pages held by a live process whose file was not walked: memfd, shared anonymous memory, files from outside the rootfs | cache |
| `anon` | private memory of a live process: heap, stack, copy-on-write copies | anon |

`shm` and `mapped` are not disk caches, but the kernel keeps them in the page cache too (they count
as "Cached" in `/proc/meminfo`), so they are grouped with cache.

## Sources: where a page comes from (by path, analysis only)

| source | rule | kind |
|---|---|---|
| environment | `/opt`, `/usr`, `/lib`, `/bin`, `/sbin`, `/etc`, `/root`, `/var` (conda, Python, libraries, tools, config) | file |
| task repo | `/testbed` | file |
| `/tmp` | `/tmp`: a normal directory on the container filesystem, so ordinary file cache | file |
| process memory | all anonymous pages | anon |

## Why `shm`, `mapped` and `anon` are (almost) zero

Each tool call runs as `docker exec … bash -c "<command>"`, and everything it starts exits before the
container is frozen. The only process alive at every boundary is the container's `sleep`.

- `shm`: nothing in these tasks leaves files in `/dev/shm`.
- `mapped` and `anon`: these exist only while a process holds them. Memory used by processes
  during a call is freed before we look.
- The keep-alive `sleep` is our harness, not the agent, and is **not measured**. Freezing interrupts
  its `nanosleep`, and the kernel then writes the remaining time to its stack. That single stack page
  was the only anon page in the first 69 tasks (2,879 page-uses). `replay_reuse.py` now leaves the
  keep-alive out. For results recorded before that change, `analyze_reuse.py` and `plot_reuse.py`
  remove its page (process id 0) themselves. Their `calls.csv`/`trial.json` still include it.
- After a thaw, `sleep` runs a few `libc` instructions to restart its sleep. Those may touch a `libc`
  page that bash and Python use in every call anyway. This cannot be separated, and is at most a
  page or two per call.

So the zeros mean **no shared or private memory survives from one tool call to the next**, not
that none is used inside a call. The kernel test shows memfd, `/dev/shm` and anon pages are detected
when their process stays alive. Measuring in-call memory would require sampling during the call.

## Why reuse is high

- **Per-call startup.** The task sets `BASH_ENV=/root/.bashrc`, which runs `conda activate testbed`.
  So every call starts bash and Python and loads conda's modules: about 3,100 identical pages per
  call, the same as in the original agent run. Environment pages are 89% of all page-uses, and 92% of
  them are reused, almost all at distance 1.
- **Repeated reads by the agent.** Repo-wide `find | xargs grep` is repeated, and files are `cat`ed
  again after an earlier grep already read them. A call is 100% reused when it only touches pages an
  earlier call touched.
- **Task repo alone:** 51% reused, with longer distances (69% at distance 1, p90 = 8 calls).

## Two ways to report the reuse fraction

- **Page-weighted** (reused page-uses / all page-uses): 87.8% overall, task median 84%. Large
  calls, such as a first grep over the whole repo, dominate.
- **Per-call mean** (each call's reuse fraction averaged, every call weighted equally): task median 92%.

Both are in `analysis/summary.json`. Report the environment / task-repo split next to them, since
most reuse is per-call startup rather than the agent's work on the code.

## Measurement caveats

- **Kernel 5.15 `read()` bug.** It never marks the first page of each 15-page batch (after the first)
  as accessed. An idle file page whose previous and next pages were both used is counted as used.
- **Loading counts as use**, readahead included.
- **Pages loaded and evicted within one call are not seen.** This is rare with ample memory.
- **One replay difference in 2,879 calls** (django-11740): a migration file name contains its
  creation time, so `cat` of the original name fails in the replay.

## Plots

- `reuse_per_task.png`: the "other" segment (`/tmp`, shm, any agent process memory) is ≤ 0.02% of
  each task, so it is not visible.
- `reuse_distance_cdf.png`: the "other" line has only a handful of `/tmp` reuses. 1.7% of file reuses cannot be matched to a source; they appear only in the
  "All" line. The y-axis starts at 60%.
