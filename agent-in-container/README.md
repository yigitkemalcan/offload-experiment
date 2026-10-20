# Agent inside the container (SWE-bench, trace replay)

In the original runs (`run-swebench.sh`), one mini-swe-agent controller runs on the host for all tasks and
drives each task's container with `docker exec`. The container's cgroup therefore holds only the commands,
never the agent. Here each task's container runs **its own agent**, so the agent's memory and CPU are part of
the task's cgroup. vLLM is not needed: the agent replays the recorded model replies of a finished run.

| File | Role |
|---|---|
| `setup_runtime.sh` | Builds `runtime/`: a relocatable copy of the CPython the original agent ran on, with this checkout of mini-swe-agent installed at the original venv's package versions. It is mounted read-only into every container at `/opt/mswea-agent`, and the images' own Python is not touched. |
| `agent.py` | Runs **in** the container (main process, under `--init`). It is mini-swe-agent's `DefaultAgent` with two substitutions: `ReplayModel` (LiteLLM's model class returning the recorded replies and re-raising recorded format errors) and `ContainerEnvironment` (the original Docker environment's `bash -c`, `/testbed`, environment variables and 60 s timeout, run with `subprocess`). Writes the trajectory (original format) and its own RSS/PSS/USS samples. With `--hold` (`../end-offload/replay_end.py --agent-inside`, `../compression/replay_compress.py --agent-inside`) it stays alive after the task, with its conversation in memory, until the container is killed. |
| `private_root.py` | `--private`: gives each task container its own copy of the image and the agent's files, written to disk and evicted from the page cache before it starts, and deleted after the task. Everything runs in helper containers, so no root is needed. |
| `replay_in_container.py` | Host launcher. Starts the task containers `--concurrency` at a time (`--network none`), samples their cgroups with the original collector (CPU and memory only), runs `analyze_task_metrics.py`, and compares every replay with its original. |
| `test_agent.py` | No Docker. Replays all 100 traces through the real agent loop with the recorded command outputs: every conversation must come out identical. Also tests the environment (bash, cwd, environment variables, timeout kills the process group). |
| `test_private.py` | With Docker, about 1 min. Two containers read the same 416 MiB file. Shared, the page cache is charged once in total (here not at all, because it was already cached). Private, each container is charged the whole file, even when both read at the same time. |

## Usage

```bash
bash setup_runtime.sh                                         # once (and after changing mini-swe-agent)
runtime/python/bin/python3 -B -m unittest test_agent -v       # offline check
../.venv/bin/python -B -m unittest test_private -v            # share-nothing check (Docker)

# two agents at a time, recorded model replies returned at once
../.venv/bin/python replay_in_container.py ../../swebench-runs/qwen-run-20261017T063628Z-XIPXDG \
    --tasks astropy__astropy-12907 astropy__astropy-13033 --concurrency 2
# same, each model call taking its recorded inference time (realistic timeline)
../.venv/bin/python replay_in_container.py ../../swebench-runs/qwen-run-20261017T063628Z-XIPXDG \
    --tasks astropy__astropy-12907 astropy__astropy-13033 --concurrency 2 --inference-delay
# share nothing: every task on its own uncached copy of its files
../.venv/bin/python replay_in_container.py ../../swebench-runs/qwen-run-20261017T063628Z-XIPXDG \
    --tasks astropy__astropy-12907 astropy__astropy-13033 --concurrency 2 --private
```

Without `--tasks`, every task of the run is replayed in instance-ID order, as a pool of `--concurrency` slots.
Output goes to `results/<run>-c<N>[-delay]-<time>/` with the same layout as an original run (`results/<task>/`,
`system_metrics/`, `task_metrics/`), plus `containers.jsonl` and `replay_summary.csv`. The per-run plot
scripts work on it unchanged.

## Differences from the original run

- The agent runs in the container: its interpreter, LiteLLM and the conversation are now charged to the
  task's cgroup (about 200 MiB RSS per agent).
- Commands are started by the agent with `subprocess` instead of `docker exec` from the host, so tool times
  lose the `docker exec` round trip. On a timeout the command's process group is killed (mini-swe-agent's
  local-environment behaviour); `docker exec` left it running.
- `environment_setup_s` covers `docker run` plus the agent's start-up (Python and LiteLLM imports, about 2.3 s).
  `task_timing.agent_start_mono_ns` marks when the process started.
- Model replies come from the trace; the observations come from this run's own command outputs. A replay is
  *faithful* when the exit status, model calls, commands and command return codes match. Outputs and the
  submission can still differ in timestamps (`ls -l` dates, `diff` headers).
- The runtime is visible inside the container at `/opt/mswea-agent`, so a command like `find /` would see it.
  No command in the 4-worker or the serial traces uses `kill`, `pkill`, `ps` or `/proc`.
- Without `--private`, the host page cache is not dropped. Concurrent tasks share cached files they have
  in common (image layers of the same base or repository version, and the runtime), charged to whichever
  container read them first, or to none of them if the file was already cached.

## Share nothing (`--private`)

SWE-bench images share most of their files: about 1.4 GB of Ubuntu, apt packages and Miniconda per base
family, and 1.33 GB of conda environment per repository version. Only about 0.1 GB (`/testbed`) is unique
to a task. The kernel caches each file page once and charges it to the first cgroup that reads it. Dropping
the whole cache between tasks is impossible while other tasks run, so `--private` gives each task its own
files instead:

1. **Prepare.** A helper container from the task's image runs `cp -a` (as root, so owners, modes,
   timestamps and hard links are kept) on every top-level directory except proc, sys, dev, run and tmp. That
   covers usr, opt, testbed, etc, root, var, home and so on, plus the agent's files, which are mounted
   under /opt. The copy goes to `private/<container>/`, then `sync -f`, then `posix_fadvise(DONTNEED)` on
   every file. `mincore` then checks that nothing of the copy is cached (`resident_bytes` in
   `containers.jsonl`, 0 so far).
2. **Run.** The task container bind-mounts each copied directory over the image's and starts Docker's init
   from its own copy. Every file it reads (libc, bash, git, conda, the repository, the agent's Python) is
   its own, read from disk, and charged to it alone.
3. **Remove.** A helper container deletes the copy after the task.

A copy is 3.42 GB (67k files) for a SWE-bench task. It takes about 22 s to prepare and 3.4 s to remove.
Both happen outside the task's timed window but on the same disk as the running tasks. With `--concurrency
N`, about N × 3.6 GB must be free (checked at start). Still shared: the kernel and its dentry/inode caches,
Docker's own processes, files directly in `/` (such as `/miniconda.sh`, never read by the tasks), and `/run`
and `/tmp`.

What it changes (2-task trial, astropy-12907 and astropy-13033, concurrency 2, no delay; both faithful):

| Task | Page cache charged (peak), shared → private | Usage peak, shared → private |
|---|---|---|
| astropy-13033 | 36 → 205 MiB | 382 → 493 MiB |
| astropy-12907 | 40 → 179 MiB | 238 → 375 MiB |

The shared numbers depend on what other containers (or earlier runs) had already cached. The RSS peaks
(13033: 338 and 281 MiB; 12907: 191 MiB in both) depend on which short-lived test processes the 0.1 s
sampling catches. The cache-adjusted metric of `task_metrics/` (Docker's definition: usage minus inactive
file cache) does not show the difference (average 203 → 189 and 171 → 187 MiB), because pages read once stay
on the inactive list. For share-nothing comparisons, look at `mem_usage_bytes` and `mem_cache_bytes` in
`system_metrics/container_samples.csv`.

## How it was checked

- **Replay logic, offline** (`test_agent.py`): all 100 traces, replayed through the real agent loop with
  the recorded command outputs, give exactly the original conversations, model calls, exit statuses
  (Submitted, RepeatedFormatError, LimitsExceeded) and submissions.
- **Command environment**: inside a container, a command run by `ContainerEnvironment` sees exactly the same
  21 environment variables as one run by `docker exec` with the original settings. An earlier version leaked
  `LC_CTYPE=C.UTF-8` (Python's C-locale coercion); a test now covers it. None of the 4,583 commands in the
  traces timed out, so the different timeout handling never applies to these traces.
- **Private copy**: `mincore` sees a cached file (64 MiB) and 0 after `evict()`. In a container, all 11
  copied directories are bind mounts from `private/`, PID 1 is the copied init, and the copy matches the
  image: all 56,069 files have the same content (md5), owner, mode, timestamp and size. The only difference
  is one `.pyc` in `pkgs/` whose second name a later image layer replaced (overlay still reports 2 links).
- **Sharing** (`test_private.py`): two containers read the same 416 MiB file. Shared, neither was charged
  (already cached). Private, each was charged 420 MiB, also when both read at the same time.
- **Trials**: both modes, 2 tasks at once (started 0.34 s apart), both faithful. 1 of 108 command outputs
  differs, only in `ls -la` dates; sampling covered each container at about 10 samples/s. The faithful
  check compares outputs and submissions with dates and times masked, and flags any other difference.
- **Not tested**: `--inference-delay` end to end (only the sleep, offline), stopping a run with Ctrl-C,
  and more than 2 tasks.
