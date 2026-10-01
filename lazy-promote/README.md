# Lazy promotion slowdown

How much slower is each tool call when the container is demoted to disk before it and nothing is
promoted back? The next call faults in only what it touches.

    slowdown = call time (lazy) / call time (regular)

## Modes

Each task is replayed in a fresh container, once per mode (default order: `regular`, then `lazy`):

- **regular:** calls run back to back, nothing demoted. This is the denominator.
- **lazy:** before every call, the container is frozen and demoted exactly as in `peak/replay_offload.py`:
  1. fsync the writable layer (dirty pages to disk)
  2. `memory.swappiness = 100`
  3. squeeze `memory.limit_in_bytes` to ~0: page cache dropped, anonymous/shmem pages swapped
  4. restore the limit, thaw. No promotion.

Both modes freeze and thaw at every boundary. The only difference between them is the demotion. The host
page cache is dropped before every (task, mode), as in the offload experiment.

Between calls, the agent's processes have exited (only the container's `sleep` survives), so what the next
call loses is the page cache: runtime, libraries and repo files.

## Same measurement as the original run

The replay uses mini-swe-agent's own code, so the timing is identical by construction:

- `DockerEnvironment(**config)`: the same `docker run`, and the same `docker exec` command, environment,
  interpreter (`bash -c`) and timeout, from the trajectory's config
- `DefaultAgent._measure_call`: `monotonic_ns` around `env.execute`, as in `step_timings[].tools[].elapsed_s`

Freezing, demotion and counter reads all happen outside the timed window.

The original agent times are kept as `orig_elapsed_s`. They were measured the same way, but with the agent
and the model running and with a different cache state. So the primary denominator is the regular replay,
run by the same harness right before the lazy one. `slowdown_vs_orig` is reported as a second reference.

## Run (operator, as root)

```sh
sudo ../setup_swap.sh on
cd lazy-promote
sudo ../../mini-swe-agent/.venv/bin/python replay_lazy.py ../../swebench-runs/qwen-run-20260923T053154Z-PF12hU
sudo ../../mini-swe-agent/.venv/bin/python replay_lazy.py ../../swebench-runs/qwen-run-20260923T073021Z-LtPzSb
../../mini-swe-agent/.venv/bin/python analyze_lazy.py results
```

Use `--tasks <id>` for a single task, and `--out` for a repeat run. Finished (task, mode) pairs are skipped,
so an interrupted run can be resumed.

## Output

`results/<run>/<task>/<mode>/`:

- `calls.csv`: one row per call:
  - `call_s`, `outcome`, `returncode`, and the original time, outcome and returncode
  - memory at call start, plus major faults, `pgpgin` and blkio read bytes during the call
  - lazy only: demotion time, demoted bytes, residual (user / kernel) and swap
- `trial.json`: per-task totals and errors

`results/<run>/{calls,trials}.csv`: all tasks combined.

`results/analysis/` (from `analyze_lazy.py`):

- `calls.csv`: paired calls, `slowdown`, `added_s`, `slowdown_vs_orig`
- `tasks.csv`: per-task total-time slowdown
- `summary.json`:
  - per-call slowdown (median, p90, geomean …) and the time-weighted overall slowdown
  - `regular_vs_orig`, a noise check of the replay against the original run
  - demotion residuals

A call is compared only if it returned the same returncode and outcome in both modes.

## Tests

```sh
MSWEA_SILENT_STARTUP=1 PYTHONDONTWRITEBYTECODE=1 ../../mini-swe-agent/.venv/bin/python -m unittest test_replay_lazy -v
```
