#!/usr/bin/env python3
"""The mini-SWE-agent controller of one task, run INSIDE the task's container, replaying a recorded trajectory.

Started by replay_in_container.py as the container's main process (under --init) with the runtime from
setup_runtime.sh mounted read-only at /opt/mswea-agent. It runs mini-swe-agent's own DefaultAgent loop, so
the container holds what the original host-side controller held for this task: the interpreter with
mini-swe-agent and LiteLLM imported, the growing conversation (recorded model messages, observations from
this run's own command outputs), step timings and the trajectory it saves. Only two parts are replaced:

  model        ReplayModel: LiteLLM's model class (same observation formatting, same imports) whose query()
               returns the recorded reply of the step instead of calling vLLM, or re-raises the recorded
               FormatError; --inference-delay first sleeps the step's recorded inference time.
  environment  ContainerEnvironment: the original DockerEnvironment's command semantics (interpreter
               `bash -c`, cwd /testbed, its environment variables, its timeout), executed with subprocess
               from inside the container instead of `docker exec` from the host. On a timeout the
               command's process group is killed (LocalEnvironment's behaviour); `docker exec` left it running.

The first two messages (system prompt, task) are the recorded ones. Commands get the environment this
process was started with (/proc/self/environ), so nothing set by or for the agent leaks into them.
Writes to --output: <instance_id>.traj.json (the original runner's format: step_timings, task_timing,
container_id, exit_status, submission) and agent_process_samples.csv (this process's RSS/PSS/USS every
--sample-interval seconds from the end of its imports, columns as in system_metrics.py). All times are
CLOCK_MONOTONIC (shared with the host: containers have no time namespace).

--repeat (evaluation/capacity_run.py): the process stays up and runs the task again and again, each cycle with a
fresh agent (empty conversation): run, restore every --restore directory (/testbed, /tmp) exactly from its
pristine copy (private_root.py --pristine: contents, types, modes, owners and timestamps; only what differs
is rewritten), idle for a random --idle time, run again. Each cycle's trajectory is cycle-NNN.traj.json, and
cycles.jsonl logs every cycle and idle period. SIGTERM (`docker stop`) ends it: a cycle in progress is saved
with exit status "Stopped".

--hold (../end-offload, ../compression --agent-inside), as the DABStep replay's driver
(../../offload-experiment-dabstep/replay/driver.py): the task runs once, and when the agent has finished (run()
returned, before the trajectory is saved) the process writes done.json and waits for the host's release file,
with the agent and its conversation in memory, so the host can freeze and measure the container at the end of
the task. After the release it saves the trajectory and exits. The recorded trajectory, the replay's input, is
evicted from the page cache right after it is read, and done.json right after it is written: neither is the
agent's memory.
"""
import argparse
import copy
import csv
import json
import os
import random
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

AGENT_START_NS = time.monotonic_ns()
# The environment the process was started with (what `docker exec` gives a command), not os.environ: Python's
# start-up already added LC_CTYPE=C.UTF-8 there (C locale coercion), and below adds agent-only variables.
BASE_ENV = dict(os.fsdecode(x).split("=", 1) for x in Path("/proc/self/environ").read_bytes().split(b"\0") if x)
os.environ |= {
    "MSWEA_SILENT_STARTUP": "1",
    "MSWEA_GLOBAL_CONFIG_DIR": str(Path(sys.prefix).parent / "config"),  # exists in the read-only runtime
    "LITELLM_LOCAL_MODEL_COST_MAP": "True",  # --network none: don't try to fetch the cost map
}

from minisweagent.agents.default import DefaultAgent  # noqa: E402
from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig  # noqa: E402
from minisweagent.exceptions import FormatError  # noqa: E402
from minisweagent.models.litellm_model import LitellmModel  # noqa: E402

AGENT_PROCESS_COLUMNS = ["t_mono_ns", "pid", "rss_bytes", "pss_bytes", "uss_bytes"]


class ReplayExhausted(Exception):
    """The agent asked for more model replies than the trajectory recorded (the replay diverged)."""


def recorded_steps(trajectory: dict) -> list[dict]:
    """One entry per model call, in order: the reply message, or the FormatError's messages."""
    steps = []
    for message in trajectory["messages"][2:]:
        if message["role"] == "assistant":
            steps.append({"kind": "returned", "message": message})
        elif message["role"] == "user" and "interrupt_type" in message.get("extra", {}):
            steps.append({"kind": message["extra"]["interrupt_type"], "messages": [message]})
    for step, timing in zip(steps, trajectory["step_timings"]):
        step["inference_s"] = timing["inference"]["elapsed_s"]
    return steps


class ReplayModel(LitellmModel):
    def __init__(self, *, steps: list[dict], inference_delay: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.steps, self.inference_delay = list(steps), inference_delay

    def query(self, messages: list[dict], **kwargs) -> dict:
        if not self.steps:
            raise ReplayExhausted("no recorded model reply left")
        step = self.steps.pop(0)
        if self.inference_delay:
            time.sleep(step["inference_s"])
        if step["kind"] == "FormatError":
            raise FormatError(*copy.deepcopy(step["messages"]))
        message = copy.deepcopy(step["message"])
        message["extra"]["timestamp"] = time.time()
        return message


class ContainerEnvironmentConfig(LocalEnvironmentConfig):
    interpreter: list[str] = ["bash", "-c"]


class ContainerEnvironment(LocalEnvironment):
    def __init__(self, **kwargs):
        super().__init__(config_class=ContainerEnvironmentConfig, **kwargs)

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
        command = action.get("command", "")
        try:
            result = run([*self.config.interpreter, command], cwd or self.config.cwd, BASE_ENV | self.config.env,
                         timeout or self.config.timeout)
            output = {"output": result.stdout, "returncode": result.returncode, "exception_info": ""}
        except Exception as e:  # same observation as DockerEnvironment.execute
            raw_output = getattr(e, "output", None)
            raw_output = raw_output.decode("utf-8", errors="replace") if isinstance(raw_output, bytes) else (raw_output or "")
            output = {
                "output": raw_output,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        self._check_finished(output)
        return output


def run(args: list[str], cwd: str, env: dict, timeout: int) -> subprocess.CompletedProcess:
    """minisweagent.environments.local._run with an argument list instead of `sh -c`."""
    process = subprocess.Popen(args, text=True, cwd=cwd, env=env, encoding="utf-8", errors="replace",
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        stdout, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, _ = process.communicate()
        raise subprocess.TimeoutExpired(args, timeout, output=stdout)
    return subprocess.CompletedProcess(args, process.returncode, stdout=stdout)


class ReplayAgent(DefaultAgent):
    def __init__(self, *args, prompts: list[str], **kwargs):
        super().__init__(*args, **kwargs)
        self.prompts = list(prompts)

    def _render_template(self, template: str) -> str:
        return self.prompts.pop(0)  # run() renders the system prompt, then the task: use the recorded ones


def own_container_id() -> str | None:
    match = re.search(r"/containers/([0-9a-f]{64})/", Path("/proc/self/mountinfo").read_text())
    return match and match.group(1)


def sample_memory(path: Path, interval: float, stop: threading.Event):
    with path.open("w", newline="", buffering=1 << 16) as f:
        writer = csv.writer(f)
        writer.writerow(AGENT_PROCESS_COLUMNS)
        while not stop.is_set():
            t = time.monotonic_ns()
            values = {line.split(":", 1)[0]: int(line.split()[1]) * 1024
                      for line in Path("/proc/self/smaps_rollup").read_text().splitlines()[1:]}
            uss = sum(values.get(k, 0) for k in ("Private_Clean", "Private_Dirty", "Private_Hugetlb"))
            writer.writerow([t, os.getpid(), values["Rss"], values["Pss"], uss])
            stop.wait(interval)


class Stopped(BaseException):
    """SIGTERM (`docker stop`): end the current cycle where it is. Not an Exception, so the agent loop lets it pass."""


def stop_on_sigterm(*_):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)  # the first one is enough; don't interrupt the save
    raise Stopped()


def evict(path: Path):
    """Write the file to disk and drop it from the page cache."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def hold(output: Path, end: dict):
    """--hold: tell the host the task is over (done.json) and wait for its release file."""
    done, release = output / "done.json", output / "release"
    tmp = done.with_suffix(".tmp")
    tmp.write_text(json.dumps(end))
    evict(tmp)
    tmp.rename(done)  # appears whole
    while not release.exists():
        time.sleep(0.01)


def run_cycle(trajectory: dict, args, path: Path, start_ns: int, cycle: int | None = None) -> dict:
    """Run the task once with a fresh agent (empty conversation), save its trajectory to path."""
    config = trajectory["info"]["config"]
    env_config = {k: config["environment"][k] for k in ("cwd", "env", "timeout", "interpreter")}
    model = ReplayModel(steps=recorded_steps(trajectory), inference_delay=args.inference_delay, **config["model"])
    agent = ReplayAgent(model, ContainerEnvironment(**env_config), **config["agent"],
                        prompts=[m["content"] for m in trajectory["messages"][:2]])
    ready_ns = time.monotonic_ns()
    extra_info = {}
    try:
        info = agent.run()
        exit_status, submission = info.get("exit_status"), info.get("submission")
    except Stopped:
        exit_status, submission = "Stopped", ""
    except Exception as e:
        exit_status, submission = type(e).__name__, ""
        extra_info = {"traceback": traceback.format_exc(), "exception_str": str(e)}
    end_ns = time.monotonic_ns()
    if args.hold:
        hold(args.output, {"ready_mono_ns": ready_ns, "end_mono_ns": end_ns, "exit_status": exit_status,
                           "model_calls": agent.n_calls})
    agent.save(path, {
        "info": {
            "exit_status": exit_status, "submission": submission, "container_id": own_container_id(),
            "task_timing": {"start_mono_ns": start_ns, "environment_ready_mono_ns": ready_ns,
                            "end_mono_ns": end_ns, "agent_start_mono_ns": AGENT_START_NS},
            "replay": {"original_container_id": trajectory["info"].get("container_id"),
                       "inference_delay": args.inference_delay, "cycle": cycle},
            **extra_info,
        },
        "instance_id": trajectory["instance_id"],
    })
    return {"cycle": cycle, "start_mono_ns": start_ns, "ready_mono_ns": ready_ns, "end_mono_ns": end_ns,
            "exit_status": exit_status, "model_calls": agent.n_calls}


def _lstat(path: str) -> os.stat_result | None:
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None


def restore_tree(pristine: str, live: str) -> dict:
    """Make live identical to pristine: same entries, types, contents, symlink targets, modes, owners, atimes and
    mtimes. Only entries that differ are rewritten (a regular file whose size and mtime match is kept, like rsync)."""
    removed = rewritten = 0
    for dirpath, dirnames, filenames in os.walk(live):  # 1. remove what pristine lacks or has as another type
        rel = os.path.relpath(dirpath, live)
        for name in [*dirnames, *filenames]:
            path, st = os.path.join(dirpath, name), os.lstat(os.path.join(dirpath, name))
            ref = _lstat(os.path.join(pristine, rel, name))
            if ref is None or stat.S_IFMT(ref.st_mode) != stat.S_IFMT(st.st_mode):
                if stat.S_ISDIR(st.st_mode):
                    shutil.rmtree(path)
                    dirnames.remove(name)
                else:
                    os.unlink(path)
                removed += 1
    for dirpath, dirnames, filenames in os.walk(pristine):  # 2. create or rewrite what is missing or differs
        target = os.path.join(live, os.path.relpath(dirpath, pristine))
        for name in [*dirnames, *filenames]:  # by type: a symlink to a directory is listed in dirnames
            src, dst = os.path.join(dirpath, name), os.path.join(target, name)
            ref, st = os.lstat(src), _lstat(dst)
            if stat.S_ISDIR(ref.st_mode):
                if st is None:
                    os.mkdir(dst)
                    rewritten += 1
            elif stat.S_ISLNK(ref.st_mode):
                if st is None or os.readlink(dst) != os.readlink(src):
                    if st is not None:
                        os.unlink(dst)
                    os.symlink(os.readlink(src), dst)
                    rewritten += 1
            elif stat.S_ISREG(ref.st_mode) and (st is None or (st.st_size, st.st_mtime_ns) != (ref.st_size, ref.st_mtime_ns)):
                shutil.copyfile(src, dst)
                rewritten += 1
    for dirpath, dirnames, filenames in os.walk(pristine, topdown=False):  # 3. metadata, children before parents
        rel = os.path.relpath(dirpath, pristine)
        for name in [*filenames, *dirnames, *([""] if rel == "." else [])]:
            src, dst = os.path.join(dirpath, name), os.path.normpath(os.path.join(live, rel, name))
            ref, st = os.lstat(src), _lstat(dst)
            if st is None:  # a type restore does not create (fifo, socket, device)
                continue
            if (st.st_uid, st.st_gid) != (ref.st_uid, ref.st_gid):
                os.lchown(dst, ref.st_uid, ref.st_gid)
            if not stat.S_ISLNK(ref.st_mode) and stat.S_IMODE(st.st_mode) != stat.S_IMODE(ref.st_mode):
                os.chmod(dst, stat.S_IMODE(ref.st_mode))
            if (st.st_atime_ns, st.st_mtime_ns) != (ref.st_atime_ns, ref.st_mtime_ns):
                os.utime(dst, ns=(ref.st_atime_ns, ref.st_mtime_ns), follow_symlinks=False)
    return {"removed": removed, "rewritten": rewritten}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", type=Path, required=True, help="the original task's .traj.json")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--start-ns", type=int, required=True, help="host CLOCK_MONOTONIC just before `docker run`")
    ap.add_argument("--inference-delay", action="store_true", help="sleep each step's recorded inference time")
    ap.add_argument("--sample-interval", type=float, default=0.1,
                    help="seconds between samples of this process's memory (smaps_rollup); 0: no sampling")
    ap.add_argument("--owner", help="UID:GID to give the output files")
    ap.add_argument("--repeat", action="store_true",
                    help="run the task again and again until SIGTERM: restore, idle, run (cycle-NNN.traj.json)")
    ap.add_argument("--restore", action="append", default=[], metavar="DIR=PRISTINE",
                    help="--repeat: after every cycle make DIR identical to PRISTINE (e.g. /testbed=<pristine copy>)")
    ap.add_argument("--idle", type=float, nargs=2, default=(180, 300), metavar=("MIN", "MAX"),
                    help="--repeat: idle seconds between cycles, uniform in [MIN, MAX], drawn per cycle")
    ap.add_argument("--seed", type=int, default=0, help="--repeat: seed of the idle times")
    ap.add_argument("--hold", action="store_true",
                    help="at the end of the task write <output>/done.json and wait for <output>/release")
    args = ap.parse_args()
    if args.hold and args.repeat:
        ap.error("--hold runs the task once: not with --repeat")

    trajectory = json.loads(args.trace.read_text())
    if args.hold:
        evict(args.trace)
    instance_id = trajectory["instance_id"]
    stop = threading.Event()
    sampler = None
    if args.sample_interval > 0:
        sampler = threading.Thread(target=sample_memory, daemon=True,
                                   args=(args.output / "agent_process_samples.csv", args.sample_interval, stop))
        sampler.start()
    try:
        if not args.repeat:
            record = run_cycle(trajectory, args, args.output / f"{instance_id}.traj.json", args.start_ns)
            print(f"{instance_id}: {record['exit_status']} after {record['model_calls']} model calls", flush=True)
            return
        signal.signal(signal.SIGTERM, stop_on_sigterm)
        rng, cycle, start = random.Random(args.seed), 0, args.start_ns
        with (args.output / "cycles.jsonl").open("w") as log:
            try:
                while True:
                    record = run_cycle(trajectory, args, args.output / f"cycle-{cycle:03d}.traj.json", start, cycle)
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(f"{instance_id} cycle {cycle}: {record['exit_status']} after {record['model_calls']} model calls",
                          flush=True)
                    if record["exit_status"] == "Stopped":
                        break
                    t = time.monotonic()
                    restored = {d: restore_tree(src, d) for d, src in (x.split("=", 1) for x in args.restore)}
                    idle = {"restore_s": time.monotonic() - t, "restored": restored, "idle_s": rng.uniform(*args.idle)}
                    log.write(json.dumps({"cycle": cycle, "idle_start_mono_ns": time.monotonic_ns(), **idle}) + "\n")
                    log.flush()
                    time.sleep(idle["idle_s"])
                    cycle, start = cycle + 1, time.monotonic_ns()
            except Stopped:  # while restoring or idle
                pass
    finally:
        stop.set()
        if sampler:
            sampler.join()
        if args.owner:
            uid, gid = map(int, args.owner.split(":"))
            for p in args.output.iterdir():
                os.chown(p, uid, gid)


if __name__ == "__main__":
    main()
