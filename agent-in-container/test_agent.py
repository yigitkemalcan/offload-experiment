"""agent.py without Docker. Run with the runtime's Python (it has mini-swe-agent), from this directory:

    runtime/python/bin/python3 -m unittest test_agent -v
    SWE_TRACE_RUN=../../swebench-runs/<run> runtime/python/bin/python3 -m unittest test_agent -v

ReplayTest replays every trajectory of the trace run through the real agent loop, with an environment that
returns the recorded command outputs instead of running them: the conversation must come out identical to
the original (roles, contents, tool call IDs), with the same model calls, exit status and submission.
"""
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import agent as ag
from minisweagent.exceptions import Submitted

PROJECT = Path(__file__).resolve().parents[2]
TRACE_RUN = Path(os.environ.get("SWE_TRACE_RUN", PROJECT / "swebench-runs" / "qwen-run-20261017T063628Z-XIPXDG"))


class RecordedOutputs(ag.ContainerEnvironment):
    """Returns the original run's outputs in order; the final submission command prints the submission."""

    def __init__(self, trajectory: dict, **kwargs):
        super().__init__(**kwargs)
        self.outputs = [m["extra"] for m in trajectory["messages"] if m["role"] == "tool"]
        self.submission = trajectory["info"]["submission"]

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
        if self.outputs:
            e = self.outputs.pop(0)
            output = {"output": e["raw_output"], "returncode": e["returncode"], "exception_info": e["exception_info"]}
        else:
            output = {"output": f"COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n{self.submission}", "returncode": 0,
                      "exception_info": ""}
        self._check_finished(output)
        return output


def conversation(messages: list[dict]) -> list[tuple]:
    return [(m["role"], m["content"], m.get("tool_call_id"), m.get("extra", {}).get("exit_status"))
            for m in messages]


class ReplayTest(unittest.TestCase):
    def test_recorded_steps_match_step_timings(self):
        for path in sorted(TRACE_RUN.glob("results/*/*.traj.json")):
            t = json.loads(path.read_text())
            steps = ag.recorded_steps(t)
            self.assertEqual([s["kind"] for s in steps], [s["inference"]["outcome"] for s in t["step_timings"]], path)
            self.assertEqual(len(steps), t["info"]["model_stats"]["api_calls"], path)

    def test_replay_reproduces_every_conversation(self):
        paths = sorted(TRACE_RUN.glob("results/*/*.traj.json"))
        self.assertEqual(len(paths), 100)
        for path in paths:
            with self.subTest(task=path.parent.name):
                t = json.loads(path.read_text())
                config = t["info"]["config"]
                env = RecordedOutputs(t, **{k: config["environment"][k] for k in ("cwd", "env", "timeout", "interpreter")})
                model = ag.ReplayModel(steps=ag.recorded_steps(t), **config["model"])
                agent = ag.ReplayAgent(model, env, **config["agent"], prompts=[m["content"] for m in t["messages"][:2]])
                info = agent.run()
                self.assertEqual(conversation(agent.messages), conversation(t["messages"]))
                self.assertEqual(agent.n_calls, t["info"]["model_stats"]["api_calls"])
                self.assertEqual((info["exit_status"], info["submission"]), (t["info"]["exit_status"], t["info"]["submission"]))

    def test_running_out_of_replies_is_an_error(self):
        model = ag.ReplayModel(steps=[], model_name="hosted_vllm/x")
        with self.assertRaises(ag.ReplayExhausted):
            model.query([])

    def test_inference_delay_sleeps_the_recorded_time(self):
        step = {"kind": "returned", "message": {"role": "assistant", "content": "", "extra": {"actions": []}},
                "inference_s": 0.3}
        model = ag.ReplayModel(steps=[step], inference_delay=True, model_name="hosted_vllm/x")
        t0 = time.monotonic()
        model.query([])
        self.assertGreaterEqual(time.monotonic() - t0, 0.3)


def tree(root: Path) -> dict:
    """Every entry: type, mode, owner, size, mtime, and content hash or symlink target."""
    out = {}
    for p in [root, *sorted(root.rglob("*"))]:
        st = p.lstat()
        data = os.readlink(p) if p.is_symlink() else hashlib.md5(p.read_bytes()).hexdigest() if p.is_file() else None
        out[str(p.relative_to(root))] = (stat.S_IFMT(st.st_mode), stat.S_IMODE(st.st_mode), st.st_uid, st.st_gid,
                                         None if p.is_dir() else st.st_size, st.st_mtime_ns, data)
    return out


class RestoreTreeTest(unittest.TestCase):
    def test_live_becomes_identical_to_pristine_rewriting_only_changes(self):
        base = Path(tempfile.mkdtemp())
        pristine, live = base / "pristine", base / "live"
        (pristine / "pkg" / "sub").mkdir(parents=True)
        (pristine / "pkg" / "a.py").write_text("a = 1\n")
        (pristine / "pkg" / "sub" / "b.py").write_text("b = 2\n")
        (pristine / "README").write_text("readme\n")
        (pristine / "same").write_text("untouched\n")
        (pristine / "link").symlink_to("README")
        (pristine / "dirlink").symlink_to("pkg")
        os.chmod(pristine / "pkg" / "a.py", 0o777)
        os.chmod(pristine / "pkg" / "sub", 0o750)
        for i, p in enumerate(sorted(pristine.rglob("*"), reverse=True)):
            os.utime(p, ns=(10**18 + i, 10**18 + i), follow_symlinks=False)
        subprocess.run(["cp", "-a", pristine, live], check=True)
        inode = (live / "same").stat().st_ino
        (live / "pkg" / "a.py").write_text("a = 9\n")           # same size, new content and mtime
        os.chmod(live / "README", 0o600)                          # metadata only
        (live / "pkg" / "sub" / "b.py").unlink()                  # deleted
        (live / "new").mkdir(); (live / "new" / "x").write_text("x")  # added (nested)
        (live / "link").unlink(); (live / "link").symlink_to("pkg")   # retargeted
        (live / "dirlink").unlink(); (live / "dirlink").mkdir()       # symlink replaced by a directory
        (live / "pkg" / "c.pyc").write_bytes(b"\0")                  # ignored-style artifact
        self.assertNotEqual(tree(live), tree(pristine))
        counts = ag.restore_tree(str(pristine), str(live))
        self.assertEqual(tree(live), tree(pristine))
        self.assertEqual((live / "same").stat().st_ino, inode)  # an unchanged file is not rewritten
        self.assertEqual(ag.restore_tree(str(pristine), str(live)), {"removed": 0, "rewritten": 0})
        self.assertGreater(counts["removed"], 0)


class ContainerEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        # generous: on a saturated host even starting bash can take seconds; only the timeout test uses 2 s
        self.env = ag.ContainerEnvironment(cwd=self.dir, env={"PAGER": "cat"}, timeout=60, interpreter=["bash", "-c"])

    def test_bash_cwd_and_environment(self):
        out = self.env.execute({"command": '[[ -n $PAGER ]] && echo "$PAGER $PWD ${MSWEA_SILENT_STARTUP:-unset}"'})
        self.assertEqual((out["returncode"], out["output"]), (0, f"cat {self.dir} unset\n"))

    def test_commands_get_exactly_the_start_environment(self):
        """C locale, like the images: Python sets LC_CTYPE=C.UTF-8 in os.environ; commands must not see it."""
        start_env = {"PATH": "/usr/bin:/bin", "HOME": "/root", "HOSTNAME": "x"}
        code = ("import agent; e = agent.ContainerEnvironment(cwd='/', env={'PAGER': 'cat'}, timeout=5, "
                "interpreter=['bash', '-c']); print(e.execute({'command': 'env -0'})['output'], end='')")
        out = subprocess.run([sys.executable, "-I", "-B", "-c", f"import sys; sys.path.insert(0, {str(Path(__file__).parent)!r}); {code}"],
                             env=start_env, capture_output=True, check=True).stdout
        seen = dict(x.split(b"=", 1) for x in out.split(b"\0") if x)
        bash_own = {b"PWD", b"SHLVL", b"_"}
        self.assertEqual({k: v for k, v in seen.items() if k not in bash_own},
                         {k.encode(): v.encode() for k, v in (start_env | {"PAGER": "cat"}).items()})

    def test_timeout_kills_the_command_and_its_children(self):
        token = f"{time.time_ns()}"
        out = self.env.execute({"command": f"bash -c 'exec -a sleeper-{token} sleep 30' & echo started; wait"}, timeout=2)
        self.assertEqual(out["returncode"], -1)
        self.assertIn("timed out", out["exception_info"])
        self.assertEqual(out["output"], "started\n")
        deadline = time.monotonic() + 5  # SIGKILL is asynchronous: wait for the processes to be gone
        while subprocess.run(["pgrep", "-f", f"sleeper-{token}"], capture_output=True).returncode == 0:
            self.assertLess(time.monotonic(), deadline, "a child of the timed-out command survived")
            time.sleep(0.05)

    def test_submission_marker_ends_the_task(self):
        with self.assertRaises(Submitted) as e:
            self.env.execute({"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT; echo patch"})
        self.assertEqual(e.exception.messages[0]["extra"]["submission"], "patch\n")


if __name__ == "__main__":
    unittest.main()
