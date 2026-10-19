#!/usr/bin/env bash
# Build runtime/: the agent's Python, mounted read-only into every task container at /opt/mswea-agent.
# A copy of the uv-managed CPython the original controller ran on (relocatable, so it works at any mount
# point and on the images' glibc), with this checkout of mini-swe-agent installed (not editable) into its own
# site-packages at the package versions of mini-swe-agent/.venv, so the agent matches the one that recorded
# the traces. SWE-bench images keep their own Python untouched. No root needed.
#   bash setup_runtime.sh
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
mswea=$(cd "$here/../../mini-swe-agent" && pwd)
home=$(sed -n 's/^home = //p' "$mswea/.venv/pyvenv.cfg")  # the venv's base interpreter: <python>/bin
python_src=$(realpath "${MSWEA_PYTHON_SRC:-$(dirname "$home")}")
runtime="$here/runtime"
rm -rf "$runtime"
mkdir -p "$runtime/config"  # MSWEA_GLOBAL_CONFIG_DIR: must exist, the mount is read-only
cp -a "$python_src" "$runtime/python"
uv pip freeze --python "$mswea/.venv/bin/python" | grep -v -e '^-e ' -e '^mini-swe-agent' > "$runtime/constraints.txt"
uv pip install --python "$runtime/python/bin/python3" --break-system-packages --compile-bytecode \
    -c "$runtime/constraints.txt" "$mswea"
git -C "$mswea" rev-parse HEAD > "$runtime/mini-swe-agent-commit.txt"
git -C "$mswea" status --porcelain > "$runtime/mini-swe-agent-status.txt"
"$runtime/python/bin/python3" -c 'import minisweagent, litellm, sys; print("runtime ok:", sys.version.split()[0], "mini-swe-agent", minisweagent.__version__)'
du -sh "$runtime"
