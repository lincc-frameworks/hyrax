#!/bin/bash
set -euo pipefail

# SessionStart hook for Hyrax - activates the Python virtual environment
# in a claude code web environment, and completes any interrupted setup.

# Only run in remote Claude Code on the web environment
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

VENV_PATH="$HOME/hyrax-venv"
SENTINEL="$VENV_PATH/.setup-complete"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Check if venv exists, if not exit (venv must be created by the setup script)
if [ ! -d "$VENV_PATH" ]; then
  echo "No virtual environment found... exiting"
  exit 0
fi

# Run conda-unpack if the container setup script was interrupted before it could.
# On a normal cache hit this sentinel is already present and this block is skipped.
if [ ! -f "$SENTINEL" ]; then
  echo "Running conda-unpack (container setup was interrupted)..."
  export PATH="$VENV_PATH/bin:$PATH"
  conda-unpack
  touch "$SENTINEL"
fi

# Activate the venv. PS1 and CONDA_PREFIX may be unset in non-interactive shells;
# temporarily disable -u so the conda activate script doesn't abort on them.
export CONDA_PREFIX="${CONDA_PREFIX:-}"
export PS1="${PS1:-}"
set +u
source "$VENV_PATH/bin/activate"
set -u

# Ensure hyrax is editable-installed from the current checkout. Editable installs
# are path-specific and must not be cached in the container image, so this runs
# every fresh container start. The path check makes it a no-op on resumed
# containers where the install is already correct (~0.5s vs ~15s).
EXPECTED_INIT="$REPO_ROOT/src/hyrax/__init__.py"
INSTALLED_INIT="$(python -c "import hyrax; print(hyrax.__file__)" 2>/dev/null || true)"
if [ "$INSTALLED_INIT" != "$EXPECTED_INIT" ]; then
  echo "Installing hyrax in editable mode from $REPO_ROOT..."
  python -m pip install -e "$REPO_ROOT[dev]" --quiet
fi

# Persist environment variables to the Claude Code session so they are
# visible in every subsequent Bash tool call this session.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo "export PATH=\"$PATH\"" >> "$CLAUDE_ENV_FILE"
  echo "export CONDA_PREFIX=\"$CONDA_PREFIX\"" >> "$CLAUDE_ENV_FILE"
fi

echo "Virtual environment activated: $VENV_PATH"
