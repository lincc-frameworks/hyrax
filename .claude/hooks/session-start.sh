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

# Complete setup if the init script was interrupted before finishing.
# conda-unpack fixes hardcoded paths baked in at CI build time, and
# pip install -e . ensures hyrax resolves to the local working tree.
if [ ! -f "$SENTINEL" ]; then
  echo "Completing venv setup (this runs once)..."
  export PATH="$VENV_PATH/bin:$PATH"
  conda-unpack
  python -m pip install -e "$REPO_ROOT[dev]"
  touch "$SENTINEL"
  echo "Venv setup complete."
fi

# Activate the venv. CONDA_PREFIX may be unset; guard against set -u.
export CONDA_PREFIX="${CONDA_PREFIX:-}"
source "$VENV_PATH/bin/activate"

# Persist environment variables to the Claude Code session so they are
# visible in every subsequent Bash tool call this session.
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo "export PATH=\"$PATH\"" >> "$CLAUDE_ENV_FILE"
  echo "export CONDA_PREFIX=\"$CONDA_PREFIX\"" >> "$CLAUDE_ENV_FILE"
fi

echo "Virtual environment activated: $VENV_PATH"
