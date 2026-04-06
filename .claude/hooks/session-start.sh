#!/bin/bash
set -euo pipefail

# SessionStart hook for Hyrax - activates the Python virtual environment
# in a claude code web virtual environment

# Only run in remote Claude Code on the web environment
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

VENV_PATH="$HOME/hyrax-venv"

# Check if venv exists, if not exit (venv must be created by the setup script)
if [ ! -d "$VENV_PATH" ]; then
  echo "No virtual environment found... exiting"
  exit 0
fi

# Activate the venv to capture environment modifications
source "$VENV_PATH/bin/activate"

# Persist environment variables to the Claude Code session
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo "export PATH=\"$PATH\"" >> "$CLAUDE_ENV_FILE"
  echo "export VIRTUAL_ENV=\"$VIRTUAL_ENV\"" >> "$CLAUDE_ENV_FILE"
fi

echo "Virtual environment activated: $VENV_PATH"
