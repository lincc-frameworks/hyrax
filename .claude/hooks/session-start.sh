#!/bin/bash
set -euo pipefail

# SessionStart hook for Hyrax - activates the Python virtual environment
# in a claude code web virtual environment

# Only run in remote Claude Code on the web environment
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

VENV_PATH="$HOME/hyrax-venv"

# Check if venv exists, if not create it
if [ ! -d "$VENV_PATH" ]; then
  echo "No virtual environment found... exiting"
  exit 0
fi

# Activate the venv and install dependencies
source "$VENV_PATH/bin/activate"
echo "Virtual environment activated: $VENV_PATH"
