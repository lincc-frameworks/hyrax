#!/bin/bash
set -euo pipefail

# SessionStart hook for Hyrax - activates the Python virtual environment

# Only run in remote Claude Code on the web environment
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

VENV_PATH="/home/user/hyrax-venv"

# Check if venv exists, if not create it
if [ ! -d "$VENV_PATH" ]; then
  echo "Creating virtual environment at $VENV_PATH..."
  python -m venv "$VENV_PATH"
fi

# Activate the venv and install dependencies
source "$VENV_PATH/bin/activate"
echo "Virtual environment activated: $VENV_PATH"

# Install hyrax in editable mode with dev dependencies
echo "Installing hyrax with dev dependencies..."
pip install -e '.[dev]'

# Export the activation command so it's available for all subsequent commands in the session
echo "export VIRTUAL_ENV=$VENV_PATH" >> "$CLAUDE_ENV_FILE"
echo "export PATH=$VENV_PATH/bin:\$PATH" >> "$CLAUDE_ENV_FILE"

echo "Setup complete!"
