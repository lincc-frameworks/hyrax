#!/bin/bash
set -euo pipefail

# To use this as your maintenance script in codex
# put the following line into the maintenance script
# setup box on the webpage:
#
# ./agent_scripts/codex_maintain_container.sh
#
# This script is intentionally idempotent. It ensures:
# 1) hyrax is editable-installed from the current checkout path
# 2) login shells auto-activate the hyrax conda-packed environment

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

ENV_DIR="$HOME/hyrax-venv"

# shellcheck disable=SC1091
if [ -f "${ENV_DIR}/bin/activate" ]; then
  set +u
  source "${ENV_DIR}/bin/activate"
  set -u
fi

# Ensure hyrax is editable-installed from the current checkout path.
EXPECTED_INIT="$REPO_ROOT/src/hyrax/__init__.py"
INSTALLED_INIT="$(python -c "import hyrax; print(hyrax.__file__)" 2>/dev/null || true)"
if [ "$INSTALLED_INIT" != "$EXPECTED_INIT" ]; then
  python -m pip install -e '.[dev]'
fi

# Persist venv pathing for future non-interactive login shells in Codex.
PROFILE_FILE="$HOME/.profile"
BEGIN_MARKER="# >>> hyrax codex venv >>>"
END_MARKER="# <<< hyrax codex venv <<<"
if ! grep -Fq "$BEGIN_MARKER" "$PROFILE_FILE"; then
  printf '\n' >> "$PROFILE_FILE"
  cat >> "$PROFILE_FILE" <<PROFILE_BLOCK
$BEGIN_MARKER
if [ -f "$HOME/hyrax-venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$HOME/hyrax-venv/bin/activate"
fi
$END_MARKER
PROFILE_BLOCK
fi
