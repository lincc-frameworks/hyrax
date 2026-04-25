#!/bin/bash
set -euo pipefail

# Container setup script for codex
# Enable this by putting the following line
# in your codex env setup script web box
#
# ./agent_scripts/codex_setup_container.sh
#

pyenv global 3.11
apt -y install pandoc

ARTIFACT_REPO="lincc-frameworks/hyrax"
ARTIFACT_NAME="hyrax-agent-conda-env-main"
ENV_DIR="$HOME/hyrax-venv"
AUTH_HEADER=(-H "Authorization: Bearer ${GITHUB_TOKEN:?GITHUB_TOKEN must be set}")

ARTIFACT_ID=$(curl -fsSL "${AUTH_HEADER[@]}" "https://api.github.com/repos/${ARTIFACT_REPO}/actions/artifacts?per_page=100" | jq -r ".artifacts[] | select(.name == \"${ARTIFACT_NAME}\") | .id" | head -n1)
curl -fsSL "${AUTH_HEADER[@]}" "https://api.github.com/repos/${ARTIFACT_REPO}/actions/artifacts/${ARTIFACT_ID}/zip" -o /tmp/hyrax-agent-conda-env.zip

rm -rf "$ENV_DIR"
mkdir -p "$ENV_DIR"
rm -rf /tmp/hyrax-agent-conda-env
unzip -q /tmp/hyrax-agent-conda-env.zip -d /tmp/hyrax-agent-conda-env
TARBALL_PATH=$(find /tmp/hyrax-agent-conda-env -name '*.tar.gz' | head -n1)
tar -xzf "$TARBALL_PATH" -C "$ENV_DIR"

# shellcheck disable=SC1091
set +u
source "${ENV_DIR}/bin/activate"
set -u
conda-unpack

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"

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
  cat >> "$PROFILE_FILE" <<EOF
$BEGIN_MARKER
if [ -f "$HOME/hyrax-venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$HOME/hyrax-venv/bin/activate"
fi
$END_MARKER
EOF
fi
