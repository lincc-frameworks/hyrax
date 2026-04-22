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

ARTIFACT_REPO="${HYRAX_REPO:-lincc-frameworks/hyrax}"
ARTIFACT_NAME="${HYRAX_ENV_ARTIFACT_NAME:-hyrax-agent-venv-main}"
ARTIFACT_ID="${HYRAX_ENV_ARTIFACT_ID:-}"
VENV_DIR="${HYRAX_VENV_DIR:-$HOME/hyrax-venv}"
GITHUB_TOKEN_VALUE="${HYRAX_GITHUB_TOKEN:-${GITHUB_TOKEN:-}}"
AUTH_HEADER=()

if [ -n "$GITHUB_TOKEN_VALUE" ]; then
    AUTH_HEADER=(-H "Authorization: Bearer $GITHUB_TOKEN_VALUE")
fi

if [ -z "$ARTIFACT_ID" ]; then
    ARTIFACT_ID=$(curl -fsSL "${AUTH_HEADER[@]}" "https://api.github.com/repos/${ARTIFACT_REPO}/actions/artifacts?per_page=100" | jq -r ".artifacts[] | select(.name == \"${ARTIFACT_NAME}\") | .id" | head -n1)
fi

curl -fsSL "${AUTH_HEADER[@]}" "https://api.github.com/repos/${ARTIFACT_REPO}/actions/artifacts/${ARTIFACT_ID}/zip" -o /tmp/hyrax-agent-venv.zip

rm -rf "$VENV_DIR"
mkdir -p "$VENV_DIR"
rm -rf /tmp/hyrax-agent-venv
unzip -q /tmp/hyrax-agent-venv.zip -d /tmp/hyrax-agent-venv
TARBALL_PATH=$(find /tmp/hyrax-agent-venv -name '*.tar.gz' | head -n1)
tar -xzf "$TARBALL_PATH" -C "$VENV_DIR"

source "$VENV_DIR/bin/activate"
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"
pip install -e '.[dev]'
