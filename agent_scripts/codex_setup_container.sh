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
AUTH_HEADER=(-H "Authorization: Bearer $GITHUB_TOKEN")

ARTIFACT_ID=$(curl -fsSL "${AUTH_HEADER[@]}" "https://api.github.com/repos/${ARTIFACT_REPO}/actions/artifacts?per_page=100" | jq -r ".artifacts[] | select(.name == \"${ARTIFACT_NAME}\") | .id" | head -n1)
curl -fsSL "${AUTH_HEADER[@]}" "https://api.github.com/repos/${ARTIFACT_REPO}/actions/artifacts/${ARTIFACT_ID}/zip" -o /tmp/hyrax-agent-conda-env.zip

rm -rf "$ENV_DIR"
mkdir -p "$ENV_DIR"
rm -rf /tmp/hyrax-agent-conda-env
unzip -q /tmp/hyrax-agent-conda-env.zip -d /tmp/hyrax-agent-conda-env
TARBALL_PATH=$(find /tmp/hyrax-agent-conda-env -name '*.tar.gz' | head -n1)
tar -xzf "$TARBALL_PATH" -C "$ENV_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
conda-unpack

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"
pip install -e '.[dev]'
