#!/bin/bash
set -euo pipefail

# To use this script in claude code
# edit your environment and put the following
# as your setup script:
#
# #!/bin/bash
# ./hyrax/agent_scripts/claude_code_web_setup_container.sh
#
# This works together with a session start hook to place
# claude code in the created venv

apt -y install pandoc

ARTIFACT_REPO="lincc-frameworks/hyrax"
ARTIFACT_NAME="hyrax-agent-conda-env-main"
VENV_DIR="$HOME/hyrax-venv"
AUTH_HEADER=(-H "Authorization: Bearer $GITHUB_TOKEN")

ARTIFACT_ID=$(curl -fsSL "${AUTH_HEADER[@]}" "https://api.github.com/repos/${ARTIFACT_REPO}/actions/artifacts?per_page=100" | jq -r ".artifacts[] | select(.name == \"${ARTIFACT_NAME}\") | .id" | head -n1)
curl -fsSL "${AUTH_HEADER[@]}" "https://api.github.com/repos/${ARTIFACT_REPO}/actions/artifacts/${ARTIFACT_ID}/zip" -o /tmp/hyrax-agent-conda-env.zip

rm -rf "$VENV_DIR"
mkdir -p "$VENV_DIR"
rm -rf /tmp/hyrax-agent-conda-env
unzip -q /tmp/hyrax-agent-conda-env.zip -d /tmp/hyrax-agent-conda-env
TARBALL_PATH=$(find /tmp/hyrax-agent-conda-env -name '*.tar.gz' | head -n1)
tar -xzf "$TARBALL_PATH" -C "$VENV_DIR"

echo "activating env..."
source "$VENV_DIR/bin/activate"
conda-unpack

echo "installing hyrax..."
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"
pip install -e '.[dev]'
