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
#
# We use a venv because claude code's current container (Apr 2026)
# Doesn't allow us to install all of hyrax's dependencies

apt -y install pandoc

ARTIFACT_REPO="${HYRAX_REPO:-lincc-frameworks/hyrax}"
ARTIFACT_NAME="${HYRAX_ENV_ARTIFACT_NAME:-hyrax-agent-venv-main}"
VENV_DIR="${HYRAX_VENV_DIR:-$HOME/hyrax-venv}"

ARTIFACT_ID=$(curl -fsSL "https://api.github.com/repos/${ARTIFACT_REPO}/actions/artifacts?per_page=100" | jq -r ".artifacts[] | select(.name == \"${ARTIFACT_NAME}\") | .id" | head -n1)
curl -fsSL "https://api.github.com/repos/${ARTIFACT_REPO}/actions/artifacts/${ARTIFACT_ID}/zip" -o /tmp/hyrax-agent-venv.zip

rm -rf "$VENV_DIR"
mkdir -p "$VENV_DIR"
rm -rf /tmp/hyrax-agent-venv
unzip -q /tmp/hyrax-agent-venv.zip -d /tmp/hyrax-agent-venv
TARBALL_PATH=$(find /tmp/hyrax-agent-venv -name '*.tar.gz' | head -n1)
tar -xzf "$TARBALL_PATH" -C "$VENV_DIR"

echo "activating venv..."
source "$VENV_DIR/bin/activate"

echo "installing hyrax..."
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"
pip install -e '.[dev]'
