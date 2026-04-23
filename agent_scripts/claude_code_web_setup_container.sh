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
# claude code in the created conda env

apt -y -v install pandoc

echo "INSTALLED PANDOC"

ARTIFACT_REPO="lincc-frameworks/hyrax"
ARTIFACT_NAME="hyrax-agent-conda-env-main"
ENV_DIR="$HOME/hyrax-venv"
AUTH_HEADER=(-H "Authorization: Bearer $GITHUB_TOKEN")

ARTIFACT_ID=$(curl -fsSL "${AUTH_HEADER[@]}" "https://api.github.com/repos/${ARTIFACT_REPO}/actions/artifacts?per_page=100" | jq -r ".artifacts[] | select(.name == \"${ARTIFACT_NAME}\") | .id" | head -n1)
# GitHub redirects to a signed Azure Blob URL; follow the redirect manually so the
# Bearer token is not forwarded to Azure (Azure rejects it with 503).
DOWNLOAD_URL=$(curl -s -o /dev/null -w '%{redirect_url}' "${AUTH_HEADER[@]}" "https://api.github.com/repos/${ARTIFACT_REPO}/actions/artifacts/${ARTIFACT_ID}/zip")
curl -fsSL "$DOWNLOAD_URL" -o /tmp/hyrax-agent-conda-env.zip

rm -rf "$ENV_DIR"
mkdir -p "$ENV_DIR"
rm -rf /tmp/hyrax-agent-conda-env
unzip -q /tmp/hyrax-agent-conda-env.zip -d /tmp/hyrax-agent-conda-env
TARBALL_PATH=$(find /tmp/hyrax-agent-conda-env -name '*.tar.gz' | head -n1)
tar -xzf "$TARBALL_PATH" -C "$ENV_DIR"

echo "activating env..."
# shellcheck disable=SC1091
set +u
source "$ENV_DIR/bin/activate"
set -u
conda-unpack

echo "installing hyrax..."
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"
python -m pip install -e '.[dev]'
