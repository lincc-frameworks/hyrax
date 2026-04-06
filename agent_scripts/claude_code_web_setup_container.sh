#!/bin/bash

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

echo "creating venv in $HOME/hyrax-venv"
python -m venv $HOME/hyrax-venv

echo "activating venv..."
source $HOME/hyrax-venv/bin/activate

echo "installing hyrax..."
cd hyrax
pip install -e '.[dev]'

