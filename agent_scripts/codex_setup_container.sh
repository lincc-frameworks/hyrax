#!/bin/bash

# Container setup script for codex
# Enable this by putting the following line
# in your codex env setup script web box
#
# ./agent_scripts/codex_setup_container.sh
# 

pyenv global 3.11
pip install --upgrade pip
apt -y install pandoc
pip install -e .'[dev]'

