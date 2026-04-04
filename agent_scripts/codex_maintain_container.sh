#!/bin/bash

# To use this as your maintenance script in codex
# put the following line into the maintenence script
# setup box on the webpage:
#
# ./agent_scripts/codex_maintain_container.sh
#

pip install -e .'[dev]'
pushd docs
make clean
popd

