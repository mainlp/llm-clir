#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

set -a
source $SCRIPT_DIR/../../.env
set +a

git clone -b tevatron-v1 https://github.com/texttron/tevatron.git
cd tevatron && pip install -e .
cd examples/repllama

rm -f ./data.py
cp "$PROJECT_ROOT/retrieval/repllama/data.py" ./data.py
cp "$PROJECT_ROOT/retrieval/repllama/encode_and_rank.sh" ./encode_and_rank.sh
pip install -r $PROJECT_ROOT/retrieval/requirements_repllama.txt

chmod +x ./encode_and_rank.sh
cmd='sed -i "s|PROJECT_ROOT=.*|PROJECT_ROOT=\"$PROJECT_ROOT\"|g" ./encode_and_rank.sh'
echo "Updating encode_and_rank.sh in repllama folder ($(pwd))"
echo $cmd
eval $cmd

echo "Setup complete."
