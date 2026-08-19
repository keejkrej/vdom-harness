#!/usr/bin/env bash
# Install official tau2 (https://github.com/sierra-research/tau2-bench) editable
# so `python -m tau2_vdom --domain mock --smoke` can run a real mock-domain eval.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${TAU2_SRC:-$ROOT/.tau2-bench}"

if [ ! -d "$DEST/.git" ]; then
  if [ -d /tmp/tau2-bench/.git ]; then
    echo "copying /tmp/tau2-bench → $DEST"
    cp -a /tmp/tau2-bench "$DEST"
  else
    git clone --depth 1 https://github.com/sierra-research/tau2-bench.git "$DEST"
  fi
fi

python3 -m pip install -e "$DEST"
export TAU2_DATA_DIR="$DEST/data"
echo ""
echo "tau2 installed from $DEST"
echo "export TAU2_DATA_DIR=$DEST/data"
echo "Then: PYTHONPATH=python python3 -m tau2_vdom --domain mock --smoke"
