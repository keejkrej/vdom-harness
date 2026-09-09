#!/bin/bash
set -euo pipefail
cd /workspace/vdom-pr21
export TAU2_DATA_DIR=/workspace/vdom-harness/.tau2-bench/data
export PYTHONPATH=/workspace/vdom-pr21/python
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export OPENAI_MODEL=deepseek/deepseek-v4-flash-0731
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "live key missing; not writing pendingKey JSON" >&2
  exit 2
fi
if [ -n "${OPENROUTER_API_KEY:-}" ]; then
  export OPENAI_API_KEY="$OPENROUTER_API_KEY"
fi
echo "[34-runner] start=$(date -u +%Y-%m-%dT%H:%M:%SZ) HEAD=$(git rev-parse --short HEAD)"
exec /workspace/vdom-harness/.tau2-venv/bin/python -m tau2_vdom.improve --live-hang-obs-isku 7
