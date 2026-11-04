#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -n "${TMP_DOCKER_CONFIG:-}" ]]; then
  export DOCKER_CONFIG="$TMP_DOCKER_CONFIG"
fi

bash "$script_dir/agents/layer_selective_dcp/run-agent.sh" \
  --agent codex \
  --agent-model deepseek-flash \
  --model dsv4_flash \
  --preset h200 \
  --results "$script_dir/results/dsv4_flash_pareto" \
  --env-file "${RUNTIME_ENV:-$script_dir/agents/layer_selective_dcp/agent.env}" \
  --build "$@"
