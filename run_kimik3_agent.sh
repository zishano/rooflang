#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -n "${TMP_DOCKER_CONFIG:-}" ]]; then
  export DOCKER_CONFIG="$TMP_DOCKER_CONFIG"
fi

bash agents/find_pareto_frontier/run-agent.sh \
  --agent codex \
  --agent-model 'gpt-6.1-sol' \
  --model kimi_k3 \
  --preset gb300 \
  --results "$script_dir/results/kimi_k3_gb300_8gpu_decode" \
  --env-file "${RUNTIME_ENV:-$script_dir/agents/find_pareto_frontier/agent.env}" \
  --build
