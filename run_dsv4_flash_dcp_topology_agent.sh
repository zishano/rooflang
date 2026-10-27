#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
env_file=${RUNTIME_ENV:-$script_dir/agents/dcp_topology_search/agent.env}

if [[ ! -f "$env_file" ]]; then
  echo "Missing agent env file: $env_file" >&2
  echo "Copy agents/dcp_topology_search/agent.env.example to that path and set DEEPSEEK_API_KEY." >&2
  exit 2
fi

exec bash "$script_dir/agents/dcp_topology_search/run-agent.sh" \
  --agent codex \
  --agent-model gpt-5.6-sol \
  --model dsv4_flash \
  --preset h200 \
  --env-file "$env_file" \
  --build "$@"
