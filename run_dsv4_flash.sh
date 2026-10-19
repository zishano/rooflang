DOCKER_CONFIG="$TMP_DOCKER_CONFIG" \
bash agents/find_pareto_frontier/run-agent.sh \
  --agent codex \
  --agent-model 'gpt-6-astra' \
  --model dsv4_flash \
  --preset b300 \
  --results "$PWD/dsv4_flash_pareto" \
  --env-file "$RUNTIME_ENV" \
  --build
