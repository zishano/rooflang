# DCP/EP topology search agent

The task prompt is `AGENTS.md`. This agent has its own container image, run
directory and results input. It does not replace or modify the existing
`layer_selective_dcp` or `dcp_arch` agents.

Configure credentials without committing them:

```bash
cp agents/dcp_topology_search/agent.env.example agents/dcp_topology_search/agent.env
# Set DEEPSEEK_API_KEY in agent.env using your editor.
```

Run the bounded search:

```bash
./run_dsv4_flash_dcp_topology_agent.sh
```

The default read-only historical input is
`results/dsv4_flash_h200_64gpu_64k_1m/raw_results.jsonl`. To choose another
result directory or environment file:

```bash
ROOFLANG_RESULTS=/path/to/results \
RUNTIME_ENV=/path/to/agent.env \
./run_dsv4_flash_dcp_topology_agent.sh
```

Each launch creates an isolated source copy and retains its trace and output
under `agents/dcp_topology_search/runs/`. Only the optimizer, DCP graph/operator
files, experiment code and program tests are writable in that copy.
