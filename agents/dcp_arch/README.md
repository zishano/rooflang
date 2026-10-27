# Rooflang DCP and Graph Optimization Agent Launcher

Ubuntu 24.04 launcher for the English task in `AGENTS.md`, supporting Codex and Claude Code.

## Run

```bash
cp agent.env.example /secure/path/agent.env
./run-agent.sh \
  --agent codex \
  --model dsv4_pro \
  --preset b300 \
  --results ../../../dsv4_pro_pareto_frontier \
  --env-file /secure/path/agent.env \
  --build
```

Use `--agent claude` for Claude Code, `--agent-model` for a model override, and `--runs-root` to change the retained-run parent. Equivalent `ROOFLANG_*` environment variables are supported; see `./run-agent.sh --help`.

## Isolation and Output

Every invocation creates `runs/run-<timestamp>-<random>/` containing the rendered task, an independent Rooflang copy, persistent `artifacts/`, and `agent.trace.log`. Runs are retained and safe to execute in parallel; the input Rooflang tree is untouched.

Writable paths in the private copy: `programs/models/<model>/optimization.py`, `programs/experiments/`, `language/optimization/`, `language/kernels/`, `language/placement.py`, `language/graph.py`, `language/tensor.py`, `runtime/`, and `tests/`. Changes remain subject to the semantic and accounting constraints in `AGENTS.md`. New scripts belong in `programs/experiments/`; reports belong in `artifacts/`. Prior results, task, model definitions, hardware, presets, remaining source, and container root are read-only. Provider instruction/config files and `.git` are excluded from the copy.

The exact rendered `AGENTS.md` content is the sole prompt. `/workspace/artifacts` is a private writable host mount; HOME/CLI/XDG state and the work directory remain fresh tmpfs. Host CLI config, auth directories, and proxy settings are not inherited.

Put credentials, provider endpoints, and proxies explicitly in `--env-file`. Codex and Claude are installed with their official native `install.sh` scripts and the selected CLI updates at every run; Node/npm are not installed.

Codex emits JSONL events and Claude emits verbose `stream-json`. CLI updates, versions, stdout, stderr, metadata, and exit status are all streamed to the terminal and saved in `agent.trace.log`.
