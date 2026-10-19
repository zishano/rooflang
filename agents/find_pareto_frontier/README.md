# Rooflang Pareto Agent Launcher

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

### Build networking without changing system configuration

Use `--base-image REGISTRY/PATH/ubuntu:24.04` (or `ROOFLANG_BASE_IMAGE`)
to select an Ubuntu 24.04 image from a registry reachable by the Docker daemon.
The default remains `ubuntu:24.04`. An explicit alternative registry avoids
Docker Hub mirror selection; it must contain the image and be reachable without
depending on shell proxy variables being inherited by the daemon.

Exported host `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, `NO_PROXY` and lowercase
equivalents are forwarded as predefined Docker build arguments for `RUN` steps.
They do not configure the daemon's base-image downloads. Build credentials and
API keys are not read from `--env-file`; that file is only for the running agent.
Runtime proxies must still be included explicitly in that file.

## Isolation and Output

Every invocation creates `runs/run-<timestamp>-<random>/` containing the rendered task, an independent Rooflang copy, persistent `artifacts/`, and `agent.trace.log`. Runs are retained and safe to execute in parallel; the input Rooflang tree is untouched.

Only the selected `programs/models/<model>/optimization.py` and copied `programs/experiments/` are writable. Prior results, task, remaining source, and container root are read-only. Provider instruction/config files and `.git` are excluded from the copy.

The exact rendered `AGENTS.md` content is the sole prompt. `/workspace/artifacts` is a private writable host mount; HOME/CLI/XDG state and the work directory remain fresh tmpfs. Host CLI config, auth directories, and proxy settings are not inherited.

Put credentials, provider endpoints, and proxies explicitly in `--env-file`. Codex and Claude are installed with their official native `install.sh` scripts and the selected CLI updates at every run; Node/npm are not installed.

Codex emits JSONL events and Claude emits verbose `stream-json`. CLI updates, versions, stdout, stderr, metadata, and exit status are all streamed to the terminal and saved in `agent.trace.log`.
