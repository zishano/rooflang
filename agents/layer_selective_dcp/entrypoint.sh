#!/usr/bin/env bash
# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

set -euo pipefail

readonly task_file=/workspace/task/AGENTS.md

if [[ ! -r "$task_file" ]]; then
    echo "error: $task_file is not readable" >&2
    exit 2
fi

if [[ ! -r /workspace/rooflang/programs/experiments/find_pareto_frontier.py ]]; then
    echo "error: /workspace/rooflang is not a Rooflang source tree" >&2
    exit 2
fi

if [[ ! -d /workspace/artifacts || ! -w /workspace/artifacts ]]; then
    echo "error: /workspace/artifacts is not a writable directory" >&2
    exit 2
fi

mkdir -p \
    "$HOME" \
    "$CODEX_HOME" \
    "$CLAUDE_CONFIG_DIR" \
    "$XDG_CONFIG_HOME" \
    "$XDG_DATA_HOME" \
    "$XDG_STATE_HOME" \
    "$HOME/.local/bin" \
    /tmp/cache \
    /tmp/matplotlib \
    /tmp/rooflang-pareto-agent

agent=${ROOFLANG_AGENT:-}
agent_model=${ROOFLANG_AGENT_MODEL:-}
cd /tmp/rooflang-pareto-agent

case "$agent" in
    codex)
        mkdir -p "$CODEX_HOME/packages"
        cp -R /opt/rooflang-agent/codex/standalone \
            "$CODEX_HOME/packages/standalone"
        ln -s "$CODEX_HOME/packages/standalone/current/bin/codex" \
            "$HOME/.local/bin/codex"
        echo "[agent-update] Running codex update" >&2
        codex update >&2
        hash -r
        codex --version >&2

        command=(
            codex exec
            --dangerously-bypass-approvals-and-sandbox
            --skip-git-repo-check
            --ignore-user-config
            --ignore-rules
            --ephemeral
            --json
            --color never
            --cd /tmp/rooflang-pareto-agent
            --add-dir /workspace/rooflang
            --add-dir /workspace/artifacts
        )
        if [[ -n "$agent_model" ]]; then
            command+=(--model "$agent_model")
        fi
        if [[ "${ROOFLANG_CODEX_PROVIDER:-}" == deepseek ]]; then
            : "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY in the agent env file}"
            : "${DEEPSEEK_BASE_URL:?Set DEEPSEEK_BASE_URL in the agent env file}"
            case "$DEEPSEEK_BASE_URL" in
                http://*|https://*) ;;
                *) echo "error: DEEPSEEK_BASE_URL must use http:// or https://" >&2; exit 2 ;;
            esac
            catalog=/opt/rooflang-agent/deepseek-models.json
            [[ -r "$catalog" ]] || {
                echo "error: DeepSeek model catalog missing; rebuild the image" >&2
                exit 2
            }
            jq -e --arg model "$agent_model" \
                '.models | any(.slug == $model)' "$catalog" >/dev/null || {
                echo "error: main model is missing from the DeepSeek catalog: $agent_model" >&2
                exit 2
            }
            command+=(
                --config 'model_provider="deepseek"'
                --config 'model_providers.deepseek.name="DeepSeek"'
                --config "model_providers.deepseek.base_url=$(jq -cn --arg url "$DEEPSEEK_BASE_URL" '$url')"
                --config 'model_providers.deepseek.env_key="DEEPSEEK_API_KEY"'
                --config 'model_providers.deepseek.wire_api="responses"'
                --config 'model_providers.deepseek.requires_openai_auth=false'
                --config 'model_providers.deepseek.supports_websockets=false'
                --config 'model_catalog_json="/opt/rooflang-agent/deepseek-models.json"'
                --config 'model_reasoning_effort="high"'
                --config 'web_search="disabled"'
            )
            echo "[provider] DeepSeek: URL from DEEPSEEK_BASE_URL (Responses HTTP), model=$agent_model" >&2
        elif [[ -n "${OPENAI_BASE_URL:-}" ]]; then
            [[ -n "${OPENAI_API_KEY:-}" ]] || {
                echo "error: OPENAI_API_KEY is required" >&2
                exit 2
            }

            command+=(
                --config 'model_provider="local_gateway"'
                --config 'model_providers.local_gateway.name="Local Gateway"'
                --config "model_providers.local_gateway.base_url=$(jq -cn --arg url "$OPENAI_BASE_URL" '$url')"
                --config 'model_providers.local_gateway.env_key="OPENAI_API_KEY"'
                --config 'model_providers.local_gateway.wire_api="responses"'
                --config 'model_providers.local_gateway.requires_openai_auth=false'
            )
        fi

        if [[ -n "${ROOFLANG_SUBAGENT_MODEL:-}" ]]; then
            command+=(
                --config 'agents.enabled=true'
                --config "agents.default_subagent_model=$(jq -cn --arg model "$ROOFLANG_SUBAGENT_MODEL" '$model')"
            )
            echo "[subagent] model=$ROOFLANG_SUBAGENT_MODEL" >&2
            if [[ "${ROOFLANG_CODEX_PROVIDER:-}" == deepseek ]] && ! jq -e \
                --arg model "$ROOFLANG_SUBAGENT_MODEL" \
                '.models | any(.slug == $model)' "$catalog" >/dev/null; then
                echo "warning: subagent model is absent from the current official catalog; verify legacy model availability or explicitly choose deepseek-flash" >&2
            fi
        fi

        exec "${command[@]}" - < "$task_file"
        ;;
    claude)
        mkdir -p "$XDG_DATA_HOME/claude"
        cp -R /opt/rooflang-agent/claude/versions \
            "$XDG_DATA_HOME/claude/versions"
        claude_seed=$(find "$XDG_DATA_HOME/claude/versions" \
            -mindepth 1 -maxdepth 1 -type f -print -quit)
        [[ -n "$claude_seed" ]] \
            || { echo "error: Claude CLI seed is missing" >&2; exit 2; }
        ln -s "$claude_seed" "$HOME/.local/bin/claude"
        echo "[agent-update] Running claude update" >&2
        claude update >&2
        hash -r
        claude --version >&2

        command=(
            claude
            --print
            --safe-mode
            --dangerously-skip-permissions
            --no-session-persistence
            --output-format stream-json
            --verbose
            --add-dir /workspace/rooflang
            --add-dir /workspace/artifacts
        )
        if [[ -n "$agent_model" ]]; then
            command+=(--model "$agent_model")
        fi
        exec "${command[@]}" < "$task_file"
        ;;
    *)
        echo "error: ROOFLANG_AGENT must be 'codex' or 'claude'" >&2
        exit 2
        ;;
esac
