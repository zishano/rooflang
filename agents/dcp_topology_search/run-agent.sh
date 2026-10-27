#!/usr/bin/env bash
# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
default_rooflang_dir=$(cd -- "$script_dir/../.." && pwd)

agent=${ROOFLANG_AGENT:-}
agent_model=${ROOFLANG_AGENT_MODEL:-}
rooflang_dir=${ROOFLANG_ROOT:-$default_rooflang_dir}
model=${ROOFLANG_MODEL:-}
preset=${ROOFLANG_PRESET:-}
default_results=$script_dir/../../results/dsv4_flash_h200_64gpu_64k_1m
results_path=${ROOFLANG_RESULTS:-$default_results}
image=${ROOFLANG_AGENT_IMAGE:-rooflang-dcp-topology-agent:latest}
env_file=${ROOFLANG_AGENT_ENV_FILE:-}
runs_root=${ROOFLANG_RUNS_ROOT:-$script_dir/runs}
build_image=false

usage() {
    cat <<'EOF'
Usage:
  run-agent.sh --agent codex|claude --model NAME --preset NAME \
      [--results PATH] [options]

Required:
  --agent NAME          Coding-agent frontend: codex or claude.
  --model NAME          Rooflang model package name.
  --preset NAME         Rooflang hardware preset/module name.
  --results PATH        Prior result JSONL/directory (default: 64-GPU H200 baseline).

Options:
  --rooflang PATH       Input Rooflang root (default: two levels above script).
  --runs-root PATH      Parent for isolated retained runs (default: ./runs).
  --agent-model NAME    Optional model override passed to the coding-agent CLI.
  --image NAME          Docker image tag (default: rooflang-dcp-topology-agent:latest).
  --env-file PATH       Explicit Docker env file for credentials/provider config.
  --build               Rebuild the Docker image before running.
  -h, --help            Show this help.

The corresponding ROOFLANG_* environment variables may be used instead of
flags. Command-line flags take precedence. Host CLI configuration and proxy
variables are not inherited; place every required value explicitly in the env
file. Every invocation creates and retains a private Rooflang copy and trace.
EOF
}

die() {
    echo "error: $*" >&2
    exit 2
}

while (($#)); do
    case "$1" in
        --agent)
            (($# >= 2)) || die "--agent requires a value"
            agent=$2
            shift 2
            ;;
        --agent-model)
            (($# >= 2)) || die "--agent-model requires a value"
            agent_model=$2
            shift 2
            ;;
        --rooflang)
            (($# >= 2)) || die "--rooflang requires a value"
            rooflang_dir=$2
            shift 2
            ;;
        --runs-root)
            (($# >= 2)) || die "--runs-root requires a value"
            runs_root=$2
            shift 2
            ;;
        --model)
            (($# >= 2)) || die "--model requires a value"
            model=$2
            shift 2
            ;;
        --preset)
            (($# >= 2)) || die "--preset requires a value"
            preset=$2
            shift 2
            ;;
        --results)
            (($# >= 2)) || die "--results requires a value"
            results_path=$2
            shift 2
            ;;
        --image)
            (($# >= 2)) || die "--image requires a value"
            image=$2
            shift 2
            ;;
        --env-file)
            (($# >= 2)) || die "--env-file requires a value"
            env_file=$2
            shift 2
            ;;
        --build)
            build_image=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

command -v docker >/dev/null 2>&1 || die "docker is not installed"
command -v rsync >/dev/null 2>&1 || die "rsync is not installed"

[[ "$agent" == codex || "$agent" == claude ]] \
    || die "--agent must be 'codex' or 'claude'"
[[ "$model" =~ ^[a-zA-Z0-9_]+$ ]] \
    || die "--model must contain only letters, digits, and underscores"
[[ "$preset" =~ ^[a-zA-Z0-9_]+$ ]] \
    || die "--preset must contain only letters, digits, and underscores"
[[ -d "$rooflang_dir" ]] || die "Rooflang directory does not exist: $rooflang_dir"
[[ -e "$results_path" ]] || die "results path does not exist: $results_path"

rooflang_dir=$(realpath -e -- "$rooflang_dir")
if [[ -d "$results_path" ]]; then
    results_path=$(realpath -e -- "$results_path")
    [[ -f "$results_path/raw_results.jsonl" ]] \
        || die "results directory has no raw_results.jsonl: $results_path"
    results_target=/workspace/results
else
    results_path=$(realpath -e -- "$results_path")
    [[ "$results_path" == *.jsonl ]] \
        || die "a results file must be JSONL (expected a .jsonl suffix)"
    results_target=/workspace/results/raw_results.jsonl
fi

target_rel=programs/models/$model/optimization.py
experiments_rel=programs/experiments
target_file=$rooflang_dir/$target_rel
experiments_dir=$rooflang_dir/$experiments_rel
preset_file=$rooflang_dir/programs/presets/$preset.py
searcher_file=$experiments_dir/find_pareto_frontier.py

[[ -f "$target_file" ]] || die "model optimization file does not exist: $target_file"
[[ -d "$experiments_dir" ]] || die "experiment directory does not exist: $experiments_dir"
[[ -f "$preset_file" ]] || die "preset module does not exist: $preset_file"
[[ -f "$searcher_file" ]] || die "find_pareto_frontier.py does not exist: $searcher_file"
[[ -f "$rooflang_dir/requirements.txt" ]] \
    || die "Rooflang requirements.txt does not exist: $rooflang_dir/requirements.txt"

if [[ -n "$env_file" ]]; then
    [[ -f "$env_file" ]] || die "env file does not exist: $env_file"
    env_parent=$(cd -- "$(dirname -- "$env_file")" && pwd)
    env_file=$env_parent/$(basename -- "$env_file")
fi

build_agent_image() (
    build_context=$(mktemp -d /tmp/rooflang-agent-build.XXXXXX)
    trap 'rm -rf -- "$build_context"' EXIT
    cp -- "$script_dir/Dockerfile" "$build_context/Dockerfile"
    cp -- "$script_dir/entrypoint.sh" "$build_context/entrypoint.sh"
    cp -- "$script_dir/deepseek-models.json" "$build_context/deepseek-models.json"
    cp -- "$rooflang_dir/requirements.txt" "$build_context/requirements.txt"
    # Docker's runtime --env-file does not configure build networking.
    # Forward explicit shell proxy settings, normalizing case for pip/curl.
    # Name-only build args keep proxy credentials out of command arguments.
    build_proxy_args=()
    for proxy_name in http_proxy https_proxy all_proxy no_proxy; do
        proxy_upper=${proxy_name^^}
        if [[ -v "$proxy_name" || -v "$proxy_upper" ]]; then
            proxy_value=${!proxy_name-${!proxy_upper-}}
            export "$proxy_name=$proxy_value" "$proxy_upper=$proxy_value"
            build_proxy_args+=(--build-arg "$proxy_name" --build-arg "$proxy_upper")
        fi
    done
    docker build \
        "${build_proxy_args[@]}" \
        --file "$build_context/Dockerfile" \
        --tag "$image" \
        "$build_context"
)

if [[ "$build_image" == true ]] || ! docker image inspect "$image" >/dev/null 2>&1; then
    build_agent_image
fi

mkdir -p -- "$runs_root"
runs_root=$(realpath -e -- "$runs_root")
default_runs_root=$(realpath -m -- "$script_dir/runs")

rsync_args=(
    -a
    --exclude AGENTS.md
    --exclude AGENTS.override.md
    --exclude CLAUDE.md
    --exclude .agents/
    --exclude .claude/
    --exclude .codex/
    --exclude .git/
    --exclude .mcp.json
)
if [[ "$runs_root" == "$rooflang_dir" ]]; then
    die "--runs-root may not be the Rooflang root"
elif [[ "$runs_root" == "$rooflang_dir/"* ]]; then
    [[ "$runs_root" == "$default_runs_root" ]] \
        || die "a runs root inside Rooflang is allowed only at $default_runs_root"
    runs_rel=${runs_root#"$rooflang_dir/"}
    rsync_args+=(--exclude "/$runs_rel/")
fi

[[ "$rooflang_dir" != *','* \
    && "$results_path" != *','* \
    && "$runs_root" != *','* ]] \
    || die "Docker bind-mount source paths may not contain commas"

run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir=$(mktemp -d "$runs_root/run-${run_stamp}-XXXXXX")
run_rooflang=$run_dir/rooflang
run_task_dir=$run_dir/task
run_artifacts_dir=$run_dir/artifacts
trace_log=$run_dir/agent.trace.log

mkdir -p -- "$run_task_dir" "$run_artifacts_dir"
rsync "${rsync_args[@]}" "$rooflang_dir/" "$run_rooflang/"

run_target_file=$run_rooflang/$target_rel
run_experiments_dir=$run_rooflang/$experiments_rel
model_dir_rel=programs/models/$model
model_dir=$run_rooflang/$model_dir_rel
tests_rel=tests/unit_tests/programs
tests_dir=$run_rooflang/$tests_rel
[[ -f "$run_target_file" ]] || die "isolated copy is missing $target_rel"
[[ -d "$run_experiments_dir" ]] || die "isolated copy is missing $experiments_rel"
[[ -d "$model_dir" ]] || die "isolated copy is missing $model_dir_rel"
[[ -d "$tests_dir" ]] || die "isolated copy is missing $tests_rel"
chmod u+rw -- "$run_target_file"
chmod -R u+rwX -- "$run_experiments_dir"
chmod -R u+rwX -- "$tests_dir"

# Only the DSV4 Flash DCP implementation files are writable within the
# otherwise read-only model tree. The model definition and config stay read-only.
dcp_files=(dcp_topology.py dcp_graph.py dcp_ops.py dcp_validation.py dcp_reuse.py)
for dcp_file in "${dcp_files[@]}"; do
    source_file=$model_dir/$dcp_file
    [[ -f "$source_file" ]] || die "isolated copy is missing $model_dir_rel/$dcp_file"
    chmod u+rw -- "$source_file"
done

sed \
    -e "s/{{MODEL}}/$model/g" \
    -e "s/{{PRESET}}/$preset/g" \
    "$script_dir/AGENTS.md" > "$run_task_dir/AGENTS.md"
chmod 0444 "$run_task_dir/AGENTS.md"

docker_args=(
    run
    --rm
    --init
    --network host
    --read-only
    --cap-drop ALL
    --security-opt no-new-privileges
    --user "$(id -u):$(id -g)"
    --tmpfs "/tmp:rw,exec,nosuid,nodev,uid=$(id -u),gid=$(id -g),mode=1777"
    --tmpfs "/home/agent:rw,exec,nosuid,nodev,uid=$(id -u),gid=$(id -g),mode=0700"
)

if [[ -n "$env_file" ]]; then
    docker_args+=(--env-file "$env_file")
fi

# These values deliberately follow --env-file so session locations cannot be
# redirected to host or shared state by provider configuration.
docker_args+=(
    --env "ROOFLANG_AGENT=$agent"
    --env "ROOFLANG_AGENT_MODEL=$agent_model"
    --env "HOME=/home/agent"
    --env "CODEX_HOME=/home/agent/.codex"
    --env "CLAUDE_CONFIG_DIR=/home/agent/.claude"
    --env "XDG_CONFIG_HOME=/home/agent/.config"
    --env "XDG_DATA_HOME=/home/agent/.local/share"
    --env "XDG_STATE_HOME=/home/agent/.local/state"
    --env "XDG_CACHE_HOME=/tmp/cache"
    --env "ROOFLANG_ARTIFACTS=/workspace/artifacts"
    --mount "type=bind,source=$run_artifacts_dir,target=/workspace/artifacts"
    --mount "type=bind,source=$run_task_dir/AGENTS.md,target=/workspace/task/AGENTS.md,readonly"
    --mount "type=bind,source=$run_rooflang,target=/workspace/rooflang,readonly"
    --mount "type=bind,source=$run_target_file,target=/workspace/rooflang/$target_rel"
    --mount "type=bind,source=$run_experiments_dir,target=/workspace/rooflang/$experiments_rel"
    --mount "type=bind,source=$tests_dir,target=/workspace/rooflang/$tests_rel"
    --mount "type=bind,source=$results_path,target=$results_target,readonly"
)

for dcp_file in "${dcp_files[@]}"; do
    docker_args+=(--mount "type=bind,source=$model_dir/$dcp_file,target=/workspace/rooflang/$model_dir_rel/$dcp_file")
done

{
    echo "Agent:             $agent"
    echo "Input Rooflang:    $rooflang_dir"
    echo "Isolated Rooflang: $run_rooflang"
    echo "Model:             $model"
    echo "Preset:            $preset"
    echo "Results:           $results_path"
    echo "Writable model:    $run_target_file"
    echo "Writable search:   $run_experiments_dir"
    printf 'Writable DCP:      %s\n' "${dcp_files[@]/#/$model_dir/}"
    echo "Writable tests:    $tests_dir"
    echo "Run artifacts:     $run_artifacts_dir"
    echo "Docker image:      $image"
    echo "Run directory:     $run_dir"
    echo "Trace log:         $trace_log"
} | tee "$trace_log"

set +e
docker "${docker_args[@]}" "$image" 2>&1 | tee -a "$trace_log"
docker_status=${PIPESTATUS[0]}
set -e

echo "Agent exit status: $docker_status" | tee -a "$trace_log"
echo "Retained run:      $run_dir" | tee -a "$trace_log"
exit "$docker_status"
