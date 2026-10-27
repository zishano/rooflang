#!/usr/bin/env bash
# Compare the selected decode-layer window for all valid CP/DP factorizations
# with 64 H200 GPUs, EP=64, PP=1, and a fixed global batch of 64. The analyzer
# truncates the graph after the target layer; CP1 is cross-checked against the
# full-model analyzer before using the faster layer slice.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$script_dir"
export PYTHONPATH="$(dirname -- "$script_dir")${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$script_dir/.cache/matplotlib"
export MPLBACKEND=Agg

python_bin="${ROOFLANG_PYTHON:-$script_dir/.venv/bin/python}"
if [[ ! -x "$python_bin" && -z "${ROOFLANG_PYTHON:-}" ]]; then
  python_bin="/mnt/nvme1n1/data/lmk/PROJECT/rooflang/.venv/bin/python"
fi
if [[ ! -x "$python_bin" ]]; then
  printf 'Python 不可执行：%s；请设置 ROOFLANG_PYTHON。\n' "$python_bin" >&2
  exit 1
fi

base_dir="${OUTPUT_DIR:-$script_dir/results/dsv4_flash_h200_ep64_cp_layer_compare/isolated-layer}"
contexts=(65536 1048576)
cp_values=(1 2 4 8 16 32 64)
batch_size=64
ep=64
layer=2 # 0-based layer 2 = third model layer; compression ratio 4

for context in "${contexts[@]}"; do
  for cp in "${cp_values[@]}"; do
    dp=$((ep / cp))
    out="$base_dir/layer-${layer}/context-${context}/cp-${cp}-dp-${dp}"
    printf '\n=== context=%s batch=%s CP=%s DP=%s EP=%s PP=1 ===\n' \
      "$context" "$batch_size" "$cp" "$dp" "$ep"
    "$python_bin" "$script_dir/analysis/single_layer_timing_truncated.py" \
      --stage decode \
      --model dsv4_flash \
      --hardware H200Cluster1Node \
      --nodes 8 \
      --batch-size "$batch_size" \
      --context "$context" \
      --cp "$cp" --dp "$dp" --ep "$ep" \
      --pp-partition 3 \
      --gpu-index 0 \
      --layer "$layer" \
      --output-dir "$out"
  done
done

printf '\nRun artifacts are under: %s\n' "$base_dir"
