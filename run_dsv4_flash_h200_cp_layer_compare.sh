#!/usr/bin/env bash
# Compare single-layer decode timing at CP=1/2/4 with fixed global batch,
# context, 64 H200 GPUs, EP=8 and PP=8. Runs both the 64K baseline and 1M.
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

base_dir="${OUTPUT_DIR:-$script_dir/results/dsv4_flash_h200_cp_layer_compare}"
contexts=(65536 1048576)
pp_partition=(6 6 6 5 5 5 5 5) # 43 layers over 8 PP stages
batch_size=64
ep=8
layer=2 # compression ratio 4; layer 0 is capped by the 128-token window

for context in "${contexts[@]}"; do
  for cp in 1 2 4; do
    dp=$((ep / cp))
    out="$base_dir/layer-${layer}/context-${context}/cp-${cp}"
    printf '\n=== context=%s batch=%s CP=%s DP=%s EP=%s PP=8 ===\n' \
      "$context" "$batch_size" "$cp" "$dp" "$ep"
    "$python_bin" "$script_dir/analysis/single_layer_timing.py" \
      --stage decode \
      --model dsv4_flash \
      --hardware H200Cluster1Node \
      --nodes 8 \
      --batch-size "$batch_size" \
      --context "$context" \
      --cp "$cp" --dp "$dp" --ep "$ep" \
      --pp-partition "${pp_partition[@]}" \
      --gpu-index 0 \
      --layer "$layer" \
      --output-dir "$out" \
      "$@"
  done
done

"$python_bin" "$script_dir/analysis/summarize_cp_layer_compare.py" \
  --root "$base_dir" --layer "$layer"
printf '\nAll layer timing runs are under: %s\n' "$base_dir"
