#!/usr/bin/env bash
# Pareto sweep for low-request-count, long-context DSV4 Flash decode.
# Uses the same searcher and metrics as run_dsv4_flash_h200_pareto.sh,
# with explicit contexts and batch multipliers to keep the sweep bounded.
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

read -r -a workloads <<< "${WORKLOADS:-decode-64k decode-256k decode-1m}"
read -r -a batch_multipliers <<< "${BATCH_MULTIPLIERS:-1 2 4}"
read -r -a pp_degrees <<< "${PP_DEGREES:-8 16 32}"
output_dir="${OUTPUT_DIR:-$script_dir/results/dsv4_flash_h200_longctx_pareto_pp8_16_32}"

exec "$python_bin" -m rooflang.programs.experiments.find_pareto_frontier \
  --model dsv4_flash \
  --workloads "${workloads[@]}" \
  --hardware h200 \
  --gpu-counts 64 \
  --pp-degrees "${pp_degrees[@]}" \
  --batch-multipliers "${batch_multipliers[@]}" \
  --workers "${WORKERS:-2}" \
  --output-dir "$output_dir" \
  --point-labels \
  "$@"
