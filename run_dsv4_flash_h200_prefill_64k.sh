#!/usr/bin/env bash
# DSV4 Flash prefill 单层时间线：单节点 8×H200，context=64K。
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
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

exec "$python_bin" "$script_dir/analysis/single_layer_timing.py" \
  --stage prefill --model dsv4_flash --hardware H200Cluster1Node \
  --batch-size 8 --context 65536 --cp 2 --dp 4 --ep 8 \
  --pp-partition 43 --gpu-index 0 --layer 0 \
  --output-dir "$script_dir/results/dsv4_flash_h200_prefill_64k" --two-lane "$@"
