#!/usr/bin/env bash
# DSV4 Flash：H200 64 GPU，64K/1M prefill 与 decode 并行策略搜索。
# 每个 workload 搜索所有合法 CP/DP/EP/PP 配置，batch 自动翻倍至 OOM。
# 结果写入 results/dsv4_flash_h200_64gpu_64k_1m；同目录可续跑。
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

exec "$python_bin" -m rooflang.programs.experiments.find_pareto_frontier \
  --model dsv4_flash \
  --workloads decode-64k \
  --hardware h200 \
  --gpu-counts 64 \
  --workers "${WORKERS:-2}" \
  --output-dir results/dsv4_flash_pareto \
  --point-labels \
  "$@"
