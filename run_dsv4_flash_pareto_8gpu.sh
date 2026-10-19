#!/usr/bin/env bash
# DSV4 Pro：B300 8 GPU、8K decode；搜索合法并行配置，batch 翻倍直到 OOM。
# PP 使用均匀层数划分；这是当前搜索器的采样范围，不是所有架构的穷举。
# 同目录结果自动续跑。WORKERS=1 可降低主机资源占用；追加 --dry-run 查看规模。
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$script_dir"
export PYTHONPATH="$(dirname -- "$script_dir")${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$script_dir/.cache/matplotlib"
export MPLBACKEND=Agg

exec "$script_dir/.venv/bin/python" -m rooflang.programs.experiments.find_pareto_frontier \
  --model dsv4_flash \
  --workloads decode-8k \
  --hardware b300 \
  --gpu-counts 8 \
  --workers "${WORKERS:-2}" \
  --output-dir results/dsv4_flash_decode \
  --point-labels \
  "$@"
