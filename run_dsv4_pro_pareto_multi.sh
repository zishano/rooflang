#!/usr/bin/env bash
# DSV4 Pro：B300 多 GPU 数量、多上下文 decode 搜索。
# 每个工作负载/硬件/GPU 数量单独计算前沿；PP 均匀划分，batch 翻倍直到 OOM。
# 同目录结果自动续跑。WORKERS=1 可降低主机资源占用；追加 --dry-run 查看规模。
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$script_dir"
export PYTHONPATH="$(dirname -- "$script_dir")${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$script_dir/.cache/matplotlib"
export MPLBACKEND=Agg

exec "$script_dir/.venv/bin/python" -m rooflang.programs.experiments.find_pareto_frontier \
  --model dsv4_pro \
  --workloads decode-8k decode-64k \
  --hardware b300 \
  --gpu-counts 8 16 32 64 \
  --workers "${WORKERS:-2}" \
  --output-dir results/dsv4_pro_decode_multi \
  --point-labels \
  "$@"
