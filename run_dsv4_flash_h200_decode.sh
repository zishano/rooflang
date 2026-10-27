#!/usr/bin/env bash
# Flash 单步 decode：8 张 H200，batch=8，CP=2，DP=4，EP=8，PP=1。
# --dcp 启用可复用的显式状态更新图，默认 1 步；--decode-steps N 指定步数。
# --dcp-trace first/all 导出逐步完整 trace；默认仅输出小型汇总。
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

exec bash "$script_dir/run.sh" \
  --model dsv4_flash \
  --hardware H200Cluster1Node \
  --stage decode \
  --batch-size 8 \
  --context 65536 \
  --cp 2 --dp 4 --ep 8 \
  --pp-partition 43 "$@"
