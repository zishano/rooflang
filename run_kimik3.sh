#!/usr/bin/env bash
# Kimi K3 单步 decode：8 张 GB300，batch=8，CP=1，DP=8，EP=8，PP=1。
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

exec bash "$script_dir/run.sh" \
  --model kimi_k3 \
  --hardware GB300Cluster8GPU \
  --stage decode \
  --batch-size 8 \
  --cp 1 --dp 8 --ep 8 \
  --pp-partition 93
