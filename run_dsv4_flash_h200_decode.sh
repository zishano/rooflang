#!/usr/bin/env bash
# Flash 单步 decode：8 张 H200，batch=8，CP=1，DP=8，EP=8，PP=1。
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

exec bash "$script_dir/run.sh" \
  --model dsv4_flash \
  --hardware H200Cluster1Node \
  --stage decode \
  --batch-size 8 \
  --cp 1 --dp 8 --ep 8 \
  --pp-partition 43
