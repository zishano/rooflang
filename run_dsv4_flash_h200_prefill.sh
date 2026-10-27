#!/usr/bin/env bash
# Flash 单步 decode：8 张 H200，batch=8，CP=1，DP=8，EP=8，PP=1。
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

exec bash "$script_dir/run.sh" \
  --model dsv4_flash \
  --hardware H200Cluster1Node \
  --stage prefill \
  --batch-size 8 \
  --cp 2 --dp 2 --ep 4 \
  --pp-partition 22 21