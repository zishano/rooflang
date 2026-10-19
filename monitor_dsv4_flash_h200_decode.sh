#!/usr/bin/env bash
# 监控单步 decode 仿真；额外参数传给监控脚本，使用 --help 查看。
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$script_dir"
exec "$script_dir/.venv/bin/python" "$script_dir/tools/monitor_decode.py" "$@"
