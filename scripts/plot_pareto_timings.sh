#!/usr/bin/env bash
# 绘制三种计时前沿、合并前沿和灰色候选点；不运行仿真。
# 同时生成 PNG/PDF 和交互式 HTML；悬停前沿点可查看 case_id 与配置。
# 默认读取 Flash 论文实验结果；其他参数直接传给 Python 绘图脚本。
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd -- "$script_dir/.." && pwd)
python_bin="${PYTHON_BIN:-$project_dir/.venv/bin/python}"

exec "$python_bin" -B "$script_dir/plot_pareto_timings.py" \
  --input "$project_dir/results/dsv4_flash_paper_figures" \
  "$@"
