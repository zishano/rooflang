#!/usr/bin/env bash
# 独立脚本：单次 decode 仿真，按 Pareto 代码定义输出三种时间模型。
# 不依赖 run_layer_timeline.sh 或已有 CSV/JSON。
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# ==================== 在这里修改参数 ====================
model="dsv4_flash"                 # dsv4_flash / dsv4_pro
hardware="H200Cluster1Node"        # H200Cluster1Node / B300Cluster1Node
batch_size=8
context=8192
cp=2
dp=2
ep=4
pp_partition=(22 21)               # 不使用 PP 的示例：(43)，同时改 dp=4、ep=8
output_dir="$script_dir/results/pipeline_timing"
# cp*dp == ep；ep*PP段数 == 8；所有 PP 段的层数之和等于模型层数。
# original：单批输出完成延迟；elapsed：最慢 stage 周期；overlapped：理想隐藏后的最慢 stage 周期。
# ======================================================

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
exec "$python_bin" "$script_dir/analysis/pipeline_timing.py" \
  --model "$model" --hardware "$hardware" \
  --batch-size "$batch_size" --context "$context" \
  --cp "$cp" --dp "$dp" --ep "$ep" \
  --pp-partition "${pp_partition[@]}" --output-dir "$output_dir" "$@"
