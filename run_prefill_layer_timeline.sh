#!/usr/bin/env bash
# 修改下面的参数区后，直接运行本脚本即可生成单 GPU 单层 prefill 三轨时间线。
# 使用 --print-config 仅显示实际参数，不执行模拟。
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# ==================== 模拟参数（与 run_layer_timeline.sh 对齐） ====================
model="dsv4_flash"                  # dsv4_flash 或 dsv4_pro
hardware="H200Cluster1Node"         # H200Cluster1Node 或 B300Cluster1Node
batch_size=8                       # 全局 batch size
context=8192                       # prefill 输入序列长度
cp=2                               # Context Parallel
dp=2                               # Data Parallel
ep=4                               # Expert Parallel
pp_partition=(22 21)               # 每个 PP stage 的层数；Flash 共 43 层，Pro 共 61 层
gpu_index=0                        # GPU 编号：0～7
layer=0                            # 模型层编号，从 0 开始
include_pp=true                   # 包含本层输出侧 PP 边界；(22 21) 时选 layer=21、gpu_index=0
output_dir="$script_dir/results/prefill_layer_timing"
# 约束：cp * dp == ep；ep * PP stage 数 == 8。
# batch_size 必须能被 cp*dp 整除；选定 GPU 必须属于该层所在 PP stage。
# ==============================================================

export PYTHONPATH="$(dirname -- "$script_dir")${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$script_dir/.cache/matplotlib"
export MPLBACKEND=Agg

python_bin="${ROOFLANG_PYTHON:-$script_dir/.venv/bin/python}"
if [[ ! -x "$python_bin" && -z "${ROOFLANG_PYTHON:-}" ]]; then
  python_bin="/mnt/nvme1n1/data/lmk/PROJECT/rooflang/.venv/bin/python"
fi
if [[ ! -x "$python_bin" ]]; then
  printf 'Python 不可执行：%s\n请通过 ROOFLANG_PYTHON 指定 Python 解释器的完整路径。\n' "$python_bin" >&2
  exit 1
fi

pp_options=()
if [[ "$include_pp" == true ]]; then
  pp_options+=(--include-pp)
fi
exec "$python_bin" "$script_dir/analysis/single_layer_timing.py" \
  --stage prefill \
  --model "$model" \
  --hardware "$hardware" \
  --batch-size "$batch_size" \
  --context "$context" \
  --cp "$cp" --dp "$dp" --ep "$ep" \
  --pp-partition "${pp_partition[@]}" \
  --gpu-index "$gpu_index" \
  --layer "$layer" \
  --output-dir "$output_dir" \
  --two-lane "${pp_options[@]}" "$@"
