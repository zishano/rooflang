#!/usr/bin/env bash
# 修改下面的参数区后，直接运行本脚本即可生成单 GPU 单层时间线。
# 使用 --print-config 仅显示实际参数，不执行模拟。
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# ==================== 模拟参数（在这里修改） ====================
model="kimi_k3"                    # Kimi K3
hardware="GB300Cluster8GPU"        # GB300Cluster(nvl_scope=8, n_nodes=1)
batch_size=8                       # 全局 batch size
context=8192                       # decode KV context 长度
cp=2                               # Context Parallel
dp=4                               # Data Parallel
ep=8                               # Expert Parallel
pp_partition=93                    # 单个 PP stage 包含 Kimi K3 的全部 93 层
gpu_index=0                        # GPU 编号：0～7
layer=0                            # 模型层编号，从 0 开始
include_pp=false                    # PP=1，没有跨 stage 的 PP 边界
output_dir="$script_dir/results/kimi_k3_gb300_8gpu_layer_timing"
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

# 保留调用者工作目录，--output-dir 相对路径按调用者目录解析。
pp_options=()
if [[ "$include_pp" == true ]]; then
  pp_options+=(--include-pp)
fi
exec "$python_bin" "$script_dir/analysis/kimi_k3_single_layer_timing.py" \
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
