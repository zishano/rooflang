#!/usr/bin/env bash
# Kimi K3：GB300 8 GPU、32K/64K/128K decode；搜索合法并行配置，batch 翻倍直到 OOM。
# --hardware gb300 --gpu-counts 8 对应 GB300Cluster(nvl_scope=8, n_nodes=1)。
# PP 使用均匀层数划分；这是当前搜索器的采样范围，不是所有架构的穷举。
# 同目录结果自动续跑。WORKERS=1 可降低主机资源占用；追加 --dry-run 查看规模。
# 从已有结果单独绘制固定 batch：追加 --plot-only --plot-batch-sizes 256 512。
# 从已有结果重绘各 CP 曲线：追加 --plot-only；默认叠加 CP1/2/4/8。
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$script_dir"
export PYTHONPATH="$(dirname -- "$script_dir")${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$script_dir/.cache/matplotlib"
export MPLBACKEND=Agg

python_bin="${ROOFLANG_PYTHON:-$script_dir/.venv/bin/python}"
if [[ ! -x "$python_bin" && -z "${ROOFLANG_PYTHON:-}" ]]; then
  python_bin="/mnt/nvme1n1/data/lmk/PROJECT/rooflang/.venv/bin/python"
fi
if [[ ! -x "$python_bin" ]]; then
  echo "Python environment not found: $python_bin" >&2
  echo "Set ROOFLANG_PYTHON to a Python executable with requirements installed." >&2
  exit 127
fi

exec "$python_bin" -m rooflang.programs.experiments.find_pareto_frontier \
  --model kimi_k3 \
  --workloads decode-32k decode-64k decode-128k \
  --hardware gb300 \
  --gpu-counts 8 \
  --workers "${WORKERS:-2}" \
  --output-dir results/kimi_k3_gb300_8gpu_decode \
  --point-labels \
  --plot-cp-degrees 1 2 4 8 \
  "$@"
