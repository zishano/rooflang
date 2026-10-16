# 已有基线证据与复现

## Flash 单步 Decode 示例

以下为已在本地运行的完整模型结果摘要，模型与模拟器代码对应提交 `ad01ff8`。这是一个已测配置，**不是已证明最优的强基线**。原始大规模trace与其他搜索输出未随本次设计文档提交；后续D02需归档正式基线manifest、完整记录与必要trace。

配置：`dsv4_flash`，B300×8，batch=512，历史8192，CP1/DP8/EP8，PP=[43]。

| 项目 | 时间 |
|---|---:|
| 历史 KV 预加载 | 143.248883647 ms |
| original 输出延迟 | 4.198020139 ms |
| 输出后收尾 | 0.001000004 ms |
| 全部仿真时间 | 147.447903790 ms |

| 完整 Transformer 层类型 | 数量 | 单层时间 μs |
|---|---:|---:|
| SWA，ratio=0 | 2 | 89.431034 |
| CSA，ratio=4 | 21 | 92.678238 |
| HCA，ratio=128 | 20 | 89.701635 |

每层 MoE 暴露通信尾部约 `5.871461849 μs`。仅将这部分全部隐藏的机制上限：

```text
可隐藏时间 = 43 × 5.871461849 ≈ 252.472860 μs
占原始输出延迟 = 252.472860 / 4198.020139 ≈ 6.0141%
```

此估算只限定“其他工作不变、仅隐藏现有MoE通信尾部”的方案，不是所有图改写或调度策略的上限。

### 复现单配置

从仓库根目录运行，使用已安装依赖的 Python（本环境可用 `.venv/bin/python`）。该命令调用原搜索器的同一 `run_case`，输出其真实计时字段，不把CLI主机运行时间当推理延迟。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$(pwd)/.." python - <<'PY'
import json
from rooflang.programs.experiments import find_pareto_frontier as finder

finder._select_model("dsv4_flash")
case = finder.Case(
    workload="decode-8k", hardware="b300", n_gpus=8,
    batch_size=512, cp=1, dp=8, ep=8, pp_partition=(43,),
)
record = finder.run_case(case)
assert record["status"] == "ok", record
print(json.dumps(record, indent=2))
PY
```

该命令验证完整配置计时；分层trace和全域强基线属于后续D02交付，不把单点复现等同于搜索完成。

## 已提交的异步执行 MVP

[源码和说明](../../results/autonomous_optimization_design/README.md) 与 [结果 CSV](../../results/autonomous_optimization_design/validation/results.csv) 已在版本控制中。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$(pwd)/.." python \
  results/autonomous_optimization_design/minimal_validation.py \
  --output-dir /tmp/rooflang-autonomous-mvp
```

已记录：正例片段延迟 `4.445239 → 3.261297 μs`，峰值内存增加128 KiB；本地资源争用和显式依赖反例不加速；容量反例拒绝OOM候选。

该片段所有根输入预先就绪，不是完整43层模型、目标低精度数值执行或真实GPU测量。现阶段没有新路线A的完整实现，也没有完整模型30%收益或H100实测结果。
