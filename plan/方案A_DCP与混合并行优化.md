# 方案 A：DCP 与混合并行优化

日期：2026-10-28  
项目：`/mnt/nvme1n1/data/lmk/PROJECT/TEST2/rooflang`  
性质：设计与可行性分析；依据现有代码和已有结果整理，未开展新的实验。

配套方案：[方案 B：通信组织与执行调度优化](方案B_通信组织与执行调度优化.md)。

## 1. 结论与定位

建议将本方案作为主攻路线，但首先限定为 GB300 8 GPU、Kimi K3 decode、PP1。目标是扩展可信的 TP/DPA/DCP 组合表达，并研究低并发、长上下文及 KV 容量受限场景。

完整 DCP 在工程上可实现，但不能保证相对当前最优 DPA 前沿提升30%。实现 DCP、证明语义正确、发现性能收益是三个独立目标。不要把新增功能本身作为优化成功的证据。

当前 CP 搜索已经覆盖多个分割比例，继续扩大同类搜索投入价值较低。更有价值的是补齐当前模型不能准确表达的 TP、KDA head 分片和小 batch 场景。

## 2. 固定条件与目标

- 固定模型语义、权重、精度和硬件：Kimi K3，GB300 8 GPU。
- 仿真硬件构造：`GB300Cluster(nvl_scope=8, n_nodes=1)`。
- 首版限定 decode、PP1；不引入量化变化、裁剪、近似 attention 或改变专家选择。
- DCP 是候选动作，允许 Agent 选择 DCP1，不能强制每个负载都使用 DCP。
- 主要交付：设计文档、合法动作空间、最小正确性用例、全模型仿真接入、Agent 闭环。

建议主性能指标为：相同负载、同一延迟约束下，相对预先冻结的可运行基线提升吞吐。另行报告固定 batch 下的每步延迟。吞吐提升30%不等同于延迟降低30%。

## 3. 当前实现与真实 DCP 的差异

相关代码：

- `programs/models/kimi_k3/optimization.py`：decode 分割、放置和状态持久化。
- `language/optimization/split.py`：context split、Query Broadcast 和输出 Reduce。
- `language/kernels/forward.py`：MLA、KDA 等算子的 FLOPs/字节模型。
- `language/kernels/comm.py`：现成的 AllToAll 等集合通信原语。
- `programs/experiments/find_pareto_frontier.py`：配置枚举、OOM、Pareto 和计时投影。

当前8卡 PP1搜索满足 `CP × DP = EP = 8`。MLA 按 context 分片，KDA 则把 CP 用作额外 batch 分片。当前 MLA 合并是通信代价抽象，没有显式表达 partial output、LSE 和精确 softmax merge。

拟实现的 DCP 数据流：

```text
每 rank 获得本 DCP 组所需 Query
    → 扫描本地 KV 位置分片
    → partial output + LSE
    → packed All-to-All
    → 按 LSE 权重合并
    → 输出满足后续 TP/head 布局
```

位置分配可表达为 `owner(p) = p mod dcp`。归并公式为：

```text
lse = logsumexp_r(lse_r)
out = Σ_r exp(lse_r - lse) × out_r
```

普通求和不能替代 softmax 分片合并。数学等价也不自动意味着浮点逐位相等，必须预先定义数值容限、归约精度和精度不变的含义。

KDA 状态没有历史 token 位置轴，应保持按 head/TP 分片；其投影、门控、卷积状态和输出投影需要一致布局。

## 4. 为什么已有结果 CP1 最优

现有结果目录：`results/kimi_k3_gb300_8gpu_decode`。

此前读取的 Pareto CSV 包含35个前沿点，CP均为1；这些是当时结果快照，不代表文件以后更新仍保持相同统计。

固定全局 batch、context、PP1、EP8时：

```text
每卡 MLA KV ∝ (B / DP) × (L / CP)
             = B × L / 8
```

因此 CP 增大同时 DP 减小，没有减少每卡 KV 总量或近似 attention 总工作量，却增加通信。

已有结果中，batch128、PP1、EP8：

| Context | CP1延迟 | CP8延迟 | 两者KV HBM记录 |
|---|---:|---:|---:|
| 8K | 32.463 ms | 33.664 ms | 5.61 GB |
| 32K | 33.143 ms | 34.344 ms | 11.05 GB |
| 64K | 34.049 ms | 35.250 ms | 18.29 GB |
| 128K | 35.861 ms | 37.062 ms | 32.79 GB |

该 KV 记录来自项目统计字段，不应未经拆分就解释为纯 MLA KV；项目还统计 KDA 状态等持久缓存。

这说明当前采样和代价模型下CP1占优，不能推导所有真实场景下CP1必然最优。

## 5. 基线选择决定收益含义

必须同时保留两种基线：

1. TP8/DCP1：同一批请求的 MLA KV 在 attention TP rank 间重复保存。
2. 现有 DPA8：不同 rank 处理不同请求，本来就没有同一请求 KV 的跨卡重复。

相对第一种基线，DCP8 可将 MLA KV 每卡占用近似降到1/8；这不是整张卡显存下降8倍。权重、KDA state、临时缓冲不会同步缩小，Query复制还可能增加内存。

相对第二种基线，均匀同一 batch 下 DCP8 和 DPA8 的每卡 MLA KV 都近似为 `B×L/8`，不能期待相同容量增益。

因此应分别报告“DCP相对TP收益”和“最终方案相对现有最佳DPA收益”。新增batch1可行点也不能直接与batch8基线比较后宣布提升30%。

## 6. 动作与前置条件

| 动作 | 前置条件 | 额外代价 | 适用场景 |
|---|---|---|---|
| MLA位置分片 | DCP组合法嵌套、KV归属唯一、支持精确merge | partial/LSE缓冲、A2A与merge | 长context、重复KV受限 |
| KDA head/TP | head和投影可分片、状态归属一致 | 投影集合通信、布局衔接 | 低并发利用多卡 |
| DPA与DCP组合 | 每组拥有正确请求集合、attention TP可整除DCP | 分组与层间布局成本 | 多负载折中 |
| Query投影复制 | 权重精度、形状与显存允许 | 额外权重、GEMM和读取 | 用计算换同步减少 |

概念约束可用 `attention_tp_size % dcp_size == 0`，但配置接口应明确总设备数、DPA、TP、EP的复用关系。不能把所有并行度简单相乘，也不能把原来的CP直接更名为DCP。

首版不加入多节点、复杂PP、动态offload和在线变更DCP组。

## 7. 预期效果与30%门槛

| 目标 | 静态判断 |
|---|---|
| 降低TP基线重复MLA KV | 有明确机制，可信度高 |
| 增大长context容量 | 条件成立：其他状态和权重不能先成为瓶颈 |
| 改善低并发长请求延迟 | 值得研究，需完整TP/KDA路径 |
| 高并发均匀负载超越DPA8 30% | 当前证据不足 |
| 不均匀请求的尾延迟优化 | 有潜力，但现有固定长度负载表达不足 |

以可优化部分占比f、局部加速s计算，整体加速 `S = 1/(1-f+f/s)`。若要求S至少1.3：

- 局部耗时完全消除，也需要f至少23.1%。
- 局部只加速2倍，需要f至少46.2%。
- 局部加速8倍，需要f至少26.4%。

实际还有通信、复制和merge开销，要求更高。context达到128K本身不足以证明满足这些条件。`compute_ratio`包含本地计算/访存路径，不能解释为Tensor Core利用率。

## 8. 工程成本

以下按一名熟悉Python和LLM并行的工程人员、Agent辅助开发估计，包含必要检查和调试，不包含真实GPU kernel开发和硬件校准。

| 工作包 | 估计人日 |
|---|---:|
| 设计、参数与合法性契约 | 2–3 |
| 单MLA层DCP、LSE/A2A和参考用例 | 3–6 |
| KDA head/TP、投影、状态、层间布局 | 7–12 |
| 全模型接入、计费和回归 | 4–7 |
| 架构合计 | 16–28 |

主要风险是层间AttnRes、状态持久化、权重去重与缓冲峰值，不是LSE公式本身。单层原型更便宜，但不能支撑全模型30%结论。

## 9. Agent闭环的实施边界

现有Agent模板 `agents/find_pareto_frontier/AGENTS.md` 只允许优化文件和实验目录变更，禁止language/runtime修改及非默认stream，目标是理想overlapped前沿。这些限制无法直接支持本方案完整开发。

建议分层：

- 开发层：工程人员与编码Agent补齐可信原语、布局和评价接口。
- 搜索层：冻结模型、硬件、原语语义和评分，Agent只提交受约束配置或可信图变换。

闭环为：提出结构化候选 → 合法性检查 → 仿真 → Pareto更新 → 失败/瓶颈反馈 → 下一候选。

现有进程调度、JSONL、OOM、续跑和Pareto可复用。闭环新增成本约6–12人日，包含候选契约、缓存与版本、反馈、预算控制以及规则/随机搜索对照；与架构工作可部分重叠。

若只有DCP四个取值，穷举足够。Agent价值应由相同预算下的搜索效率、无效候选率和跨负载迁移体现，而不是宣称超过同一有限空间的完整穷举。

## 10. 后续最小验证计划（本次未执行）

1. 小张量完整attention与分片LSE合并数值对照，覆盖非均匀logits和分片边界。
2. 检查KV唯一归属、每卡权重/状态占用、A2A字节和临时缓冲。
3. 检查KDA按head分片后的状态更新与输出投影一致性。
4. 接入全模型后统一比较TP、DPA、DCP，固定模型、精度、硬件和负载。
5. 候选实验采样可从batch1/8/32、context8K/128K/512K、DCP1/2/4/8开始；这不是收益保证。
6. 原始elapsed作为可执行结果，理想overlap单列上界；预加载与稳态decode分开报告。

## 11. 投入与停止规则

- 优先交付PP1受限实现，不同时扩展所有并行维度。
- 数学资源核算若显示候选与DPA每卡工作量相同、只增加通信，降低其搜索优先级。
- 无足够关键路径或容量收益空间时，不以30%为理由修改计费常数。
- 若30%必须相对当前最佳DPA实现，应承认它目前是高风险目标，不应全部押注DCP。
- 架构加闭环可按约4–7周安排；工程估计不代表性能结果保证。

## 参考

- [LMSYS Kimi K3 day-0 support](https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support)
- [SGLang DCP文档](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/dcp.mdx)
- 项目现有结果：`results/kimi_k3_gb300_8gpu_decode/all_points.csv`、`pareto_frontier.csv`。

外部文档会随版本变化；这里记录的是本次讨论采用的设计依据，不代表部署版本兼容性承诺。
