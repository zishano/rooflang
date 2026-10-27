# AGENTS.md — RoofLang DCP Architecture Discovery

## 目标

针对 DSV4 Flash Decode，基于现有 RoofLang DCP Baseline，自主挖掘新的 DCP 数据流架构。

已知实验结论：Layer-Selective CP 未获得收益。本轮不再搜索分层 CP。

优先研究：

**Chunked DCP + Pipelined Attention Merge**

目标是在不改变 Attention 语义、不改变硬件资源预算的情况下，通过分块计算、部分结果合并和合法的计算通信重叠，改善 Decode Pareto Frontier。

## 1. 源码审计

首先定位真实仓库中的：

- DSV4 Flash Decode 图。
- DCP KV Sharding。
- Q Broadcast。
- Local Attention。
- Attention Merge。
- Graph Rewrite。
- Placement。
- Simulator。
- Pareto Search。

不得凭空假设文件路径或接口。

## 2. 图修改规则

原始 Baseline Graph 保持不变。

Agent 应通过独立的 Rewrite Pass 生成候选图：

Baseline:

Q Distribution → Local Attention → Global Merge

Candidate:

Q Distribution → KV Chunking → Chunk Attention → Partial Merge → Final Merge

每个候选图必须保留全部 KV 依赖与 Attention 全局 Softmax 语义。

## 3. 语义保持

使用稳定的在线 Softmax 统计量：

- 局部最大值 m。
- 局部指数和 l。
- 局部加权输出 a。

两个部分的合并为：

m = max(m1, m2)

l = exp(m1-m) × l1 + exp(m2-m) × l2

a = exp(m1-m) × a1 + exp(m2-m) × a2

最终输出 O = a/l。

必须验证候选输出与原始 Attention 的数值一致性，并检查浮点误差。

## 4. 架构搜索

搜索：

- kv_chunk_count = [1, 2, 4, 8]
- merge_topology = [flat, tree, hierarchical]
- overlap_policy = [sequential, pipelined]

所有候选必须建模：

- KV Read Bytes。
- HBM Read Time。
- Attention Compute Time。
- Kernel Launch Overhead。
- Partial Merge Communication。
- Final Merge Communication。
- GPU Memory Usage。
- Resource Contention。
- Critical Path。

禁止假设无限并发或零成本通信。

## 5. 自主优化循环

每轮：

1. 分析 Baseline 关键路径。
2. 提出新的 Chunking/Merge/Overlap 架构假设。
3. 生成候选执行图。
4. 验证语义与资源约束。
5. 运行 RoofLang 仿真。
6. 与固定配置 Baseline 做消融。
7. 比较 Pareto Frontier。
8. 记录成功和失败原因。
9. 自主决定继续优化或停止。

不需要人工逐轮审批。

## 6. 验收

至少交付：

- 一个新的 DCP 图重写规则。
- 对应的语义测试。
- 通信与计算重叠分析。
- 同配置性能对照。
- Pareto Frontier 对比。
- 失败案例和适用范围分析。

如果 Chunked DCP 在现有模拟器中无法产生可信的重叠收益，必须如实报告，并自主转向 Query Distribution 或 Attention Merge 通信组织优化，不得通过人为降低通信成本制造性能提升。