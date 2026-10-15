# LLM 推理软件自主优化：方案与最小验证

- `方案设计.md`：两类软件架构、主攻路线、动作前置条件与代价、AI闭环、验证门槛及实施计划。
- `minimal_validation.py`：可执行的 DSV4 Flash MoE 片段，使用原有 RoofLang 类与模拟器。
- `validation/results.json`：包含逐算子 trace、峰值内存、冻结契约及正反例结果。
- `validation/results.csv`：结果汇总。

在仓库根目录运行（Python 环境需安装 RoofLang 依赖）：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$(pwd)/.." python \
  results/autonomous_optimization_design/minimal_validation.py \
  --output-dir results/autonomous_optimization_design/validation
```

正例片段耗时 4.445239 → 3.261297 μs，降低26.63%；峰值内存增加128 KiB。资源不互补/有显式依赖时不加速，容量约束反例返回OOM。

这是输入预先就绪的三算子片段，不是完整43层模型或真实B300运行。没有实现AI规划器，也没有执行数值等价和GPU实测。源代码与原有搜索器未被修改。

重新绘制时间线：

```bash
python results/autonomous_optimization_design/plot_timeline.py
```

绘图需要 Matplotlib。默认尝试使用系统 Noto CJK 字体；可用 `--font /path/to/font.ttf` 指定其他中文字体。
