# Phase 4 面试指导：证据绑定的参数抽取与指标对齐

## 本阶段目的

Phase 3 已经可以运行模拟，但模拟参数仍由用户直接提供。Phase 4 开始处理最容易产生幻觉的环节：从论文证据中抽取参数，并判断模拟结果能否与论文结果比较。

本阶段的安全规则是：没有原文片段就没有参数；出现冲突就不自动选择；指标名称或单位不一致就不比较。抽取器只读取检索到的 `EvidenceItem.text`，不会使用模型记忆或默认值补全。

## 工作流与代码对应

```text
SC-RAG evidence
  → lexical parameter extraction
  → source span binding
  → missing/conflict gate
  → metric canonicalization
  → comparable / not_comparable
```

- `src/sc_agent/phase4.py`：`EvidenceParameterExtractor` 和 `MetricAligner`。
- `src/sc_agent/models.py`：`ExtractedParameter`、`ParameterExtraction` 和 `MetricAlignment`。
- `src/sc_agent/graph.py`：Phase 2/3 结果作为 Phase 4 输入。
- `src/sc_agent/cli.py`：`sc-agent phase4` 入口。
- `tests/test_phase4_alignment.py`：来源绑定、冲突拒绝和指标不对齐测试。

当前抽取器使用可解释的正则规则作为基线。后续可以加入结构化 LLM 抽取，但必须通过同一套 span 校验和缺失门控，不能直接信任模型 JSON。

## 运行

```bash
cd /workspaces/Agent
.venv/bin/sc-agent phase4 \
  "verify stochastic multiplier error by simulation" \
  --x 0.3 --y 0.6 --length 100000 --seed 7
```

重点观察：

- `parameter_extraction.status`：`complete`、`missing` 或 `conflict`；
- 每个 parameter 是否有 `source_span_id` 和 `quote`；
- `metric_alignment.status`：只有严格可比时才是 `comparable`；
- `unresolved_issues` 是否明确说明缺失字段或冲突来源。

## 面试问题

**如何防止模型编造论文实验参数？**

参数对象必须携带检索 span ID 和原文 quote，quote 必须来自系统返回的证据。缺少字段时返回 `missing`，多个来源值不一致时返回 `conflict`，两种情况都不自动生成模拟参数。

**为什么不能把 RMSE 和绝对误差直接比较？**

它们的统计定义不同。Phase 4 只允许明确支持的同名指标对齐，未知指标返回 `not_comparable`，避免看似合理的数值比较。

**为什么 Phase 4 先用规则抽取？**

规则抽取可解释、可复现，适合建立安全基线。引入 LLM 后，模型只负责候选抽取，最终仍由 source span、数值范围、单位和冲突校验决定是否接受。

**参数冲突时应该怎么做？**

保留所有候选及其来源，不静默选择。后续 Agent 可以发起定向补充检索或请求人工确认，但在确认前不能执行带冲突参数的实验。
