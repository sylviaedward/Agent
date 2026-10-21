# Phase 6 面试指导：人工审批后的执行与报告导出

## 本阶段目的

Phase 5 负责把不可信的模型输出变成候选参数。Phase 6 建立最后一道执行边界：只有抽取完整且审批状态为 `approved` 或 `auto_approved` 时，参数才能进入模拟器；`pending_human`、`rejected`、`missing` 和 `conflict` 都会返回 `blocked`。

## 工作流

```text
Phase5Extraction + ApprovalRequest
  → approval gate
  → normalized parameter gate
  → typed SimulationSpec
  → bounded simulator
  → evidence-linked JSON report
```

代码对应关系：

- `src/sc_agent/phase6.py`：`Phase6Executor` 和报告导出。
- `src/sc_agent/models.py`：`Phase6Result`。
- `src/sc_agent/simulator.py`：受限、可复现的随机计算模拟器。
- `tests/test_phase6_execution.py`：拒绝执行、批准执行和禁止导出阻断结果测试。

## 安全规则

- 审批状态不是 `approved` 或 `auto_approved` 时，模拟器不会被调用。
- 参数必须来自 Phase 5 的 `normalized_value`，不能从未验证的模型字段直接执行。
- 模拟器仍受操作白名单、输入范围、长度和随机种子约束。
- 只有 `executed` 结果可以导出报告。
- 报告保存参数、模拟结果、审批状态、证据 span、论文别名、页码和 quote，便于复核。

## 面试问题

**为什么把人工审批放在工具执行前？**

模拟和报告导出会产生外部可见结果。审批必须先于工具调用，避免模型或不完整参数触发不可逆的研究动作。

**如何保证导出的报告可审计？**

报告同时保存参数的来源 span、quote、论文别名和页码，并保存审批状态与模拟配置。复核者可以沿着同一 span 回到检索证据。

**审批通过后是否还需要参数校验？**

需要。人工审批不能替代类型、范围和操作白名单校验；执行器会再次构造 `SimulationSpec`，由 Pydantic 和模拟器完成边界检查。

## 最小调用

```python
from sc_agent.phase6 import Phase6Executor

result = Phase6Executor().execute(request, extraction, approval)
report = Phase6Executor.export_report(result, evidence, "artifacts/phase6_report.json")
```
