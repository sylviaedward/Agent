# Phase 5 面试指导：结构化抽取、单位解析与人工审批

## 本阶段目的

Phase 4 的规则抽取可解释，但无法覆盖自然语言中的单位、条件和跨句表达。Phase 5 让 LLM 做候选发现，同时把最终可信度判断留给确定性校验和人工审批。LLM 输出永远不是执行参数的直接来源。

## 工作流

```text
EvidenceItem
  → LLM JSON candidate extraction
  → JSON schema validation
  → source span + quote verification
  → unit normalization and range checks
  → missing/conflict/condition gate
  → auto approval or pending_human
  → explicit reviewer decision
```

代码对应关系：

- `src/sc_agent/phase5.py`：`LLMParameterExtractor`、单位解析和 `HumanApprovalFlow`。
- `src/sc_agent/models.py`：候选参数、Phase 5 抽取结果和审批记录。
- `tests/test_phase5_extraction.py`：单位换算、伪造 quote、未知单位和人工审批测试。

## 安全边界

- 每个候选必须引用存在的 `source_span_id`。
- 候选数组按上限处理，默认最多接受 20 个候选；超出部分不会进入校验。
- 每个候选独立进行 Schema 校验；缺失 quote、数组值或非法类型只会拒绝该候选，不会让整个批次崩溃。
- quote 缺失时，系统会在候选声明的 `source_span_id` 原文中按参数类型做唯一 lexical match；匹配成功才自动补 quote，并记录 `quote_recovered=true`。
- 无法唯一定位时仍然拒绝候选，绝不会根据模型记忆或相邻论文内容编造 quote。
- quote 的空白差异可以归一化，但内容必须确实出现在对应证据中。
- 百分比、概率、秒/毫秒/微秒/纳秒会转换为统一数值；未知单位直接拒绝。
- `x` 和 `y` 必须落在 `[0, 1]`；长度必须为正数。
- 条件限定（例如低相关性、特定工艺或特定数据集）会进入 `pending_human`。
- 缺失、冲突、非法 JSON 或非法单位都不能触发模拟、写报告或导出结果。

## 面试重点

**为什么还需要确定性校验？**

LLM 擅长从跨句文本中发现候选，但它可能改写数字、引用错误 span 或忽略单位。校验器检查结构、来源、数值范围和单位，形成可审计的信任边界。

**为什么不让模型直接输出最终参数？**

最终参数会进入模拟器或研究报告，错误会产生可执行的假结果。模型只能提出候选，只有通过证据校验并获得审批的结果才可继续。

**什么时候自动批准？**

只有所有必需字段完整、无冲突、单位可解析且没有条件歧义时，系统才返回 `auto_approved`。其余情况必须由研究人员显式批准或拒绝。

**如何处理两个论文来源的同名参数？**

先把单位转换后的值比较；仍有不同值就保留冲突及来源，不静默选择。人工审批时需要确认实验条件是否相同。

## 最小调用示例

配置 `OPENAI_BASE_URL`、`OPENAI_API_KEY` 和 `OPENAI_MODEL_ID` 后，可以直接运行：

```bash
cd /workspaces/Agent
.venv/bin/sc-agent phase5 \
  "extract stochastic multiplier parameters and compare reported error" \
  --x 0.3 --y 0.6 --length 100000 --seed 7
```

该命令使用 OpenAI 兼容的 `/chat/completions` 接口。模型输出只进入候选校验，
终端结果中的 `approval.status` 才决定是否可以继续执行或导出。

超时和输入预算可以按网关性能调整：

```bash
export SC_AGENT_LLM_TIMEOUT_SECONDS=180
export SC_AGENT_EXTRACTION_MAX_CHARS=24000
export SC_AGENT_EXTRACTION_MAX_CHARS_PER_SPAN=5000
export SC_AGENT_EXTRACTION_MAX_CANDIDATES=20
```

证据裁剪只发生在发送给模型的 prompt 中；本地仍保留完整 `EvidenceItem`，
因此 quote 校验不会因为 prompt 预算而放宽。

```python
from sc_agent.phase5 import HumanApprovalFlow, LLMParameterExtractor

extraction = LLMParameterExtractor(llm_client).extract(evidence)
approval = HumanApprovalFlow().prepare(extraction)
# pending_human 时展示 approval.reasons 和 approval.parameters，等待人工决定。
```
