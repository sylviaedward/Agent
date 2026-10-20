# Phase 1 面试指导：确定性研究任务执行

## 本阶段目的

Phase 0 只证明 Agent 能访问 SC-RAG。Phase 1 把一次用户请求变成可观察、可测试的检索任务：先判断任务类型，再生成受限检索计划，调用 SC-RAG，校验返回结构，去重证据，并记录失败。

这一阶段还没有让模型自由选择工具，也没有执行 shell、文件写入或模拟实验。原因是先建立可靠的工具边界，后续 LangGraph 节点才有稳定的状态转换对象。

## 工作流与代码对应

```text
ResearchRequest
  → classify_request
  → build_plan
  → ScRagClient.search
  → validate / deduplicate EvidenceItem
  → Phase1Result
```

- `src/sc_agent/models.py`：请求、计划、证据和结果的数据契约。
- `src/sc_agent/planner.py`：基于可解释规则的分类、查询规范化和显式论文拆分。
- `src/sc_agent/sc_rag_client.py`：唯一的检索服务边界，包含参数校验和超时配置。
- `src/sc_agent/phase1.py`：受 `max_steps` 限制的执行器，记录成功、失败和未解决问题。
- `src/sc_agent/cli.py`：命令行演示入口。
- `tests/test_phase0_contracts.py`：分类、显式目标覆盖、证据去重和失败记录测试。

比较任务会为每个显式论文别名创建独立子任务。例如 `Sobol` 与 `P2LSG` 不会只依赖一次混合检索，这为后续跨论文证据覆盖提供了基础。

## 运行

先启动 SC-RAG API，然后在 Agent 仓库执行：

```bash
cd /workspaces/Agent
.venv/bin/sc-agent run "compare Sobol and P2LSG multiplier accuracy" \
  --paper Sobol --paper P2LSG
```

输出中的 `queries`、`evidence`、`tool_runs` 和 `unresolved_issues` 分别对应计划、证据、工具调用记录和待恢复问题。

## 面试问题

**为什么 Phase 1 不直接让大模型决定下一步？**

先用确定性规则建立可重复基线。否则出现错误时无法区分是分类、规划、工具调用还是模型自由决策导致的。后续可以把分类替换为模型，但必须保留结构化输出、白名单和预算约束。

**为什么比较任务要拆成多个检索子任务？**

一次混合检索可能被某篇论文的高相似度切片占满，导致另一篇目标论文没有证据。显式目标拆分把“每个来源必须有机会被检索”变成系统约束。

**Phase 1 如何处理恶意论文内容？**

SC-RAG 返回的正文只被当成证据字段，不能进入系统指令，也不能改变工具白名单。Phase 1 不执行论文中的命令；沙箱和注入检测会在后续安全阶段加入。

**为什么要记录失败而不是直接重试？**

Phase 1 先记录失败类型和子任务，避免隐式无限重试。后续反思阶段会根据统一的失败记录决定是否进行一次有预算的补充检索。

**这一阶段如何证明不是普通 RAG？**

系统已经具备任务分类、条件规划、显式来源覆盖、结构化状态和工具失败记录。RAG 负责证据，Agent 层负责决定当前任务需要哪些步骤以及如何处理步骤失败。
