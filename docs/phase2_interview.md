# Phase 2 面试指导：LangGraph 状态机

## 本阶段目的

Phase 1 已经有确定性的分类和检索执行器。Phase 2 把相同逻辑组织成显式 LangGraph：节点负责状态转换，条件边负责路由，终止节点负责收敛。这样 Agent 的每一步都有输入、输出、失败位置和可测试的状态，而不是把流程藏在一个大函数里。

## 工作流与代码对应

```text
START
  → classify
  → [plan_single | plan_multi]
  → retrieve
  → finalize
  → END
```

- `src/sc_agent/graph.py`：`GraphState`、节点、条件路由和编译后的图。
- `classify`：复用 Phase 1 分类规则，写入 `task_type` 和 `route`。
- `plan_single`：事实查询的一条检索路径。
- `plan_multi`：比较、计算和综述任务的多子任务路径。
- `retrieve`：复用 SC-RAG 边界，限制最大步骤，去重证据并记录失败。
- `finalize`：有证据则完成，没有证据则进入 `needs_evidence`，不会伪造答案。
- `tests/test_phase2_graph.py`：验证条件路由、证据去重和失败保留。

Phase 2 仍然不调用 LLM，也不执行任意 shell。LangGraph 在这里首先解决的是状态编排和可观察性，不是“让模型无限循环”。

## 运行

安装 Agent 依赖：

```bash
cd /workspaces/Agent
.venv/bin/python -m pip install -e '.[agent,dev]'
```

启动 SC-RAG 后运行：

```bash
.venv/bin/sc-agent graph \
  "compare Sobol and P2LSG multiplier accuracy" \
  --paper Sobol --paper P2LSG
```

## 面试问题

**为什么要用 LangGraph，而不是继续调用一个普通函数？**

普通函数适合固定流程，但后续需要根据任务类型、证据完整性、工具失败和人工审批决定下一步。LangGraph 将节点、条件边和终止条件显式化，便于检查状态、恢复执行和评测每条路径。

**Phase 2 是否已经是真正的 Agent？**

它具备 Agent 的状态机和条件执行骨架，但规划仍然是确定性的。这样先建立可控基线；在后续阶段引入模型规划器时，可以对比模型路由和确定性路由的收益与风险。

**如何防止图无限循环？**

当前图是无环的，并且检索节点使用 `max_steps` 截断计划。后续加入反思边时，还要增加全局步数、重试次数、Token 预算和重复状态检测。

**为什么 `needs_evidence` 比生成一个猜测更好？**

科研任务的错误结论成本高。没有可信证据时，状态机应该显式停止，把缺口交给后续补充检索或人工输入，而不是用生成模型填空。

**Phase 1 和 Phase 2 的边界是什么？**

Phase 1 证明工具边界和确定性执行逻辑；Phase 2 证明这些逻辑可以被拆成状态节点和条件路由。两者共享同一套数据契约，因此后续更换规划器不会重写检索工具。
