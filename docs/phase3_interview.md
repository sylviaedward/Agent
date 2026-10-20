# Phase 3 面试指导：随机计算模拟工具

## 本阶段目的

Phase 2 已经具备状态机，但还只会检索。Phase 3 增加第一个真正执行科研动作的工具：一个受限、可复现的 unipolar stochastic multiplier simulator。Agent 可以在检索后根据任务类型进入模拟节点，并返回期望值、模拟估计值和误差。

本阶段先要求用户显式提供 `x`、`y`、bitstream length 和 seed。参数抽取与论文实验复现会在后续阶段加入；先把模拟器本身做成可测试工具，避免把“参数识别错误”和“模拟器错误”混在一起。

## 工作流与代码对应

```text
compute request
  → classify
  → retrieve evidence
  → simulate
  → compare expected and estimated value
  → finalize
```

- `src/sc_agent/models.py`：`SimulationSpec` 和 `SimulationResult` 契约。
- `src/sc_agent/simulator.py`：固定 seed 的 Bernoulli bitstream、AND multiplier 和误差计算。
- `src/sc_agent/graph.py`：compute 路由到 `simulate` 节点；其他任务跳过模拟。
- `src/sc_agent/cli.py`：`--x`、`--y`、`--length`、`--seed` 参数。
- `tests/test_phase3_simulator.py`：复现性、收敛性、非法操作拒绝和图路由测试。

模拟器只允许白名单操作，长度上限为 1,000,000，不能执行模型提供的 Python 或 shell。`seed` 被保存在结果中，因此相同输入可以复现实验。

## 运行

启动 SC-RAG API 后执行：

```bash
cd /workspaces/Agent
.venv/bin/sc-agent graph \
  "verify stochastic multiplier error by simulation" \
  --x 0.3 --y 0.6 --length 100000 --seed 7
```

结果中的关键字段：

```text
simulation.expected        理论值 x × y
simulation.estimate        bitstream AND 的估计值
simulation.absolute_error  绝对误差
simulation.relative_error  相对误差
simulation.ones_output     输出 bitstream 中 1 的数量
```

## 面试问题

**为什么模拟器要作为工具，而不是让大模型直接计算？**

模拟需要随机采样、长度约束、固定 seed 和可重复输出。工具可以提供确定的输入输出契约，模型负责决定何时调用和如何解释结果。

**为什么 Phase 3 只实现 unipolar multiplier？**

这是一个最小但有研究意义的闭环：SC 中乘法可由两个独立 bitstream 的逐位 AND 实现，输出 1 的比例近似 `x × y`。先验证一个可解释操作，再扩展 bipolar、相关性控制和更多电路。

**为什么要求显式参数？**

如果论文参数抽取和模拟器同时引入，失败时无法定位责任。Phase 3 先隔离模拟器正确性，Phase 4 再加入参数抽取、证据对齐和有限补充循环。

**如何证明模拟器不是随机输出？**

固定 seed 后结果完全一致；增加 bitstream length 时，估计值应在统计意义上收敛到理论值。测试同时检查复现性和误差上限。

**如何防止 Agent 借模拟器执行任意代码？**

模拟器只接受 `SimulationSpec`，operation 必须在白名单内，数值和长度由 Pydantic 约束。没有 shell、eval 或模型生成代码入口。
