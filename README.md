# SC-Research-Agent

SC-Research-Agent is an independent, controlled Agent for reproducible
stochastic-computing research tasks. It uses the existing SC-RAG project as a
grounded evidence service and adds task routing, planning, tool execution,
simulation, recovery, human approval, and report export.

The first target workflow is:

```text
research question
  → classify and decompose
  → retrieve trusted paper evidence through SC-RAG
  → extract experiment parameters
  → run a constrained stochastic-computing simulation
  → compare results and explain discrepancies
  → request approval
  → export a cited research report
```

This repository is being developed in phases. Phase 0 establishes the stable
state and tool boundaries; later phases add LangGraph orchestration, simulation,
reflection, sandboxing, checkpoint recovery, tracing, and evaluation. See
[`docs/architecture.md`](docs/architecture.md) for the design contracts.

## Phase 0 setup

Python 3.11+ is required. From the repository root (usually
`/workspaces/Agent` in GitHub Codespaces, or `/workspace/Agent` in the cloud
onboarding environment):

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/pytest
```

The Phase 0 client expects the SC-RAG API at `http://127.0.0.1:8000`. Override
it with `SC_RAG_API_URL` when SC-RAG runs elsewhere. Configuration also exposes
bounded defaults for request timeout, graph steps, retries, token budget, and
the agent workspace. Secrets are intentionally not part of this repository.

The Agent and SC-RAG repositories are independent checkouts. A Codespace
created from this repository contains `/workspaces/Agent` only; it does not
automatically contain `/workspaces/SC-RAG`. Clone SC-RAG into the same
Codespace, or point `SC_RAG_API_URL` at an already running SC-RAG service.

To check the integration after SC-RAG is running:

```bash
.venv/bin/sc-agent check
.venv/bin/sc-agent check --query "stochastic computing multiplier"
```

## Phase 1: deterministic task execution

Phase 1 adds explainable routing and a bounded retrieval workflow. It classifies
fact, comparison, computation, and review requests, expands explicit paper
targets into separate retrieval subtasks, deduplicates trusted evidence, and
records tool failures. Run it with:

```bash
.venv/bin/sc-agent run "compare Sobol and P2LSG multiplier accuracy" \
  --paper Sobol --paper P2LSG
```

The phase guide, code mapping, workflow explanation, and interview questions
are in [`docs/phase1_interview.md`](docs/phase1_interview.md).

## Phase 2: LangGraph state machine

Install the optional graph dependency and run the explicit state-machine path:

```bash
.venv/bin/python -m pip install -e '.[agent,dev]'
.venv/bin/sc-agent graph \
  "compare Sobol and P2LSG multiplier accuracy" \
  --paper Sobol --paper P2LSG
```

The graph routes fact tasks through a single plan and comparison, computation,
and review tasks through a multi-task plan. It preserves the Phase 1 evidence
and failure contracts. See [`docs/phase2_interview.md`](docs/phase2_interview.md)
for the state transitions and interview preparation.

## Phase 3: constrained stochastic simulation

Compute requests can now route through a bounded, reproducible unipolar
multiplier simulator after retrieval:

```bash
.venv/bin/sc-agent graph \
  "verify stochastic multiplier error by simulation" \
  --x 0.3 --y 0.6 --length 100000 --seed 7
```

The simulator accepts only typed parameters and a whitelisted operation. See
[`docs/phase3_interview.md`](docs/phase3_interview.md) for the rationale,
workflow, and interview questions.

## Phase 4: evidence-bound parameter extraction

Phase 4 extracts only explicit values from retrieved evidence, records their
source spans, fails closed on missing or conflicting values, and compares
metrics only when their definitions align. Run it with:

```bash
.venv/bin/sc-agent phase4 \
  "verify stochastic multiplier error by simulation" \
  --x 0.3 --y 0.6 --length 100000 --seed 7
```

See [`docs/phase4_interview.md`](docs/phase4_interview.md) for the safety
rules and interview preparation.

## Phase 5: structured extraction and human approval

Phase 5 lets an LLM discover candidate parameters from natural-language evidence,
then validates source spans, quotes, units, ranges, and conditions before any
simulation or report export. Ambiguous candidates become `pending_human` and
require an explicit reviewer decision. See
[`docs/phase5_interview.md`](docs/phase5_interview.md) for the workflow and
interview questions.

With an OpenAI-compatible gateway configured:

```bash
.venv/bin/sc-agent phase5 \
  "extract stochastic multiplier parameters and compare reported error" \
  --x 0.3 --y 0.6 --length 100000 --seed 7
```

默认 LLM 超时为 120 秒，模型输入有字符上限；可用
`SC_AGENT_LLM_TIMEOUT_SECONDS`、`SC_AGENT_EXTRACTION_MAX_CHARS` 和
`SC_AGENT_EXTRACTION_MAX_CHARS_PER_SPAN` 调整。

LLM extraction remains behind the Phase 5 validation and approval boundary;
the agent never treats model output as executable authority.
