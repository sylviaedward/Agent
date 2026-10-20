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

Python 3.11+ is required. From the repository root:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/pytest
```

The Phase 0 client expects the SC-RAG API at `http://127.0.0.1:8000`. Override
it with `SC_RAG_API_URL` when SC-RAG runs elsewhere. Configuration also exposes
bounded defaults for request timeout, graph steps, retries, token budget, and
the agent workspace. Secrets are intentionally not part of this repository.

To check the integration after SC-RAG is running:

```bash
.venv/bin/sc-agent check
.venv/bin/sc-agent check --query "stochastic computing multiplier"
```

Phase 0 does not yet call an LLM or execute arbitrary commands. That is
deliberate: the first acceptance gate is a tested, inspectable boundary before
we add autonomous behavior.
