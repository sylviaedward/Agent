# SC-Research-Agent architecture

SC-Research-Agent is a controlled task executor for stochastic-computing
research. It is not a second copy of the SC-RAG index. SC-RAG remains the
evidence service; this repository owns task classification, planning, tool
execution, recovery, approval, and reporting.

## Initial boundary

The Phase 0 boundary is intentionally small:

```text
user request → ResearchState → SC-RAG HTTP boundary
```

The state is serializable and has explicit limits for steps, retries, tokens,
and workspace location. Credentials are process configuration, never task
state. Paper text is evidence data and cannot become system instructions.

## Planned execution graph

```text
classify
  → rewrite_and_decompose
  → retrieve_evidence
  → extract_parameters
  → execute_simulation
  → compare_results
  → critic
  → repair_or_abstain
  → human_approval
  → export_report
```

The graph will be added incrementally. A fact task will terminate after
retrieval and grounding; a compute task will continue to parameter extraction
and simulation. This conditional path is the reason to use an Agent workflow
instead of a fixed RAG chain.

## Non-negotiable contracts

- Every external call has a timeout and a bounded retry policy.
- Every tool has a typed input/output schema and an allowlist entry.
- Simulation commands run in a restricted workspace with resource limits.
- Reports are drafts until a human approval event is recorded.
- Traces record decisions, tool status, latency, token usage, and failure class,
  never secrets or unrestricted paper text.
