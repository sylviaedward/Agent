"""Phase 2 LangGraph state machine built on the Phase 1 contracts."""

from typing import Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from .config import Settings, get_settings
from .models import EvidenceItem, Phase1Result, ResearchRequest, TaskType
from .planner import build_plan, classify_request
from .sc_rag_client import ScRagClient


class GraphState(TypedDict, total=False):
    task_id: str
    request: dict[str, Any]
    task_type: str
    plan: list[dict[str, Any]]
    queries: list[str]
    evidence: list[dict[str, Any]]
    tool_runs: list[dict[str, Any]]
    unresolved_issues: list[str]
    step_count: int
    max_steps: int
    status: str
    route: str


class Phase2Graph:
    """Compile and invoke a bounded graph with an injectable retrieval client."""

    def __init__(self, client: ScRagClient | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = client or ScRagClient(self.settings)
        self.graph = self._build().compile()

    def _build(self) -> StateGraph:
        workflow = StateGraph(GraphState)
        workflow.add_node("classify", self._classify)
        workflow.add_node("plan_single", self._plan_single)
        workflow.add_node("plan_multi", self._plan_multi)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("finalize", self._finalize)
        workflow.add_edge(START, "classify")
        workflow.add_conditional_edges(
            "classify",
            self._route,
            {"single": "plan_single", "multi": "plan_multi"},
        )
        workflow.add_edge("plan_single", "retrieve")
        workflow.add_edge("plan_multi", "retrieve")
        workflow.add_edge("retrieve", "finalize")
        workflow.add_edge("finalize", END)
        return workflow

    @staticmethod
    def _request(state: GraphState) -> ResearchRequest:
        return ResearchRequest.model_validate(state["request"])

    def _classify(self, state: GraphState) -> dict[str, Any]:
        task_type = classify_request(self._request(state))
        route = "single" if task_type == TaskType.FACT else "multi"
        return {
            "task_type": task_type.value,
            "route": route,
            "status": "classified",
            "step_count": state.get("step_count", 0) + 1,
        }

    @staticmethod
    def _route(state: GraphState) -> str:
        return state.get("route", "single")

    def _make_plan(self, state: GraphState) -> dict[str, Any]:
        request = self._request(state)
        task_type = TaskType(state["task_type"])
        plan = build_plan(request, task_type)
        serialized = [item.model_dump(mode="json") for item in plan]
        return {
            "plan": serialized,
            "queries": [item["query"] for item in serialized],
            "status": "planned",
            "step_count": state.get("step_count", 0) + 1,
        }

    def _plan_single(self, state: GraphState) -> dict[str, Any]:
        return self._make_plan(state)

    def _plan_multi(self, state: GraphState) -> dict[str, Any]:
        return self._make_plan(state)

    def _retrieve(self, state: GraphState) -> dict[str, Any]:
        evidence: list[dict[str, Any]] = []
        tool_runs: list[dict[str, Any]] = []
        unresolved: list[str] = []
        seen: set[str] = set()
        plan = state.get("plan", [])[: state.get("max_steps", self.settings.max_steps)]
        for item in plan:
            subtask_id = str(item["subtask_id"])
            try:
                response = self.client.search(str(item["query"]), top_k=8)
                hits = response.get("hits", [])
                if not isinstance(hits, list):
                    raise TypeError("SC-RAG response field 'hits' must be a list")
                for hit in hits:
                    if not isinstance(hit, dict):
                        continue
                    span_id = str(hit.get("chunk_id", "")).strip()
                    if not span_id or span_id in seen:
                        continue
                    seen.add(span_id)
                    evidence.append(
                        EvidenceItem(
                            span_id=span_id,
                            alias=hit.get("alias"),
                            title=hit.get("title"),
                            page=hit.get("page"),
                            text=str(hit.get("text", "")),
                            score=hit.get("score"),
                            source_subtask_id=subtask_id,
                        ).model_dump(mode="json")
                    )
                tool_runs.append({"subtask_id": subtask_id, "status": "succeeded", "hit_count": len(hits)})
            except Exception as error:  # noqa: BLE001 - graph records boundary failures
                unresolved.append(f"{subtask_id}: {type(error).__name__}: {error}")
                tool_runs.append({"subtask_id": subtask_id, "status": "failed", "error": type(error).__name__})
        return {
            "evidence": evidence,
            "tool_runs": tool_runs,
            "unresolved_issues": unresolved,
            "status": "retrieved",
            "step_count": state.get("step_count", 0) + len(tool_runs),
        }

    @staticmethod
    def _finalize(state: GraphState) -> dict[str, Any]:
        status = "completed" if state.get("evidence") else "needs_evidence"
        return {"status": status, "step_count": state.get("step_count", 0) + 1}

    def invoke(self, request: ResearchRequest, *, task_id: str | None = None) -> Phase1Result:
        initial: GraphState = {
            "task_id": task_id or f"task-{uuid4().hex}",
            "request": request.model_dump(mode="json"),
            "max_steps": self.settings.max_steps,
            "step_count": 0,
        }
        result = self.graph.invoke(initial)
        return Phase1Result(
            task_id=result["task_id"],
            task_type=TaskType(result["task_type"]),
            status=result["status"],
            queries=result.get("queries", []),
            evidence=[EvidenceItem.model_validate(item) for item in result.get("evidence", [])],
            tool_runs=result.get("tool_runs", []),
            unresolved_issues=result.get("unresolved_issues", []),
        )
