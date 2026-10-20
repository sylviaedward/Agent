"""Phase 1 deterministic research task executor."""

from uuid import uuid4

from .config import Settings, get_settings
from .models import EvidenceItem, Phase1Result, ResearchRequest, ResearchState
from .planner import build_plan, classify_request
from .sc_rag_client import ScRagClient


class Phase1Executor:
    """Run bounded retrieval tasks without autonomous command execution."""

    def __init__(self, client: ScRagClient | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = client or ScRagClient(self.settings)

    def run(self, request: ResearchRequest) -> Phase1Result:
        state = ResearchState(task_id=f"task-{uuid4().hex}", request=request, status="running")
        state.task_type = classify_request(request)
        plan = build_plan(request, state.task_type)
        state.plan = [item.model_dump(mode="json") for item in plan]
        state.query_rewrites = [item.query for item in plan]

        evidence: list[EvidenceItem] = []
        seen: set[str] = set()
        unresolved: list[str] = []
        tool_runs: list[dict[str, object]] = []

        for item in plan[: self.settings.max_steps]:
            try:
                response = self.client.search(item.query, top_k=8)
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
                            source_subtask_id=item.subtask_id,
                        )
                    )
                tool_runs.append({"subtask_id": item.subtask_id, "status": "succeeded", "hit_count": len(hits)})
            except Exception as error:  # noqa: BLE001 - boundary records failure for later recovery
                unresolved.append(f"{item.subtask_id}: {type(error).__name__}: {error}")
                tool_runs.append({"subtask_id": item.subtask_id, "status": "failed", "error": type(error).__name__})

        state.tool_runs = tool_runs
        state.unresolved_issues = unresolved
        state.step_count = len(tool_runs)
        state.status = "completed" if evidence else "needs_evidence"
        return Phase1Result(
            task_id=state.task_id,
            task_type=state.task_type,
            status=state.status,
            queries=state.query_rewrites,
            evidence=evidence,
            tool_runs=tool_runs,
            unresolved_issues=unresolved,
        )
