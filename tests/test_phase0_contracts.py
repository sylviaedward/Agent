from sc_agent.config import Settings
from sc_agent.models import ResearchRequest, ResearchState, TaskType
from sc_agent.phase1 import Phase1Executor
from sc_agent.planner import build_plan, classify_request
from sc_agent.sc_rag_client import ScRagClient


def test_settings_have_bounded_defaults() -> None:
    settings = Settings()
    assert settings.max_steps == 12
    assert settings.max_retries == 2
    assert settings.token_budget == 12_000


def test_research_state_is_serializable() -> None:
    state = ResearchState(task_id="t-1", request=ResearchRequest(question="compare Sobol and P2LSG"))
    state.task_type = TaskType.COMPARE
    payload = state.model_dump(mode="json")
    assert payload["task_type"] == "compare"
    assert payload["approval_status"] == "not_required"


def test_sc_rag_client_validates_tool_arguments() -> None:
    client = ScRagClient(Settings())
    try:
        client.search("   ")
    except ValueError as error:
        assert "empty" in str(error)
    else:
        raise AssertionError("empty queries must be rejected before network access")


def test_phase1_classifies_and_expands_explicit_comparison_targets() -> None:
    request = ResearchRequest(question="compare multiplier accuracy", paper_aliases=["Sobol", "P2LSG"])
    assert classify_request(request) == TaskType.COMPARE
    plan = build_plan(request, TaskType.COMPARE)
    assert len(plan) == 3
    assert {item.paper_alias for item in plan[1:]} == {"Sobol", "P2LSG"}


def test_phase1_deduplicates_retrieved_chunks_and_records_failures() -> None:
    class FakeClient:
        def search(self, query: str, *, top_k: int = 8):
            if "P2LSG" in query:
                raise RuntimeError("temporary upstream failure")
            return {"hits": [{"chunk_id": "c1", "alias": "Sobol", "page": 3, "text": "evidence"}]}

    result = Phase1Executor(client=FakeClient()).run(
        ResearchRequest(question="compare multiplier accuracy", paper_aliases=["Sobol", "P2LSG"])
    )
    assert result.status == "completed"
    assert [item.span_id for item in result.evidence] == ["c1"]
    assert result.tool_runs[-1]["status"] == "failed"
    assert result.unresolved_issues
