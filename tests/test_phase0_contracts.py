from sc_agent.config import Settings
from sc_agent.models import ResearchRequest, ResearchState, TaskType
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
