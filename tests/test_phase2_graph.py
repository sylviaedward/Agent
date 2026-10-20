from sc_agent.graph import Phase2Graph
from sc_agent.models import ResearchRequest, TaskType


class FakeClient:
    def search(self, query: str, *, top_k: int = 8):
        if "P2LSG" in query:
            raise RuntimeError("simulated timeout")
        return {"hits": [{"chunk_id": "chunk-1", "alias": "Sobol", "page": 3, "text": "supported"}]}


def test_graph_routes_compare_to_multi_plan_and_preserves_failure() -> None:
    result = Phase2Graph(client=FakeClient()).invoke(
        ResearchRequest(question="compare multiplier accuracy", paper_aliases=["Sobol", "P2LSG"]),
        task_id="graph-test",
    )
    assert result.task_id == "graph-test"
    assert result.task_type == TaskType.COMPARE
    assert len(result.queries) == 3
    assert len(result.evidence) == 1
    assert result.tool_runs[-1]["status"] == "failed"
    assert result.unresolved_issues


def test_graph_routes_fact_to_single_plan() -> None:
    result = Phase2Graph(client=FakeClient()).invoke(ResearchRequest(question="what is a stochastic multiplier"))
    assert result.task_type == TaskType.FACT
    assert len(result.queries) == 1
    assert result.status == "completed"
