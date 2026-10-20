import pytest

from sc_agent.graph import Phase2Graph
from sc_agent.models import ResearchRequest, SimulationSpec, TaskType
from sc_agent.simulator import StochasticSimulator


def test_unipolar_multiplier_is_reproducible_and_converges() -> None:
    spec = SimulationSpec(x=0.3, y=0.6, length=100_000, seed=11)
    first = StochasticSimulator().simulate(spec)
    second = StochasticSimulator().simulate(spec)
    assert first == second
    assert first.expected == pytest.approx(0.18)
    assert first.absolute_error < 0.01


def test_simulator_rejects_unknown_operations() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        StochasticSimulator().simulate(SimulationSpec(operation="arbitrary_code", x=0.2, y=0.3))


def test_compute_graph_runs_simulator_after_retrieval() -> None:
    class FakeClient:
        def search(self, query: str, *, top_k: int = 8):
            return {"hits": [{"chunk_id": "compute-1", "text": "multiplier evidence"}]}

    request = ResearchRequest(
        question="verify multiplier error by simulation",
        simulation=SimulationSpec(x=0.3, y=0.6, length=10_000, seed=3),
    )
    result = Phase2Graph(client=FakeClient()).invoke(request)
    assert result.task_type == TaskType.COMPUTE
    assert result.simulation is not None
    assert result.tool_runs[-1]["tool"] == "stochastic_simulator"
