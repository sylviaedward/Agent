from sc_agent.models import EvidenceItem, SimulationResult
from sc_agent.phase4 import EvidenceParameterExtractor, MetricAligner


def test_extraction_binds_values_to_source_spans() -> None:
    evidence = [
        EvidenceItem(
            span_id="s1",
            text="bitstream length = 1000; x = 0.3; y = 0.6; RMSE; reported value = 0.02",
            source_subtask_id="retrieve-1",
        )
    ]
    result = EvidenceParameterExtractor().extract(evidence)
    assert result.status == "complete"
    assert {item.source_span_id for item in result.parameters} == {"s1"}
    assert {item.name for item in result.parameters} == {
        "bitstream_length",
        "x",
        "y",
        "metric",
        "reported_value",
    }


def test_missing_and_conflicting_values_fail_closed() -> None:
    evidence = [
        EvidenceItem(span_id="s1", text="bitstream length = 1000; x = 0.3", source_subtask_id="a"),
        EvidenceItem(span_id="s2", text="x = 0.4", source_subtask_id="b"),
    ]
    result = EvidenceParameterExtractor().extract(evidence)
    assert result.status == "conflict"
    assert "y" in result.missing
    assert result.conflicts


def test_metric_aligner_rejects_unaligned_metrics() -> None:
    evidence = [
        EvidenceItem(
            span_id="s1",
            text="bitstream length = 1000; x = 0.3; y = 0.6; RMSE; reported value = 0.02",
            source_subtask_id="a",
        )
    ]
    extraction = EvidenceParameterExtractor().extract(evidence)
    simulation = SimulationResult(
        operation="unipolar_multiplier",
        length=1000,
        seed=1,
        x=0.3,
        y=0.6,
        expected=0.18,
        estimate=0.19,
        absolute_error=0.01,
        ones_x=300,
        ones_y=600,
        ones_output=190,
    )
    alignment = MetricAligner().align(extraction, simulation)
    assert alignment.status == "not_comparable"
    assert alignment.reason == "metric_not_supported"
