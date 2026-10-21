import json

import pytest

from sc_agent.models import (
    ApprovalRequest,
    EvidenceItem,
    ExtractedParameter,
    Phase5Extraction,
    ResearchRequest,
)
from sc_agent.phase6 import Phase6Executor


def _extraction() -> Phase5Extraction:
    return Phase5Extraction(
        status="complete",
        parameters=[
            ExtractedParameter(name="bitstream_length", value=1000, normalized_value=1000, source_span_id="s1", quote="length=1000"),
            ExtractedParameter(name="x", value=0.3, normalized_value=0.3, source_span_id="s1", quote="x=0.3"),
            ExtractedParameter(name="y", value=0.6, normalized_value=0.6, source_span_id="s1", quote="y=0.6"),
        ],
    )


def test_pending_approval_blocks_simulation():
    result = Phase6Executor().execute(
        ResearchRequest(question="run experiment"),
        _extraction(),
        ApprovalRequest(status="pending_human", reasons=["condition"], parameters=_extraction().parameters),
    )
    assert result.status == "blocked"
    assert result.reason == "approval_required"
    assert result.simulation is None


def test_approved_parameters_execute_and_export(tmp_path):
    extraction = _extraction()
    result = Phase6Executor().execute(
        ResearchRequest(question="run experiment"),
        extraction,
        ApprovalRequest(status="approved", parameters=extraction.parameters),
        task_id="t-6",
    )
    assert result.status == "executed"
    assert result.simulation is not None
    evidence = [EvidenceItem(span_id="s1", source_subtask_id="r", alias="Demo", page=2, text="x=0.3 y=0.6 length=1000")]
    exported = Phase6Executor.export_report(result, evidence, tmp_path / "phase6.json")
    assert exported.report_path == str(tmp_path / "phase6.json")
    report = json.loads((tmp_path / "phase6.json").read_text(encoding="utf-8"))
    assert report["task_id"] == "t-6"
    assert report["evidence"][0]["quote"] == "length=1000"


def test_blocked_result_cannot_export(tmp_path):
    extraction = _extraction()
    result = Phase6Executor().execute(
        ResearchRequest(question="run experiment"), extraction, ApprovalRequest(status="rejected")
    )
    with pytest.raises(ValueError, match="only an executed"):
        Phase6Executor.export_report(result, [], tmp_path / "blocked.json")
