"""Human-approved execution and auditable report export for Phase 6."""

import json
from pathlib import Path
from typing import Protocol

from .models import (
    ApprovalRequest,
    EvidenceItem,
    Phase5Extraction,
    Phase6Result,
    ResearchRequest,
    SimulationResult,
    SimulationSpec,
)
from .simulator import StochasticSimulator


class Simulator(Protocol):
    def simulate(self, spec: SimulationSpec) -> SimulationResult: ...


class Phase6Executor:
    """Execute only an approved, complete extraction and export its evidence trail."""

    def __init__(self, simulator: Simulator | None = None) -> None:
        self.simulator = simulator or StochasticSimulator()

    @staticmethod
    def _blocked(
        request: ResearchRequest,
        extraction: Phase5Extraction,
        approval: ApprovalRequest,
        reason: str,
    ) -> Phase6Result:
        return Phase6Result(
            status="blocked",
            task_id="unexecuted",
            question=request.question,
            approval_status=approval.status,
            reason=reason,
            parameters=extraction.parameters,
            evidence_span_ids=[item.source_span_id for item in extraction.parameters],
        )

    def execute(
        self,
        request: ResearchRequest,
        extraction: Phase5Extraction,
        approval: ApprovalRequest,
        *,
        task_id: str = "phase6-task",
    ) -> Phase6Result:
        if approval.status not in {"approved", "auto_approved"}:
            return self._blocked(request, extraction, approval, "approval_required")
        if extraction.status != "complete":
            return self._blocked(request, extraction, approval, f"extraction_{extraction.status}")
        by_name = {item.name: item for item in extraction.parameters}
        missing = {name for name in ("bitstream_length", "x", "y") if name not in by_name}
        if missing:
            return self._blocked(request, extraction, approval, f"parameters_missing:{','.join(sorted(missing))}")
        try:
            length = int(float(by_name["bitstream_length"].normalized_value or by_name["bitstream_length"].value))
            x = float(by_name["x"].normalized_value or by_name["x"].value)
            y = float(by_name["y"].normalized_value or by_name["y"].value)
            spec = SimulationSpec(x=x, y=y, length=length, seed=7)
            simulation = self.simulator.simulate(spec)
        except (TypeError, ValueError) as error:
            return self._blocked(request, extraction, approval, f"invalid_execution_parameters:{error}")
        return Phase6Result(
            status="executed",
            task_id=task_id,
            question=request.question,
            approval_status=approval.status,
            simulation=simulation,
            parameters=extraction.parameters,
            evidence_span_ids=[item.source_span_id for item in extraction.parameters],
        )

    @staticmethod
    def export_report(result: Phase6Result, evidence: list[EvidenceItem], output: Path) -> Phase6Result:
        if result.status != "executed":
            raise ValueError("only an executed Phase 6 result can be exported")
        evidence_by_id = {item.span_id: item for item in evidence}
        report = {
            "task_id": result.task_id,
            "question": result.question,
            "approval_status": result.approval_status,
            "parameters": [item.model_dump(mode="json") for item in result.parameters],
            "simulation": result.simulation.model_dump(mode="json") if result.simulation else None,
            "evidence": [
                {
                    "span_id": span_id,
                    "alias": evidence_by_id[span_id].alias,
                    "page": evidence_by_id[span_id].page,
                    "quote": next(item.quote for item in result.parameters if item.source_span_id == span_id),
                }
                for span_id in result.evidence_span_ids
                if span_id in evidence_by_id
            ],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result.model_copy(update={"report_path": str(output)})
