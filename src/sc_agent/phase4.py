"""Evidence-bound parameter extraction and metric alignment for Phase 4."""

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import ClassVar

from .graph import Phase2Graph
from .models import (
    EvidenceItem,
    ExtractedParameter,
    MetricAlignment,
    ParameterExtraction,
    Phase1Result,
    ResearchRequest,
    SimulationResult,
)

_NUMBER = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_PATTERNS: dict[str, re.Pattern[str]] = {
    "bitstream_length": re.compile(r"(?:bitstream|bit[- ]?stream|stream)\s*(?:length|size)\s*(?:is|=|:)\s*(\d+)", re.IGNORECASE),
    "x": re.compile(rf"\bx\s*(?:is|=|:)\s*({_NUMBER})\b", re.IGNORECASE),
    "y": re.compile(rf"\by\s*(?:is|=|:)\s*({_NUMBER})\b", re.IGNORECASE),
    "metric": re.compile(r"\b(RMSE|MSE|MAE|absolute error|relative error|accuracy)\b", re.IGNORECASE),
    "reported_value": re.compile(rf"(?:reported|measured|observed|result)\s*(?:value|error)?\s*(?:is|=|:)\s*({_NUMBER})", re.IGNORECASE),
}


class EvidenceParameterExtractor:
    """Extract only explicit lexical values and bind each one to a span."""

    required = ("bitstream_length", "x", "y", "metric", "reported_value")

    def extract(self, evidence: Iterable[EvidenceItem]) -> ParameterExtraction:
        candidates: dict[str, list[ExtractedParameter]] = defaultdict(list)
        for item in evidence:
            for name, pattern in _PATTERNS.items():
                for match in pattern.finditer(item.text):
                    raw = match.group(1)
                    value: float | int | str
                    if name == "metric":
                        value = raw.lower()
                    elif name == "bitstream_length":
                        value = int(raw)
                    else:
                        value = float(raw)
                    quote = item.text[max(0, match.start() - 80) : min(len(item.text), match.end() + 80)]
                    candidates[name].append(
                        ExtractedParameter(name=name, value=value, source_span_id=item.span_id, quote=quote)
                    )

        conflicts: list[str] = []
        selected: list[ExtractedParameter] = []
        for name, values in candidates.items():
            distinct = {str(item.value) for item in values}
            if len(distinct) > 1:
                conflicts.append(f"{name}: conflicting values {sorted(distinct)}")
            else:
                selected.append(values[0])
        missing = [name for name in self.required if name not in candidates]
        status = "conflict" if conflicts else ("missing" if missing else "complete")
        return ParameterExtraction(status=status, parameters=selected, missing=missing, conflicts=conflicts)


class MetricAligner:
    """Compare only identical canonical metrics; never silently convert units."""

    _SIMULATION_FIELDS: ClassVar[dict[str, str]] = {
        "estimate": "estimate",
        "absolute error": "absolute_error",
        "relative error": "relative_error",
    }

    def align(self, extraction: ParameterExtraction, simulation: SimulationResult | None) -> MetricAlignment:
        if extraction.status != "complete":
            return MetricAlignment(status="not_comparable", reason=f"parameter_extraction_{extraction.status}")
        if simulation is None:
            return MetricAlignment(status="not_comparable", reason="simulation_result_missing")
        by_name = {item.name: item for item in extraction.parameters}
        metric = str(by_name["metric"].value).lower()
        field_name = self._SIMULATION_FIELDS.get(metric)
        if field_name is None:
            return MetricAlignment(status="not_comparable", metric_name=metric, reason="metric_not_supported")
        reported = float(by_name["reported_value"].value)
        simulated = getattr(simulation, field_name)
        if simulated is None:
            return MetricAlignment(status="not_comparable", metric_name=metric, reason="simulation_metric_missing")
        return MetricAlignment(
            status="comparable",
            metric_name=metric,
            simulated_value=float(simulated),
            reported_value=reported,
            delta=float(simulated) - reported,
        )


class Phase4Workflow:
    """Run Phase 2/3 and attach evidence-bound extraction and alignment."""

    def __init__(self, graph: Phase2Graph | None = None) -> None:
        self.graph = graph or Phase2Graph()
        self.extractor = EvidenceParameterExtractor()
        self.aligner = MetricAligner()

    def run(self, request: ResearchRequest) -> Phase1Result:
        result = self.graph.invoke(request)
        extraction = self.extractor.extract(result.evidence)
        alignment = self.aligner.align(extraction, result.simulation)
        return result.model_copy(update={"parameter_extraction": extraction, "metric_alignment": alignment})
