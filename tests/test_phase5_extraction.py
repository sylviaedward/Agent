from sc_agent.models import ApprovalDecision, EvidenceItem
from sc_agent.phase5 import (
    HumanApprovalFlow,
    LLMParameterExtractor,
    OpenAICompatibleExtractionClient,
)


class FakeExtractorClient:
    def __init__(self, payload):
        self.payload = payload

    def complete(self, system_prompt, user_prompt):
        return self.payload


def test_openai_client_sets_identifying_user_agent(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"{}"}}]}'

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAICompatibleExtractionClient("https://gateway.example/v1", "secret", "model")
    assert client.complete("system", "user") == "{}"
    assert captured["request"].headers["User-agent"].startswith("sc-research-agent/")
    assert captured["request"].headers["Accept"] == "application/json"


def _evidence():
    return [
        EvidenceItem(
            span_id="s1",
            source_subtask_id="r1",
            text="bitstream length is 1000, x=30%, y=0.6; absolute error is 2 ms",
        )
    ]


def test_llm_candidates_are_normalized_and_source_bound():
    payload = {
        "parameters": [
            {"name": "bitstream_length", "value": 1000, "source_span_id": "s1", "quote": "bitstream length is 1000"},
            {"name": "x", "value": 30, "unit": "%", "source_span_id": "s1", "quote": "x=30%"},
            {"name": "y", "value": 0.6, "source_span_id": "s1", "quote": "y=0.6"},
            {"name": "metric", "value": "absolute error", "source_span_id": "s1", "quote": "absolute error"},
            {"name": "reported_value", "value": 2, "unit": "ms", "source_span_id": "s1", "quote": "absolute error is 2 ms"},
        ]
    }
    result = LLMParameterExtractor(FakeExtractorClient(payload)).extract(_evidence())
    assert result.status == "complete"
    by_name = {item.name: item for item in result.parameters}
    assert by_name["x"].normalized_value == 0.3
    assert by_name["reported_value"].normalized_value == 0.002
    assert all(item.source_span_id == "s1" for item in result.parameters)


def test_invalid_quote_and_unknown_unit_fail_closed():
    payload = {
        "parameters": [
            {"name": "x", "value": 0.3, "source_span_id": "s1", "quote": "x=0.9", "unit": "fraction"},
            {"name": "y", "value": 0.6, "source_span_id": "s1", "quote": "y=0.6", "unit": "bananas"},
        ]
    }
    result = LLMParameterExtractor(FakeExtractorClient(payload)).extract(_evidence())
    assert result.status == "conflict"
    assert any("quote" in item for item in result.conflicts)
    assert any("unsupported unit" in item for item in result.conflicts)


def test_condition_requires_human_approval():
    payload = {
        "parameters": [
            {"name": name, "value": value, "source_span_id": "s1", "quote": quote}
            for name, value, quote in [
                ("bitstream_length", 1000, "bitstream length is 1000"),
                ("x", 0.3, "x=30%"),
                ("y", 0.6, "y=0.6"),
                ("metric", "absolute error", "absolute error"),
                ("reported_value", 0.2, "absolute error is 2 ms"),
            ]
        ]
    }
    payload["parameters"][1]["condition"] = "under low correlation"
    extraction = LLMParameterExtractor(FakeExtractorClient(payload)).extract(_evidence())
    request = HumanApprovalFlow().prepare(extraction)
    assert request.status == "pending_human"
    approved = HumanApprovalFlow().decide(
        request, ApprovalDecision(status="approved", reviewer="researcher", comment="verified")
    )
    assert approved.status == "approved"
