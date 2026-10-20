"""Stable domain contracts shared by the graph, tools, and evaluators."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class TaskType(StrEnum):
    FACT = "fact"
    COMPARE = "compare"
    COMPUTE = "compute"
    REVIEW = "review"


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1)
    paper_aliases: list[str] = Field(default_factory=list)


class EvidenceRef(BaseModel):
    """An opaque citation returned by SC-RAG; text is never trusted as an instruction."""

    span_id: str
    alias: str | None = None
    page: int | None = None
    quote: str | None = None


class ResearchState(BaseModel):
    """Serializable state carried through future LangGraph nodes."""

    task_id: str
    request: ResearchRequest
    task_type: TaskType | None = None
    sub_tasks: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    extracted_parameters: dict[str, Any] = Field(default_factory=dict)
    tool_runs: list[dict[str, Any]] = Field(default_factory=list)
    unresolved_issues: list[str] = Field(default_factory=list)
    step_count: int = 0
    retry_count: int = 0
    token_usage: int = 0
    approval_status: str = "not_required"
    query_rewrites: list[str] = Field(default_factory=list)
    plan: list[dict[str, Any]] = Field(default_factory=list)
    status: str = "created"


class PlannedSubtask(BaseModel):
    """A bounded retrieval action produced before any future tool execution."""

    subtask_id: str
    purpose: str
    query: str = Field(min_length=1)
    paper_alias: str | None = None


class EvidenceItem(BaseModel):
    """Trusted retrieval metadata copied from an SC-RAG hit."""

    span_id: str
    alias: str | None = None
    title: str | None = None
    page: int | None = None
    text: str = ""
    score: float | None = None
    source_subtask_id: str


class Phase1Result(BaseModel):
    task_id: str
    task_type: TaskType
    status: str
    queries: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    tool_runs: list[dict[str, Any]] = Field(default_factory=list)
    unresolved_issues: list[str] = Field(default_factory=list)
