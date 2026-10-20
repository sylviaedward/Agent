"""Deterministic Phase 1 classification and retrieval planning."""

import re
from uuid import uuid4

from .models import PlannedSubtask, ResearchRequest, TaskType


def classify_request(request: ResearchRequest) -> TaskType:
    """Route by observable task intent before any model call is introduced."""

    question = request.question.lower()
    if len(request.paper_aliases) >= 2 or re.search(r"比较|对比|compare|versus|\bvs\.?\b", question):
        return TaskType.COMPARE
    if re.search(r"综述|调研|研究进展|survey|literature review|review", question):
        return TaskType.REVIEW
    if re.search(r"计算|验证|复现|模拟|误差|rmse|simulate|simulation|reproduce|verify", question):
        return TaskType.COMPUTE
    return TaskType.FACT


def rewrite_query(request: ResearchRequest) -> str:
    """Apply only safe normalization; paper text never becomes an instruction."""

    normalized = " ".join(request.question.split())
    return normalized[:1000]


def build_plan(request: ResearchRequest, task_type: TaskType) -> list[PlannedSubtask]:
    query = rewrite_query(request)
    tasks = [PlannedSubtask(subtask_id=f"retrieve-{uuid4().hex[:8]}", purpose="global evidence", query=query)]
    if task_type == TaskType.COMPARE and request.paper_aliases:
        tasks.extend(
            PlannedSubtask(
                subtask_id=f"retrieve-{uuid4().hex[:8]}",
                purpose="explicit paper evidence",
                query=f"{alias} {query}",
                paper_alias=alias,
            )
            for alias in request.paper_aliases
        )
    return tasks
