"""Small typed boundary around the existing SC-RAG HTTP service."""

from typing import Any

import httpx

from .config import Settings, get_settings


class ScRagClient:
    """Call only read/answer endpoints exposed by SC-RAG.

    The client does not interpret paper text as instructions. Later graph nodes
    will validate and constrain the returned evidence before passing it to a
    model or tool.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def health(self) -> dict[str, Any]:
        response = httpx.get(
            f"{self.settings.sc_rag_api_url.rstrip('/')}/health",
            timeout=self.settings.request_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {"value": payload}

    def search(self, query: str, *, top_k: int = 8) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        response = httpx.get(
            f"{self.settings.sc_rag_api_url.rstrip('/')}/search",
            params={"q": query, "top_k": top_k},
            timeout=self.settings.request_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {"results": payload}
