"""Runtime configuration for the research agent.

The configuration deliberately separates orchestration limits from model
credentials. Credentials are read from the environment and never serialized
into task state or traces.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    sc_rag_api_url: str = Field(default="http://127.0.0.1:8000", alias="SC_RAG_API_URL")
    request_timeout_seconds: float = Field(default=20.0, alias="SC_AGENT_REQUEST_TIMEOUT_SECONDS")
    max_steps: int = Field(default=12, alias="SC_AGENT_MAX_STEPS")
    max_retries: int = Field(default=2, alias="SC_AGENT_MAX_RETRIES")
    token_budget: int = Field(default=12_000, alias="SC_AGENT_TOKEN_BUDGET")
    workspace_root: str = Field(default="workspace", alias="SC_AGENT_WORKSPACE_ROOT")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-level settings; call ``cache_clear`` in tests."""

    return Settings()
