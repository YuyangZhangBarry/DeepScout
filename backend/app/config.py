from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "DeepScout"
    debug: bool = False

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    # Literature search (default: Semantic Scholar Graph API)
    search_provider: str = "semantic_scholar"
    semantic_scholar_api_key: str = ""

    # Optional general web search (https://tavily.com/) when SEARCH_PROVIDER=tavily
    tavily_api_key: str = ""
    search_max_results_per_query: int = 5

    # HTTP fetch
    http_user_agent: str = "DeepScout/0.1 (research bot; contact: local)"
    fetch_timeout_seconds: float = 25.0
    fetch_max_concurrent: int = 5

    # Research v0 (planning + synthesis, no vector DB yet)
    research_max_subqueries: int = 8
    research_min_subqueries: int = 3
    research_max_papers: int = 14
    research_max_fetch_urls: int = 5
    research_excerpt_chars: int = 4500
    research_context_max_chars: int = 32000

    @field_validator("search_provider")
    @classmethod
    def validate_search_provider(cls, v: str) -> str:
        allowed = {"semantic_scholar", "tavily"}
        key = (v or "semantic_scholar").strip().lower()
        if key not in allowed:
            raise ValueError(f"search_provider must be one of {sorted(allowed)}")
        return key


@lru_cache
def get_settings() -> Settings:
    return Settings()
