from functools import lru_cache

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

    # Web search (Tavily: https://tavily.com/)
    tavily_api_key: str = ""
    search_max_results_per_query: int = 5

    # HTTP fetch
    http_user_agent: str = "DeepScout/0.1 (research bot; contact: local)"
    fetch_timeout_seconds: float = 25.0
    fetch_max_concurrent: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
