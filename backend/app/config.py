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
    # If non-empty, append `backend.*` logs here (UTF-8), e.g. ./data/logs/app.log (data/ is gitignored).
    log_file: str = ""

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    # Optional OpenAI key (e.g. same vendor); used as fallback for embeddings when EMBEDDING_API_KEY is empty
    openai_api_key: str = ""

    # Literature search. Comma-separated providers are supported for complementary retrieval.
    # Default: Semantic Scholar Graph API + arXiv official API.
    search_provider: str = "semantic_scholar,arxiv"
    semantic_scholar_api_key: str = ""
    # Anonymous S2 quota is tight; sequential + delay reduces 429 vs asyncio.gather burst.
    semantic_scholar_parallel: bool = False
    # Extra pause between sub-queries (in addition to min_seconds_between_requests).
    semantic_scholar_inter_query_delay_seconds: float = 0.0
    # Introductory S2 key tier is ~1 request/s across endpoints; enforce globally per process.
    semantic_scholar_min_seconds_between_requests: float = 1.0

    # Optional general web search (https://tavily.com/) when SEARCH_PROVIDER=tavily
    tavily_api_key: str = ""
    search_max_results_per_query: int = 5
    arxiv_min_seconds_between_requests: float = 3.0

    # HTTP fetch
    http_user_agent: str = "DeepScout/0.1 (research bot; contact: local)"
    fetch_timeout_seconds: float = 25.0
    fetch_max_concurrent: int = 5
    fetch_per_host_max_concurrent: int = 2
    fetch_per_host_delay_seconds: float = 0.25

    # Research v0 (planning + synthesis, no vector DB yet)
    research_max_subqueries: int = 8
    research_min_subqueries: int = 3
    research_max_papers: int = 14
    research_max_fetch_urls: int = 5
    research_excerpt_chars: int = 4500
    research_context_max_chars: int = 32000
    # Day 11–12: orchestration / resilience
    research_search_retry_on_empty: bool = True
    research_fetch_max_rounds: int = 2

    # RAG (Day 8–10): embeddings + Chroma per-request collection
    rag_enabled: bool = True
    chroma_persist_directory: str = "./data/chroma"
    embedding_api_key: str = ""  # if empty, falls back to openai_api_key
    # Root or .../v1 — code normalizes so the POST path is never .../v1/v1/embeddings
    embedding_base_url: str = "https://api.openai.com"
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 32
    # OpenAI embedding tiers are often ~100 RPM; space batches to reduce 429 bursts.
    embedding_min_seconds_between_batches: float = 0.65
    embedding_max_retries: int = 8
    embedding_retry_base_seconds: float = 1.0
    rag_chunk_size: int = 900
    rag_chunk_overlap: int = 120
    rag_top_k: int = 12
    # Hybrid retrieval: BM25 + vector (RRF fusion); pool = candidates per channel before merge
    rag_hybrid_enabled: bool = True
    rag_hybrid_pool: int = 32
    rag_hybrid_rrf_k: int = 60

    @field_validator("search_provider")
    @classmethod
    def validate_search_provider(cls, v: str) -> str:
        allowed = {"semantic_scholar", "arxiv", "tavily"}
        raw = (v or "semantic_scholar,arxiv").strip().lower()
        keys = [p.strip() for p in raw.split(",") if p.strip()]
        if not keys:
            keys = ["semantic_scholar", "arxiv"]
        invalid = [p for p in keys if p not in allowed]
        if invalid:
            raise ValueError(f"search_provider entries must be one of {sorted(allowed)}")
        if "tavily" in keys and len(keys) > 1:
            raise ValueError("tavily cannot be combined with academic providers")
        deduped = list(dict.fromkeys(keys))
        return ",".join(deduped)


@lru_cache
def get_settings() -> Settings:
    return Settings()
