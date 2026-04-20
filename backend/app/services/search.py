import asyncio
import logging
from typing import Iterable

import httpx

from backend.app.config import Settings, get_settings
from backend.app.schemas_tools import SearchHit
from backend.app.services.urlnorm import url_dedup_key

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def _dedupe_hits_preserving_order(hits: Iterable[SearchHit]) -> list[SearchHit]:
    seen: set[str] = set()
    out: list[SearchHit] = []
    for h in hits:
        key = url_dedup_key(h.url)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


async def _tavily_search_one(
    client: httpx.AsyncClient,
    *,
    api_key: str,
    query: str,
    max_results: int,
) -> list[SearchHit]:
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "include_answer": False,
        "search_depth": "basic",
    }
    response = await client.post(TAVILY_SEARCH_URL, json=payload, timeout=60.0)
    response.raise_for_status()
    data = response.json()
    hits: list[SearchHit] = []
    for item in data.get("results") or []:
        url = (item.get("url") or "").strip()
        if not url:
            continue
        hits.append(
            SearchHit(
                url=url,
                title=(item.get("title") or "").strip(),
                snippet=(item.get("content") or "").strip(),
                source_query=query,
            )
        )
    return hits


async def search_web(
    queries: list[str],
    *,
    max_results_per_query: int | None = None,
    settings: Settings | None = None,
    client: httpx.AsyncClient | None = None,
) -> list[SearchHit]:
    """
    Run web search for each query in parallel (Tavily), merge and de-duplicate by URL.
    """
    cfg = settings or get_settings()
    if not cfg.tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY is not set")

    cleaned = [q.strip() for q in queries if q and q.strip()]
    if not cleaned:
        return []

    limit = max_results_per_query or cfg.search_max_results_per_query

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient()

    try:
        tasks = [
            _tavily_search_one(
                client,
                api_key=cfg.tavily_api_key,
                query=q,
                max_results=limit,
            )
            for q in cleaned
        ]
        batches = await asyncio.gather(*tasks, return_exceptions=True)
        merged: list[SearchHit] = []
        for q, batch in zip(cleaned, batches):
            if isinstance(batch, BaseException):
                logger.warning("search failed for query=%r: %s", q, batch)
                continue
            merged.extend(batch)
        return _dedupe_hits_preserving_order(merged)
    finally:
        if own_client and client is not None:
            await client.aclose()
