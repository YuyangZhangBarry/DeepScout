import asyncio
import logging
import time
from typing import Iterable

import httpx

from backend.app.config import Settings, get_settings
from backend.app.schemas_tools import SearchHit
from backend.app.services.urlnorm import url_dedup_key

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

# Process-wide spacing for Semantic Scholar paper/search (key tier often ~1 req/s).
_s2_rate_lock = asyncio.Lock()
_s2_next_available_monotonic: float = 0.0


async def _semantic_scholar_get_throttled(
    client: httpx.AsyncClient,
    *,
    url: str,
    params: dict[str, str | int],
    headers: dict[str, str],
    timeout: float,
    settings: Settings,
) -> httpx.Response:
    global _s2_next_available_monotonic
    interval = max(0.0, settings.semantic_scholar_min_seconds_between_requests)
    async with _s2_rate_lock:
        if interval > 0:
            now = time.monotonic()
            wait_s = _s2_next_available_monotonic - now
            if wait_s > 0:
                await asyncio.sleep(wait_s)
        response = await client.get(url, params=params, headers=headers, timeout=timeout)
        if interval > 0:
            _s2_next_available_monotonic = time.monotonic() + interval
        return response


def _hit_dedup_key(hit: SearchHit) -> str:
    if hit.paper_id:
        return f"paper:{hit.paper_id}"
    return url_dedup_key(hit.url)


def _dedupe_hits_preserving_order(hits: Iterable[SearchHit]) -> list[SearchHit]:
    seen: set[str] = set()
    out: list[SearchHit] = []
    for h in hits:
        key = _hit_dedup_key(h)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


def _s2_paper_to_hit(item: dict, source_query: str) -> SearchHit | None:
    paper_id = (item.get("paperId") or "").strip()
    title = (item.get("title") or "").strip()
    abstract = (item.get("abstract") or "").strip()
    year = item.get("year")
    authors = item.get("authors") or []
    author_line = ""
    if isinstance(authors, list) and authors:
        names = []
        for a in authors[:12]:
            if isinstance(a, dict) and a.get("name"):
                names.append(str(a["name"]))
        author_line = ", ".join(names)
        if len(authors) > 12:
            author_line += ", et al."

    oa = item.get("openAccessPdf") if isinstance(item.get("openAccessPdf"), dict) else {}
    url = ""
    if isinstance(oa, dict):
        url = (oa.get("url") or "").strip()
    if not url:
        url = (item.get("url") or "").strip()
    if not url and paper_id:
        url = f"https://www.semanticscholar.org/paper/{paper_id}"
    if not url:
        return None

    meta_parts: list[str] = []
    if year is not None:
        meta_parts.append(str(year))
    if author_line:
        meta_parts.append(author_line)
    header = " · ".join(meta_parts)
    snippet = f"{header}\n\n{abstract}".strip() if header else abstract

    return SearchHit(
        url=url,
        title=title,
        snippet=snippet,
        source_query=source_query,
        paper_id=paper_id or None,
    )


async def _semantic_scholar_search_one(
    client: httpx.AsyncClient,
    *,
    query: str,
    max_results: int,
    settings: Settings,
) -> list[SearchHit]:
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,abstract,year,authors,url,openAccessPdf,paperId,externalIds",
    }
    headers: dict[str, str] = {"User-Agent": settings.http_user_agent}
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key

    response: httpx.Response | None = None
    max_attempts = 5
    for attempt in range(max_attempts):
        response = await _semantic_scholar_get_throttled(
            client,
            url=SEMANTIC_SCHOLAR_SEARCH_URL,
            params=params,
            headers=headers,
            timeout=60.0,
            settings=settings,
        )
        if response.status_code == 429 and attempt < max_attempts - 1:
            wait_s = min(30.0, 2.0 * (2**attempt))
            logger.info(
                "semantic scholar rate limited; retry in %.1fs (attempt %s/%s)",
                wait_s,
                attempt + 1,
                max_attempts,
            )
            await asyncio.sleep(wait_s)
            continue
        response.raise_for_status()
        break
    assert response is not None
    data = response.json()
    hits: list[SearchHit] = []
    for item in data.get("data") or []:
        if not isinstance(item, dict):
            continue
        hit = _s2_paper_to_hit(item, query)
        if hit is not None:
            hits.append(hit)
    return hits


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


async def _search_semantic_scholar(
    queries: list[str],
    *,
    max_results_per_query: int,
    settings: Settings,
    client: httpx.AsyncClient,
) -> list[SearchHit]:
    merged: list[SearchHit] = []
    if settings.semantic_scholar_parallel:
        tasks = [
            _semantic_scholar_search_one(
                client,
                query=q,
                max_results=max_results_per_query,
                settings=settings,
            )
            for q in queries
        ]
        batches = await asyncio.gather(*tasks, return_exceptions=True)
        for q, batch in zip(queries, batches):
            if isinstance(batch, BaseException):
                logger.warning("semantic scholar search failed for query=%r: %s", q, batch)
                continue
            merged.extend(batch)
        return _dedupe_hits_preserving_order(merged)

    delay = max(0.0, settings.semantic_scholar_inter_query_delay_seconds)
    for i, q in enumerate(queries):
        if i > 0 and delay > 0:
            await asyncio.sleep(delay)
        try:
            batch = await _semantic_scholar_search_one(
                client,
                query=q,
                max_results=max_results_per_query,
                settings=settings,
            )
            merged.extend(batch)
        except Exception as exc:  # noqa: BLE001 — one failed query should not abort others
            logger.warning("semantic scholar search failed for query=%r: %s", q, exc)
    return _dedupe_hits_preserving_order(merged)


async def _search_tavily(
    queries: list[str],
    *,
    max_results_per_query: int,
    settings: Settings,
    client: httpx.AsyncClient,
) -> list[SearchHit]:
    if not settings.tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY is not set (SEARCH_PROVIDER=tavily)")
    tasks = [
        _tavily_search_one(
            client,
            api_key=settings.tavily_api_key,
            query=q,
            max_results=max_results_per_query,
        )
        for q in queries
    ]
    batches = await asyncio.gather(*tasks, return_exceptions=True)
    merged: list[SearchHit] = []
    for q, batch in zip(queries, batches):
        if isinstance(batch, BaseException):
            logger.warning("tavily search failed for query=%r: %s", q, batch)
            continue
        merged.extend(batch)
    return _dedupe_hits_preserving_order(merged)


async def search_literature(
    queries: list[str],
    *,
    max_results_per_query: int | None = None,
    settings: Settings | None = None,
    client: httpx.AsyncClient | None = None,
) -> list[SearchHit]:
    """
    Parallel literature search, merge and de-duplicate.

    Default provider is Semantic Scholar (papers). Set SEARCH_PROVIDER=tavily for general web.
    """
    cfg = settings or get_settings()
    cleaned = [q.strip() for q in queries if q and q.strip()]
    if not cleaned:
        return []

    limit = max_results_per_query or cfg.search_max_results_per_query

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient()

    try:
        if cfg.search_provider == "tavily":
            return await _search_tavily(
                cleaned, max_results_per_query=limit, settings=cfg, client=client
            )
        return await _search_semantic_scholar(
            cleaned, max_results_per_query=limit, settings=cfg, client=client
        )
    finally:
        if own_client and client is not None:
            await client.aclose()


async def search_web(
    queries: list[str],
    *,
    max_results_per_query: int | None = None,
    settings: Settings | None = None,
    client: httpx.AsyncClient | None = None,
) -> list[SearchHit]:
    """Alias for :func:`search_literature` (backwards compatible)."""
    return await search_literature(
        queries,
        max_results_per_query=max_results_per_query,
        settings=settings,
        client=client,
    )
