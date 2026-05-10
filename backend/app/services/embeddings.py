from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import httpx

from backend.app.config import Settings

logger = logging.getLogger(__name__)


def embedding_credentials_configured(settings: Settings) -> bool:
    """True when either embedding or OpenAI key is set (non-whitespace)."""
    return bool((settings.embedding_api_key or settings.openai_api_key or "").strip())


def embeddings_post_url(base_url: str) -> str:
    """
    Build OpenAI-compatible POST URL for embeddings.

    Accepts either ``https://api.openai.com`` or ``https://api.openai.com/v1`` so callers
    never end up with ``.../v1/v1/embeddings``.
    """
    b = (base_url or "").strip().rstrip("/")
    if not b:
        b = "https://api.openai.com"
    if b.endswith("/v1"):
        return f"{b}/embeddings"
    return f"{b}/v1/embeddings"


def _embedding_api_key(settings: Settings) -> str:
    key = (settings.embedding_api_key or settings.openai_api_key or "").strip()
    if not key:
        raise RuntimeError("EMBEDDING_API_KEY or OPENAI_API_KEY is required for RAG embeddings")
    return key


def _retry_after_seconds(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw.strip()))
    except ValueError:
        return None


async def _post_embeddings_json(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    *,
    max_retries: int,
    retry_base_seconds: float,
) -> dict[str, Any]:
    """POST /v1/embeddings with retries on 429 / 503."""
    last_response: httpx.Response | None = None
    for attempt in range(max_retries + 1):
        response = await client.post(url, headers=headers, json=payload, timeout=120.0)
        last_response = response
        if response.status_code in (429, 503) and attempt < max_retries:
            header_wait = _retry_after_seconds(response)
            backoff = min(
                120.0,
                header_wait
                if header_wait is not None
                else retry_base_seconds * (2**attempt) + random.uniform(0, 0.35),
            )
            logger.warning(
                "embedding HTTP %s, sleeping %.1fs then retry (%s/%s)",
                response.status_code,
                backoff,
                attempt + 1,
                max_retries,
            )
            await asyncio.sleep(backoff)
            continue
        response.raise_for_status()
        return response.json()
    if last_response is not None:
        last_response.raise_for_status()
    raise RuntimeError("embedding request failed with no response")


async def embed_texts(
    texts: list[str],
    *,
    settings: Settings,
    client: httpx.AsyncClient | None = None,
) -> list[list[float]]:
    """
    OpenAI-compatible POST /v1/embeddings. Returns one vector per input string (same order).

    Spaces consecutive batches by ``embedding_min_seconds_between_batches`` to stay under
    typical RPM limits, and retries on 429/503 with exponential backoff (honors Retry-After).
    """
    if not texts:
        return []
    url = embeddings_post_url(settings.embedding_base_url)
    headers = {
        "Authorization": f"Bearer {_embedding_api_key(settings)}",
        "Content-Type": "application/json",
    }
    batch = max(1, settings.embedding_batch_size)
    gap = max(0.0, float(settings.embedding_min_seconds_between_batches))
    max_retries = max(0, int(settings.embedding_max_retries))
    retry_base = max(0.05, float(settings.embedding_retry_base_seconds))

    own = client is None
    if own:
        client = httpx.AsyncClient()

    all_vecs: list[list[float]] = []
    try:
        for i in range(0, len(texts), batch):
            if i > 0 and gap > 0:
                await asyncio.sleep(gap)
            chunk = texts[i : i + batch]
            payload: dict[str, Any] = {
                "model": settings.embedding_model,
                "input": chunk,
            }
            data = await _post_embeddings_json(
                client,
                url,
                headers,
                payload,
                max_retries=max_retries,
                retry_base_seconds=retry_base,
            )
            items = sorted(data.get("data") or [], key=lambda x: x.get("index", 0))
            for item in items:
                emb = item.get("embedding")
                if not isinstance(emb, list):
                    raise RuntimeError("invalid embedding response")
                all_vecs.append([float(x) for x in emb])
        if len(all_vecs) != len(texts):
            raise RuntimeError("embedding count mismatch")
        return all_vecs
    finally:
        if own and client is not None:
            await client.aclose()
