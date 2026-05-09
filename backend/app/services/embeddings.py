import logging
from typing import Any

import httpx

from backend.app.config import Settings

logger = logging.getLogger(__name__)


def embedding_credentials_configured(settings: Settings) -> bool:
    """True when either embedding or OpenAI key is set (non-whitespace)."""
    return bool((settings.embedding_api_key or settings.openai_api_key or "").strip())


def _embedding_api_key(settings: Settings) -> str:
    key = (settings.embedding_api_key or settings.openai_api_key or "").strip()
    if not key:
        raise RuntimeError("EMBEDDING_API_KEY or OPENAI_API_KEY is required for RAG embeddings")
    return key


async def embed_texts(
    texts: list[str],
    *,
    settings: Settings,
    client: httpx.AsyncClient | None = None,
) -> list[list[float]]:
    """
    OpenAI-compatible POST /v1/embeddings. Returns one vector per input string (same order).
    """
    if not texts:
        return []
    base = settings.embedding_base_url.rstrip("/")
    url = f"{base}/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {_embedding_api_key(settings)}",
        "Content-Type": "application/json",
    }
    batch = max(1, settings.embedding_batch_size)
    own = client is None
    if own:
        client = httpx.AsyncClient()

    all_vecs: list[list[float]] = []
    try:
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            payload: dict[str, Any] = {
                "model": settings.embedding_model,
                "input": chunk,
            }
            response = await client.post(url, headers=headers, json=payload, timeout=120.0)
            response.raise_for_status()
            data = response.json()
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
