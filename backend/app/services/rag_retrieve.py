from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Sequence
from typing import Any

from backend.app.config import Settings
from backend.app.services.embeddings import embed_texts
from backend.app.services.rag_chroma import _index_and_query_sync, build_chunk_corpus

logger = logging.getLogger(__name__)


async def maybe_rag_context_blocks(
    rows: Sequence[Any],
    *,
    question: str,
    settings: Settings,
    top_k: int,
) -> list[dict[str, Any]] | None:
    """
    Chunk + embed + Chroma query for this request (ephemeral collection).

    Returns list of {source_id, paper_id, url, title, text, distance} for top chunks,
    or None to signal caller should fall back to full-row excerpts.
    """
    chunk_size = settings.rag_chunk_size
    overlap = settings.rag_chunk_overlap
    chunk_texts, metadatas, ids = build_chunk_corpus(
        rows=rows,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    if not chunk_texts:
        return None

    try:
        doc_embeddings = await embed_texts(chunk_texts, settings=settings)
        query_embeddings = await embed_texts([question], settings=settings)
    except Exception as exc:
        logger.warning("[research] stage=rag embed failed: %s", exc)
        return None

    collection_name = f"r_{uuid.uuid4().hex[:24]}"
    try:
        hits = await asyncio.to_thread(
            _index_and_query_sync,
            persist_dir=settings.chroma_persist_directory,
            collection_name=collection_name,
            chunk_texts=chunk_texts,
            embeddings=doc_embeddings,
            metadatas=metadatas,
            ids=ids,
            query_embedding=query_embeddings[0],
            top_k=top_k,
        )
    except Exception as exc:
        logger.warning("[research] stage=rag chroma failed: %s", exc)
        return None

    out: list[dict[str, Any]] = []
    for doc, meta, dist in hits:
        sid = str(meta.get("source_id") or "")
        if not sid:
            continue
        ci = meta.get("chunk_index", 0)
        title = str(meta.get("title") or "")
        text = f"(chunk {ci}) {doc}".strip()
        pid_raw = meta.get("paper_id")
        paper_id_val: str | None
        if isinstance(pid_raw, str) and pid_raw.strip():
            paper_id_val = pid_raw.strip()
        else:
            paper_id_val = None
        out.append(
            {
                "source_id": sid,
                "paper_id": paper_id_val,
                "url": str(meta.get("url") or ""),
                "title": title,
                "text": text,
                "distance": dist,
            }
        )
    if not out:
        return None
    return out
