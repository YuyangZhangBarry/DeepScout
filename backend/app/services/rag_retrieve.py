from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Sequence
from typing import Any

from backend.app.config import Settings
from backend.app.services.embeddings import embed_texts
from backend.app.services.rag_chroma import _index_and_query_sync, build_chunk_corpus
from backend.app.services.rag_hybrid import (
    bm25_ranked_chunk_ids,
    build_blocks_from_chunk_ids,
    fuse_vector_and_bm25,
)

logger = logging.getLogger(__name__)


async def maybe_rag_context_blocks(
    rows: Sequence[Any],
    *,
    question: str,
    settings: Settings,
    top_k: int,
    hybrid_enabled: bool | None = None,
) -> list[dict[str, Any]] | None:
    """
    Chunk + embed + Chroma dense search; optionally BM25 + RRF hybrid, then top-k for synthesis.

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

    hybrid_on = settings.rag_hybrid_enabled if hybrid_enabled is None else hybrid_enabled

    try:
        doc_embeddings = await embed_texts(chunk_texts, settings=settings)
        query_embeddings = await embed_texts([question], settings=settings)
    except Exception as exc:
        logger.warning("[research] stage=rag embed failed: %s", exc)
        return None

    collection_name = f"r_{uuid.uuid4().hex[:24]}"
    n_chunks = len(ids)
    pool = min(max(settings.rag_hybrid_pool, top_k), n_chunks)
    vec_k = pool if hybrid_on else min(top_k, n_chunks)

    try:
        raw_hits = await asyncio.to_thread(
            _index_and_query_sync,
            persist_dir=settings.chroma_persist_directory,
            collection_name=collection_name,
            chunk_texts=chunk_texts,
            embeddings=doc_embeddings,
            metadatas=metadatas,
            ids=ids,
            query_embedding=query_embeddings[0],
            top_k=vec_k,
        )
    except Exception as exc:
        logger.warning("[research] stage=rag chroma failed: %s", exc)
        return None

    vector_ranked_ids = [h[0] for h in raw_hits if h[0]]
    dist_by_id = {h[0]: h[3] for h in raw_hits if h[0]}

    if hybrid_on:
        try:
            bm25_ids = bm25_ranked_chunk_ids(chunk_texts, ids, question, pool)
        except Exception as exc:
            logger.warning("[research] stage=rag bm25 failed, dense-only: %s", exc)
            bm25_ids = []
        fused_ids = fuse_vector_and_bm25(
            vector_ranked_ids=vector_ranked_ids,
            bm25_ranked_ids=bm25_ids,
            rrf_k=settings.rag_hybrid_rrf_k,
            top_k=top_k,
        )
        logger.info(
            "[research] stage=rag hybrid pool=%s vec_candidates=%s bm25_candidates=%s fused=%s rrf_k=%s",
            pool,
            len(vector_ranked_ids),
            len(bm25_ids),
            len(fused_ids),
            settings.rag_hybrid_rrf_k,
        )
        out = build_blocks_from_chunk_ids(
            fused_ids,
            chunk_texts=chunk_texts,
            metadatas=metadatas,
            ids=ids,
            vector_distance_by_id=dist_by_id,
        )
    else:
        out = build_blocks_from_chunk_ids(
            vector_ranked_ids[:top_k],
            chunk_texts=chunk_texts,
            metadatas=metadatas,
            ids=ids,
            vector_distance_by_id=dist_by_id,
        )
        logger.info("[research] stage=rag dense_only n_hits=%s", len(out))

    if not out:
        return None
    return out
