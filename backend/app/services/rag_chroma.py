from __future__ import annotations

import logging
import os
from typing import Any

from backend.app.services.rag_chunk import chunk_text

logger = logging.getLogger(__name__)


def _ensure_chroma_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _index_and_query_sync(
    *,
    persist_dir: str,
    collection_name: str,
    chunk_texts: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any]],
    ids: list[str],
    query_embedding: list[float],
    top_k: int,
) -> list[tuple[str, str, dict[str, Any], float]]:
    """
    Run Chroma in a blocking context (call via asyncio.to_thread).

    Returns list of (chunk_id, document, metadata, distance) in similarity order.
    """
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    _ensure_chroma_dir(persist_dir)
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    try:
        coll = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        coll.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunk_texts,
            metadatas=metadatas,
        )
        k = min(top_k, len(ids))
        if k <= 0:
            return []
        res = coll.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances", "ids"],
        )
        docs = (res.get("documents") or [[]])[0] or []
        metas = (res.get("metadatas") or [[]])[0] or []
        dists = (res.get("distances") or [[]])[0] or []
        id_rows = (res.get("ids") or [[]])[0] or []
        out: list[tuple[str, str, dict[str, Any], float]] = []
        for cid, doc, meta, dist in zip(id_rows, docs, metas, dists):
            if doc is None:
                continue
            meta = meta or {}
            chunk_id = str(cid) if cid is not None else ""
            out.append(
                (
                    chunk_id,
                    str(doc),
                    meta,
                    float(dist) if dist is not None else 0.0,
                )
            )
        return out
    finally:
        try:
            client.delete_collection(collection_name)
        except Exception as exc:  # noqa: BLE001
            logger.debug("chroma delete_collection %s: %s", collection_name, exc)


def build_chunk_corpus(
    *,
    rows: list[Any],
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """
    rows: objects with source_id, paper_id, url, title, text (e.g. _EvidenceRow).
    Returns (chunk_texts, metadatas, ids).
    """
    chunk_texts: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []
    n = 0
    for row in rows:
        raw = f"{row.title}\n\n{row.text}".strip()
        parts = chunk_text(raw, chunk_size=chunk_size, overlap=chunk_overlap)
        for ci, part in enumerate(parts):
            cid = f"c{n}"
            n += 1
            chunk_texts.append(part)
            metadatas.append(
                {
                    "source_id": row.source_id,
                    "paper_id": row.paper_id or "",
                    "url": row.url,
                    "title": row.title,
                    "chunk_index": int(ci),
                }
            )
            ids.append(cid)
    return chunk_texts, metadatas, ids
