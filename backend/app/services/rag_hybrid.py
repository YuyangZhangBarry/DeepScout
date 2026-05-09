"""
Hybrid retrieval: BM25 (lexical) + dense vectors, fused via Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import re
from typing import Any

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+", re.UNICODE)


def tokenize_for_bm25(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def bm25_ranked_chunk_ids(
    chunk_texts: list[str],
    ids: list[str],
    query: str,
    pool: int,
) -> list[str]:
    """Return chunk ids ordered by BM25 score (best first), length up to ``pool``."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "rank-bm25 is required for hybrid RAG; install with: pip install -r requirements.txt"
        ) from exc

    if not chunk_texts or not ids or len(chunk_texts) != len(ids):
        return []

    corpus = [tokenize_for_bm25(t) for t in chunk_texts]
    if all(not c for c in corpus):
        return ids[: min(pool, len(ids))]

    bm25 = BM25Okapi(corpus)
    q = tokenize_for_bm25(query)
    if not q:
        return ids[: min(pool, len(ids))]

    scores = bm25.get_scores(q)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    take = min(pool, len(order))
    return [ids[i] for i in order[:take]]


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    *,
    rrf_k: int,
    top_n: int,
) -> list[str]:
    """
    RRF: score(d) = sum_i 1 / (k + rank_i(d)); missing in a list contributes 0 for that list.
    ``ranked_lists``: each inner list is best-first unique chunk ids.
    """
    k = max(1, rrf_k)
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, cid in enumerate(lst, start=1):
            if not cid:
                continue
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    if not scores:
        return []
    merged = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return merged[: max(1, top_n)]


def fuse_vector_and_bm25(
    *,
    vector_ranked_ids: list[str],
    bm25_ranked_ids: list[str],
    rrf_k: int,
    top_k: int,
) -> list[str]:
    lists: list[list[str]] = []
    if vector_ranked_ids:
        lists.append(vector_ranked_ids)
    if bm25_ranked_ids:
        lists.append(bm25_ranked_ids)
    if not lists:
        return []
    if len(lists) == 1:
        return lists[0][:top_k]
    return reciprocal_rank_fusion(lists, rrf_k=rrf_k, top_n=top_k)


def build_blocks_from_chunk_ids(
    ordered_ids: list[str],
    *,
    chunk_texts: list[str],
    metadatas: list[dict[str, Any]],
    ids: list[str],
    vector_distance_by_id: dict[str, float],
) -> list[dict[str, Any]]:
    """Map fused chunk ids to synthesis-ready dicts (same shape as legacy RAG output)."""
    id_to_idx = {cid: i for i, cid in enumerate(ids)}
    out: list[dict[str, Any]] = []
    for cid in ordered_ids:
        idx = id_to_idx.get(cid)
        if idx is None:
            continue
        doc = chunk_texts[idx]
        meta = metadatas[idx] or {}
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
                "distance": float(vector_distance_by_id.get(cid, 0.0)),
            }
        )
    return out
