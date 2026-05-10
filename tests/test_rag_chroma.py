"""Chroma RAG path: query ``include`` must stay compatible with installed chromadb."""

from __future__ import annotations

import tempfile

from backend.app.services.rag_chroma import _index_and_query_sync


def test_index_and_query_sync_returns_ids_and_documents() -> None:
    with tempfile.TemporaryDirectory() as d:
        out = _index_and_query_sync(
            persist_dir=d,
            collection_name="ds_test_rag_chr_1",
            chunk_texts=["alpha", "beta"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            metadatas=[{"i": 0}, {"i": 1}],
            ids=["c0", "c1"],
            query_embedding=[1.0, 0.0, 0.0],
            top_k=2,
        )
    assert len(out) == 2
    assert {row[0] for row in out} == {"c0", "c1"}
    assert {row[1] for row in out} == {"alpha", "beta"}
    assert all(isinstance(row[2], dict) for row in out)
    assert all(isinstance(row[3], float) for row in out)
