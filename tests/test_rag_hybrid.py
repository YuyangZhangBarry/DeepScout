from backend.app.services.rag_hybrid import (
    bm25_ranked_chunk_ids,
    fuse_vector_and_bm25,
    reciprocal_rank_fusion,
    tokenize_for_bm25,
)


def test_tokenize_for_bm25() -> None:
    assert "hello" in tokenize_for_bm25("Hello, world 123")


def test_reciprocal_rank_fusion() -> None:
    merged = reciprocal_rank_fusion(
        [["a", "b", "c"], ["b", "d", "a"]],
        rrf_k=60,
        top_n=4,
    )
    assert "b" in merged[:2]
    assert len(merged) <= 4


def test_fuse_single_list() -> None:
    assert fuse_vector_and_bm25(
        vector_ranked_ids=["x", "y"],
        bm25_ranked_ids=[],
        rrf_k=60,
        top_k=2,
    ) == ["x", "y"]


def test_bm25_ranked_chunk_ids() -> None:
    texts = [
        "neural network attention mechanism transformer",
        "database sql index optimization btree",
        "cooking pasta tomato sauce recipe",
    ]
    ids = ["c0", "c1", "c2"]
    out = bm25_ranked_chunk_ids(texts, ids, "transformer attention neural", pool=2)
    assert out[0] == "c0"
