from backend.app.services.rag_chunk import chunk_text


def test_chunk_text_splits_and_overlaps() -> None:
    t = "x" * 200
    parts = chunk_text(t, chunk_size=50, overlap=10)
    assert len(parts) >= 3
    assert all(len(p) <= 80 for p in parts)
    assert "".join(parts).count("x") >= 200


def test_chunk_text_empty() -> None:
    assert chunk_text("", chunk_size=500, overlap=50) == []
