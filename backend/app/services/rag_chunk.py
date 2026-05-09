def chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping windows by character count.
    overlap must be < chunk_size for progress.
    """
    t = (text or "").strip()
    if not t:
        return []
    size = max(80, chunk_size)
    ov = max(0, min(overlap, size - 1))
    chunks: list[str] = []
    start = 0
    n = len(t)
    while start < n:
        end = min(n, start + size)
        chunks.append(t[start:end])
        if end >= n:
            break
        start = max(0, end - ov)
    return chunks
