from backend.app.schemas_research import (
    CitationEntry,
    EvidenceSourceOut,
    KeyPoint,
    ResearchAnswerPayload,
)
from backend.app.services.research_v0 import _finalize_answer_against_evidence


def test_finalize_drops_unknown_source_ids_and_rewrites_citations() -> None:
    ev = [
        EvidenceSourceOut(
            source_id="s0",
            paper_id="p0",
            url="https://example.com/a",
            title="Paper A",
            excerpt="x",
        ),
        EvidenceSourceOut(
            source_id="s1",
            paper_id=None,
            url="https://example.com/b",
            title="Paper B",
            excerpt="y",
        ),
    ]
    answer = ResearchAnswerPayload(
        executive_summary="Uses [s0] and fake [s99].",
        key_points=[
            KeyPoint(text="K", source_ids=["s0", "s99"]),
        ],
        limitations="",
        report_markdown="See [s1].",
        citations=[
            CitationEntry(
                source_id="s99",
                paper_id="evil",
                url="https://evil.test",
                title="Fake",
            ),
            CitationEntry(
                source_id="s0",
                paper_id=None,
                url="ignored",
                title="ignored",
            ),
        ],
    )
    out = _finalize_answer_against_evidence(answer, ev)
    assert out.key_points[0].source_ids == ["1"]
    ids = {c.source_id for c in out.citations}
    assert ids == {"s0", "s1"}
    labels = [c.citation_label for c in out.citations]
    assert sorted(x for x in labels if x is not None) == [1, 2]
    s0 = next(c for c in out.citations if c.source_id == "s0")
    assert s0.citation_label == 1
    assert s0.url == "https://example.com/a"
    assert s0.title == "Paper A"
    assert s0.paper_id == "p0"
    assert out.executive_summary == "Uses [1] and fake [s99]."
