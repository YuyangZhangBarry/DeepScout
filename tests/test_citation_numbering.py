from backend.app.schemas_research import CitationEntry, KeyPoint, ResearchAnswerPayload
from backend.app.services.citation_numbering import apply_numbered_citations


def test_apply_numbered_citations_maps_s_ids_to_sequence() -> None:
    answer = ResearchAnswerPayload(
        executive_summary="A [s5] and bare s0.",
        key_points=[KeyPoint(text="T s7 end", source_ids=["s5", "s0"])],
        limitations="",
        report_markdown="See [s0].",
        citations=[
            CitationEntry(source_id="s0", url="http://a", title="A"),
            CitationEntry(source_id="s5", url="http://b", title="B"),
            CitationEntry(source_id="s7", url="http://c", title="C"),
        ],
    )
    out = apply_numbered_citations(answer)
    assert out.executive_summary == "A [2] and bare [1]."
    assert out.report_markdown == "See [1]."
    assert out.key_points[0].text == "T [3] end"
    assert out.key_points[0].source_ids == ["2", "1"]
    assert [c.citation_label for c in out.citations] == [1, 2, 3]
    assert [c.source_id for c in out.citations] == ["s0", "s5", "s7"]


def test_apply_numbered_citations_noop_when_empty() -> None:
    a = ResearchAnswerPayload(
        executive_summary="Plain.",
        key_points=[],
        limitations="",
        report_markdown="",
        citations=[],
    )
    assert apply_numbered_citations(a).model_dump() == a.model_dump()
