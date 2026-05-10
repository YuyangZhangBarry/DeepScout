from backend.app.schemas_research import (
    CitationEntry,
    ResearchAnswerPayload,
    ResearchResponseBody,
)
from backend.app.services.report_pdf import (
    _fpdf_html_document,
    _render_pdf_fpdf2,
    _resolve_fpdf_font_set,
    render_research_pdf,
    research_pdf_body_html,
    research_result_to_markdown,
)


def test_research_pdf_body_html_internal_cite_links() -> None:
    result = ResearchResponseBody(
        question="Q?",
        planning_queries=["q"],
        evidence=[],
        answer=ResearchAnswerPayload(
            executive_summary="Intro [1] and [2](https://example.com/keep).",
            key_points=[],
            limitations="",
            report_markdown="",
            citations=[
                CitationEntry(
                    source_id="s0",
                    citation_label=1,
                    url="https://u1",
                    title="T1",
                ),
                CitationEntry(
                    source_id="s1",
                    citation_label=2,
                    url="https://u2",
                    title="T2",
                ),
            ],
        ),
    )
    html = research_pdf_body_html(result)
    main_part, _, tail = html.partition('id="section-citations"')
    assert 'href="#cite-1"' in main_part
    assert 'href="#cite-2"' not in main_part
    assert 'id="cite-1"' in tail
    assert 'id="cite-2"' in tail

    html_plain = research_pdf_body_html(result, internal_cite_links=False)
    assert "href=\"#cite-" not in html_plain
    assert 'id="cite-1"' not in html_plain


def test_research_report_pdf_starts_with_pdf_header() -> None:
    result = ResearchResponseBody(
        question="What is test?",
        planning_queries=["test"],
        evidence=[],
        answer=ResearchAnswerPayload(
            executive_summary="Summary.",
            key_points=[],
            limitations="None.",
            report_markdown="# Report\n\nBody.\n",
            citations=[],
        ),
    )
    markdown = research_result_to_markdown(result)
    pdf = render_research_pdf(result)
    assert "DeepScout Research Report" in markdown
    assert pdf.startswith(b"%PDF")
    assert b"%%EOF" in pdf


def test_fpdf_fallback_uses_bundled_dejavu_font() -> None:
    """fpdf2 path picks the in-repo DejaVu and emits a valid PDF (no garbled glyphs)."""
    fs = _resolve_fpdf_font_set()
    assert fs is not None, "bundled DejaVu Sans should resolve in CI/dev"
    assert fs.regular.name == "DejaVuSans.ttf"
    assert fs.bold is not None and fs.bold.name == "DejaVuSans-Bold.ttf"

    body_html = (
        "<h1>DeepScout</h1>"
        "<p>Hello DeepScout — diffusion <strong>compression</strong>.</p>"
    )
    pdf_bytes = _render_pdf_fpdf2(_fpdf_html_document(body_html), fs)
    assert pdf_bytes.startswith(b"%PDF")
    assert b"%%EOF" in pdf_bytes
    # DejaVu Sans is the embedded face when the bundled font is used.
    assert b"DejaVuSans" in pdf_bytes
