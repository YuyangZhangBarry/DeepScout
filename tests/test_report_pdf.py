import re
import zlib

import pytest

from backend.app.schemas_research import (
    CitationEntry,
    ResearchAnswerPayload,
    ResearchResponseBody,
)
from backend.app.services.report_pdf import (
    _BUNDLED_DEJAVU_FAMILY,
    _fpdf_html_document,
    _html_document_for_weasy,
    _render_pdf_fpdf2,
    _resolve_fpdf_font_set,
    _try_render_pdf_weasyprint,
    _weasy_font_face_css,
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


def _decompress_pdf_streams(pdf: bytes) -> bytes:
    out = bytearray()
    i = 0
    while True:
        s = pdf.find(b"stream\n", i)
        if s < 0:
            break
        e = pdf.find(b"endstream", s)
        if e < 0:
            break
        blob = pdf[s + len(b"stream\n") : e].rstrip(b"\r\n ")
        try:
            out += zlib.decompress(blob)
            out += b"\n"
        except Exception:
            pass
        i = e + len(b"endstream")
    outside = re.sub(rb"stream\n.*?endstream", b"", pdf, flags=re.S)
    return bytes(out) + outside


def test_weasy_font_face_css_uses_bundled_dejavu() -> None:
    css = _weasy_font_face_css()
    assert f'font-family: "{_BUNDLED_DEJAVU_FAMILY}"' in css
    assert "DejaVuSans.ttf" in css
    assert "DejaVuSans-Bold.ttf" in css
    assert css.count("@font-face") >= 2
    assert _BUNDLED_DEJAVU_FAMILY in _html_document_for_weasy("<p>x</p>")


def test_weasyprint_chinese_body_uses_truetype_for_preview_compat() -> None:
    """Chinese body text must also embed as TrueType (/FontFile2), never CFF.

    Without a TrueType CJK ``@font-face`` cascade ahead of PingFang, WeasyPrint
    falls back to PingFang on macOS, which is OpenType-CFF and triggers the same
    macOS Preview glyph-shift bug as the original Latin regression.
    """
    weasy = pytest.importorskip("weasyprint")
    _ = weasy  # noqa: F841

    body_html = research_pdf_body_html(
        ResearchResponseBody(
            question="扩散模型在图像压缩上的优势是什么?",
            planning_queries=["q"],
            evidence=[],
            answer=ResearchAnswerPayload(
                executive_summary="扩散模型在低比特率下感知质量优秀 [1]。",
                key_points=[],
                limitations="采样较慢。",
                report_markdown="### 背景\n扩散模型在图像压缩上表现良好 [1]。\n",
                citations=[
                    CitationEntry(
                        source_id="s1",
                        citation_label=1,
                        url="https://example.com/a",
                        title="Paper A",
                    ),
                ],
            ),
        )
    )
    pdf = _try_render_pdf_weasyprint(_html_document_for_weasy(body_html))
    if pdf is None:
        pytest.skip("WeasyPrint unavailable in this environment")

    assert pdf.startswith(b"%PDF")
    inflated = _decompress_pdf_streams(pdf)
    # Some Chinese glyphs must have been embedded (proves CJK shaping happened).
    base_fonts = re.findall(rb"/BaseFont\s*/[A-Za-z0-9+\-,.]+", inflated)
    assert base_fonts, "expected at least one embedded font"
    assert b"/FontFile2" in inflated, "expected TrueType subset for Chinese run"
    assert b"/FontFile3" not in inflated, (
        "CFF subset (e.g. PingFang) breaks macOS Preview; "
        "ensure a TrueType CJK @font-face (STHeiti/wqy/...) is in the cascade"
    )


def test_weasyprint_emits_truetype_fonts_for_preview_compat() -> None:
    """Latin body text must end up as TrueType (/FontFile2) — never CFF (/FontFile3).

    Subsetted CFF + Type0 fonts trigger a long-standing rendering bug in macOS
    Preview where letters appear shifted. Pinning the body font to bundled DejaVu
    via ``@font-face`` keeps the embedded subset TrueType.
    """
    weasy = pytest.importorskip("weasyprint")
    _ = weasy  # noqa: F841

    body_html = research_pdf_body_html(
        ResearchResponseBody(
            question="Diffusion image compression?",
            planning_queries=["q"],
            evidence=[],
            answer=ResearchAnswerPayload(
                executive_summary="Diffusion compresses well [1].",
                key_points=[],
                limitations="",
                report_markdown="## Body\nSome text [1].\n",
                citations=[
                    CitationEntry(
                        source_id="s1",
                        citation_label=1,
                        url="https://example.com/a",
                        title="Paper A",
                    ),
                ],
            ),
        )
    )
    pdf = _try_render_pdf_weasyprint(_html_document_for_weasy(body_html))
    if pdf is None:
        pytest.skip("WeasyPrint unavailable in this environment")

    assert pdf.startswith(b"%PDF")
    inflated = _decompress_pdf_streams(pdf)
    assert b"/FontFile2" in inflated, "expected TrueType subset for Preview compat"
    assert b"/FontFile3" not in inflated, "CFF subset breaks macOS Preview"
    assert b"DeepScout-Sans" in inflated or b"DeepScout Sans" in inflated


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
