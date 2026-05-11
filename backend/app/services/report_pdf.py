from __future__ import annotations

import html
import logging
import re
import textwrap
from pathlib import Path
from typing import NamedTuple

import markdown
from backend.app.schemas_research import CitationEntry, ResearchResponseBody
from backend.app.services.report_localize import get_pdf_labels

logger = logging.getLogger(__name__)

# English defaults; localized variants live in report_localize.PDF_LABELS.
DEFAULT_LABELS: dict[str, str] = get_pdf_labels("en")

# --- Markdown source (shared by all renderers) --------------------------------


def research_result_to_markdown(
    result: ResearchResponseBody,
    *,
    labels: dict[str, str] | None = None,
) -> str:
    lbl = labels or DEFAULT_LABELS
    parts = [
        f"# {lbl['report_title']}",
        "",
        f"## {lbl['question']}",
        result.question,
        "",
        f"## {lbl['executive_summary']}",
        result.answer.executive_summary,
        "",
        f"## {lbl['key_points']}",
    ]
    if result.answer.key_points:
        for point in result.answer.key_points:
            refs = ""
            if point.source_ids:
                refs = " " + ", ".join(f"[{sid}]" for sid in point.source_ids)
            parts.append(f"- {point.text}{refs}")
    else:
        parts.append(f"- {lbl['no_key_points']}")

    if result.answer.report_markdown:
        parts.extend(["", f"## {lbl['report']}", result.answer.report_markdown])
    if result.answer.limitations:
        parts.extend(["", f"## {lbl['limitations']}", result.answer.limitations])

    parts.extend(["", f"## {lbl['citations']}"])
    if result.answer.citations:
        for citation in result.answer.citations:
            title = citation.title or citation.url
            label = citation.citation_label
            tag = str(label) if label is not None else citation.source_id
            parts.append(f"- [{tag}] {title} {citation.url}")
    else:
        parts.append(f"- {lbl['no_citations']}")

    return "\n".join(parts).strip() + "\n"


def markdown_to_html_fragment(md: str) -> str:
    """Convert report markdown to HTML (body fragment only)."""
    return markdown.markdown(
        md,
        extensions=[
            "markdown.extensions.extra",
            "markdown.extensions.tables",
            "markdown.extensions.fenced_code",
            "markdown.extensions.nl2br",
            "markdown.extensions.sane_lists",
        ],
    )


def _inject_pdf_inline_citation_links(md: str, citations: list[CitationEntry]) -> str:
    """Turn ``[n]`` into internal links to ``#cite-n`` for PDF (skip ``[n](url)`` markdown links)."""
    labels = sorted({c.citation_label for c in citations if c.citation_label}, reverse=True)
    out = md
    for n in labels:
        pattern = rf"\[{n}\](?!\()"
        out = re.sub(
            pattern,
            f'<a href="#cite-{n}" class="cite-ref">{n}</a>',
            out,
        )
    return out


def _html_citations_section_for_pdf(
    citations: list[CitationEntry],
    *,
    include_anchor_ids: bool = True,
    labels: dict[str, str] | None = None,
) -> str:
    """HTML references block; optional ``id="cite-n"`` for WeasyPrint internal links only."""
    lbl = labels or DEFAULT_LABELS
    heading_html = (
        f'<h2 id="section-citations">{html.escape(lbl["citations"])}</h2>'
    )
    if not citations:
        return (
            heading_html
            + f'<p class="no-cites">{html.escape(lbl["no_citations"])}</p>'
        )
    items: list[str] = []
    for c in citations:
        lab = c.citation_label or 0
        if lab < 1:
            continue
        title = html.escape(c.title or c.url or "")
        url = html.escape(c.url or "")
        title_disp = title if title else url
        lid = f' id="cite-{lab}"' if include_anchor_ids else ""
        items.append(
            f"<li{lid} class=\"ref-row\">"
            f'<span class="ref-num">[{lab}]</span> '
            f'<a class="ext" href="{url}" target="_blank" rel="noopener noreferrer">{title_disp}</a>'
            f'<div class="ref-url">{url}</div>'
            "</li>"
        )
    return heading_html + f'<ol class="refs pdf-refs">{"".join(items)}</ol>'


def research_pdf_body_html(
    result: ResearchResponseBody,
    *,
    internal_cite_links: bool = True,
    labels: dict[str, str] | None = None,
) -> str:
    """
    Markdown → HTML body for PDF: optional ``#cite-n`` anchors (WeasyPrint).
    fpdf2 does not support HTML fragment links; use ``internal_cite_links=False`` there.
    """
    lbl = labels or DEFAULT_LABELS
    md_full = research_result_to_markdown(result, labels=lbl)
    # Split on the localized Citations heading so the references block is re-rendered
    # with anchor ids/styling by _html_citations_section_for_pdf.
    head, sep, _tail = md_full.partition(f"\n## {lbl['citations']}\n")
    main_md = head if sep else md_full
    if internal_cite_links:
        main_md = _inject_pdf_inline_citation_links(main_md, result.answer.citations)
    body_main = markdown_to_html_fragment(main_md)
    cites_html = _html_citations_section_for_pdf(
        result.answer.citations,
        include_anchor_ids=internal_cite_links,
        labels=lbl,
    )
    return body_main + cites_html


# --- WeasyPrint (best layout; needs system Pango/Cairo per WeasyPrint docs) ---

# Bundled fonts are reused by both engines (see BUNDLED_FONT_DIR below) to avoid
# letting WeasyPrint pick a system CFF/OpenType font (e.g. PingFang on macOS).
# macOS Preview has long-standing issues rendering subsetted CFF (Type0 +
# /FontFile3) fonts, which surfaces as letters shifted to the wrong codepoints.
# Using DejaVu TrueType (Type0 + /FontFile2) keeps Preview happy.
_BUNDLED_FONT_DIR = Path(__file__).resolve().parent.parent / "fonts"
_BUNDLED_DEJAVU_FAMILY = "DeepScout Sans"
_CJK_FALLBACK_FAMILY = "DeepScout CJK"

# Candidate CJK TrueType files per weight. Order = preference. We deliberately
# avoid CFF/OpenType CJK fonts (e.g. PingFang.ttc, Hiragino Sans GB.ttc,
# NotoSansCJK *.otf) because macOS Preview misrenders subsetted CFF.
_CJK_TT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "400": (
        "/System/Library/Fonts/STHeiti Light.ttc",         # macOS, TrueType ttc
        "/System/Library/Fonts/Supplemental/Songti.ttc",   # macOS, TrueType ttc
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",    # Debian/Ubuntu, TrueType
        "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",      # Fedora/Alma
    ),
    "700": (
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",
    ),
}


def _first_existing(paths: tuple[str, ...]) -> Path | None:
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            return p
    return None


def _weasy_font_face_css() -> str:
    """``@font-face`` rules for the WeasyPrint engine.

    - ``DeepScout Sans``: bundled DejaVu TTFs (Latin / shared symbols).
    - ``DeepScout CJK``: system-installed Chinese TrueType (``.ttc``) when one
      is found. Embedding TrueType (not CFF) is what keeps macOS Preview from
      garbling glyphs after subsetting.
    """
    faces: list[str] = []
    sans_specs = (
        ("DejaVuSans.ttf", "400", "normal"),
        ("DejaVuSans-Bold.ttf", "700", "normal"),
        ("DejaVuSans-Oblique.ttf", "400", "italic"),
    )
    for filename, weight, style in sans_specs:
        path = _BUNDLED_FONT_DIR / filename
        if not path.is_file():
            continue
        faces.append(
            f"""@font-face {{
  font-family: "{_BUNDLED_DEJAVU_FAMILY}";
  src: url("{path.as_uri()}") format("truetype");
  font-weight: {weight};
  font-style: {style};
}}"""
        )

    for weight, candidates in _CJK_TT_CANDIDATES.items():
        path = _first_existing(candidates)
        if path is None:
            continue
        # .ttc is a TrueType collection. Pango (used by WeasyPrint) parses it
        # transparently. format("truetype-collection") hints fontTools.
        fmt = "truetype-collection" if path.suffix.lower() == ".ttc" else "truetype"
        faces.append(
            f"""@font-face {{
  font-family: "{_CJK_FALLBACK_FAMILY}";
  src: url("{path.as_uri()}") format("{fmt}");
  font-weight: {weight};
  font-style: normal;
}}"""
        )
    return "\n".join(faces)


WEASYPRINT_CSS = """
@page { size: A4; margin: 22mm 20mm 26mm 20mm; }
html { font-size: 11pt; }
body {
  /* CJK font is intentionally a TrueType family (.ttc); CFF/OpenType CJK such
     as PingFang causes macOS Preview to misrender subsetted glyphs. */
  font-family: "DeepScout Sans", "DeepScout CJK", "PingFang SC",
    "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans CJK SC", "Noto Sans",
    "DejaVu Sans", sans-serif;
  line-height: 1.55;
  color: #1e293b;
  max-width: 100%;
}
h1 {
  font-size: 1.65rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  border-bottom: 1px solid #e2e8f0;
  padding-bottom: 0.35em;
  margin: 0 0 0.75em 0;
  color: #0f172a;
}
h2 {
  font-size: 1.2rem;
  font-weight: 600;
  margin: 1.35em 0 0.5em 0;
  color: #334155;
}
p { margin: 0.55em 0; orphans: 3; widows: 3; }
ul, ol { margin: 0.5em 0 0.75em 1.1em; padding: 0; }
li { margin: 0.25em 0; }
a { color: #1d4ed8; text-decoration: none; }
code, pre {
  font-family: "DeepScout Sans", "DeepScout CJK", ui-monospace,
    "SFMono-Regular", Menlo, Consolas, monospace;
  font-size: 0.92em;
}
pre {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 0.65em 0.85em;
  overflow: hidden;
}
table {
  border-collapse: collapse;
  width: 100%;
  margin: 0.75em 0;
  font-size: 0.95em;
}
th, td { border: 1px solid #e2e8f0; padding: 0.35em 0.5em; text-align: left; vertical-align: top; }
th { background: #f1f5f9; font-weight: 600; }
blockquote {
  margin: 0.75em 0;
  padding: 0.35em 0 0.35em 0.9em;
  border-left: 3px solid #cbd5e1;
  color: #475569;
}
a.cite-ref { font-weight: 600; text-decoration: underline; }
ol.pdf-refs {
  list-style: none;
  margin: 0.5em 0 0.75em 0;
  padding-left: 0;
}
ol.pdf-refs li.ref-row {
  margin: 0.55em 0;
  padding: 0.35em 0 0.35em 0;
  border-bottom: 1px solid #e2e8f0;
}
.ref-num { font-weight: 600; margin-right: 0.35em; }
.ref-url { font-size: 0.88em; color: #64748b; word-break: break-all; margin-top: 0.2em; }
"""


def _html_document_for_weasy(body_html: str) -> str:
    font_face_css = _weasy_font_face_css()
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>DeepScout Research Report</title>
  <style>{font_face_css}
{WEASYPRINT_CSS}</style>
</head>
<body>
<article class="report">{body_html}</article>
</body>
</html>"""


def _try_render_pdf_weasyprint(html_document: str) -> bytes | None:
    try:
        from weasyprint import HTML
    except (ImportError, OSError) as exc:
        logger.info("weasyprint not available (%s); skipping HTML/CSS engine", exc)
        return None
    try:
        base = Path(__file__).resolve().parent
        pdf = HTML(string=html_document, base_url=str(base)).write_pdf()
    except Exception as exc:  # noqa: BLE001
        logger.warning("weasyprint render failed: %s", exc)
        return None
    cjk_path = _first_existing(_CJK_TT_CANDIDATES["400"])
    logger.info(
        "PDF: weasyprint render ok (sans=%s cjk=%s)",
        _BUNDLED_DEJAVU_FAMILY,
        cjk_path.name if cjk_path else "<none; PingFang/system fallback>",
    )
    return pdf


# --- fpdf2 + write_html (no Pango; needs a TTF with glyphs you use) ------------


BUNDLED_FONT_DIR = _BUNDLED_FONT_DIR


class _FpdfFontSet(NamedTuple):
    """Resolved TTF set for the fpdf2 fallback path."""

    regular: Path
    bold: Path | None
    italic: Path | None


def _bundled_font_set() -> _FpdfFontSet | None:
    regular = BUNDLED_FONT_DIR / "DejaVuSans.ttf"
    if not regular.is_file():
        return None
    bold = BUNDLED_FONT_DIR / "DejaVuSans-Bold.ttf"
    italic = BUNDLED_FONT_DIR / "DejaVuSans-Oblique.ttf"
    return _FpdfFontSet(
        regular=regular,
        bold=bold if bold.is_file() else None,
        italic=italic if italic.is_file() else None,
    )


def _system_font_sets() -> list[_FpdfFontSet]:
    """
    System fallbacks if the bundled DejaVu is missing. Each entry pairs a regular TTF
    with the matching Bold/Italic file when one exists side-by-side.
    """
    candidates: list[_FpdfFontSet] = []

    linux_dejavu = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if linux_dejavu.is_file():
        candidates.append(
            _FpdfFontSet(
                regular=linux_dejavu,
                bold=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
                italic=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"),
            )
        )

    liberation = Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf")
    if liberation.is_file():
        candidates.append(
            _FpdfFontSet(
                regular=liberation,
                bold=Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
                italic=Path("/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf"),
            )
        )

    # Arial Unicode is huge and has shown glyph/cmap issues with fpdf2.write_html;
    # accept it only as a last resort to avoid the legacy text-only PDF.
    for arial in (
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
        Path("C:/Windows/Fonts/arialuni.ttf"),
    ):
        if arial.is_file():
            candidates.append(_FpdfFontSet(regular=arial, bold=None, italic=None))

    return [
        _FpdfFontSet(
            regular=c.regular,
            bold=c.bold if c.bold and c.bold.is_file() else None,
            italic=c.italic if c.italic and c.italic.is_file() else None,
        )
        for c in candidates
    ]


def _resolve_fpdf_font_set() -> _FpdfFontSet | None:
    """Prefer the in-repo bundled DejaVu; fall back to common system fonts."""
    bundled = _bundled_font_set()
    if bundled is not None:
        return bundled
    for fs in _system_font_sets():
        return fs
    return None


# Minimal stylesheet for the fpdf2 path. Full WEASYPRINT_CSS uses features fpdf2
# does not understand (@page, letter-spacing, etc.) and was the source of the
# garbled-looking output when WeasyPrint was unavailable.
_FPDF_CSS = """
body { color: #1e293b; font-size: 11pt; }
h1 { font-size: 18pt; color: #0f172a; }
h2 { font-size: 13pt; color: #334155; }
h3 { font-size: 12pt; color: #334155; }
a { color: #1d4ed8; }
code, pre { color: #334155; }
blockquote { color: #475569; }
.ref-num { font-weight: 600; }
.ref-url { color: #64748b; font-size: 9pt; }
"""


def _fpdf_html_document(body_html: str) -> str:
    """Slim HTML for fpdf2.write_html (only style features fpdf2 reliably handles)."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>DeepScout Research Report</title>
  <style>{_FPDF_CSS}</style>
</head>
<body>
{body_html}
</body>
</html>"""


def _render_pdf_fpdf2(html_document: str, font_set: _FpdfFontSet) -> bytes:
    from io import BytesIO

    from fpdf import FPDF
    from fpdf.enums import TextEmphasis
    from fpdf.fonts import FontFace

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(left=18, top=18, right=18)
    pdf.add_page()
    pdf.add_font("Rep", "", str(font_set.regular))
    has_bold = font_set.bold is not None
    has_italic = font_set.italic is not None
    if has_bold:
        pdf.add_font("Rep", "B", str(font_set.bold))
    if has_italic:
        pdf.add_font("Rep", "I", str(font_set.italic))

    body = FontFace(family="Rep", size_pt=11, color=(30, 41, 59))
    tag_styles: dict[str, FontFace] = {
        "h1": FontFace(family="Rep", size_pt=18, color=(15, 23, 42)),
        "h2": FontFace(family="Rep", size_pt=13, color=(51, 65, 85)),
        "h3": FontFace(family="Rep", size_pt=12, color=(51, 65, 85)),
        "p": body,
        "ul": body,
        "ol": body,
        "li": body,
        "a": FontFace(family="Rep", size_pt=11, color=(29, 78, 216)),
        "code": FontFace(family="Rep", size_pt=9.5, color=(51, 65, 85)),
        "pre": FontFace(family="Rep", size_pt=9.5, color=(30, 41, 59)),
        "blockquote": FontFace(family="Rep", size_pt=10.5, color=(71, 85, 105)),
    }
    # Only register the synthetic-style FontFace when a real face is available,
    # otherwise fpdf2 cannot resolve the variant and renders as the regular face.
    if has_bold:
        tag_styles["strong"] = FontFace(family="Rep", emphasis=TextEmphasis.B, size_pt=11)
    if has_italic:
        tag_styles["em"] = FontFace(family="Rep", emphasis=TextEmphasis.I, size_pt=11)

    pdf.write_html(html_document, tag_styles=tag_styles)
    buf = BytesIO()
    pdf.output(buf)
    return buf.getvalue()


# --- Legacy minimal PDF (Latin-1 only; last resort) ---------------------------


def _pdf_text(text: str) -> str:
    cleaned = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.encode("latin-1", "replace").decode("latin-1")
    return cleaned.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _markdown_to_lines(markdown: str, *, width: int = 92) -> list[str]:
    lines: list[str] = []
    for raw in (markdown or "").splitlines():
        text = raw.strip()
        if not text:
            lines.append("")
            continue
        if text.startswith("#"):
            text = text.lstrip("#").strip()
        chunks = textwrap.wrap(text, width=width, replace_whitespace=False) or [""]
        lines.extend(chunks)
    return lines


def render_pdf_from_markdown(markdown: str) -> bytes:
    lines = _markdown_to_lines(markdown)
    page_width = 595
    page_height = 842
    margin_x = 50
    top_y = 790
    line_height = 14
    lines_per_page = 52

    pages: list[list[str]] = [
        lines[i : i + lines_per_page] for i in range(0, len(lines), lines_per_page)
    ] or [[]]

    objects: list[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")

    page_object_ids = [3 + i * 2 for i in range(len(pages))]
    kids = " ".join(f"{obj_id} 0 R" for obj_id in page_object_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode("ascii"))

    for idx, page_lines in enumerate(pages):
        page_obj_id = 3 + idx * 2
        content_obj_id = page_obj_id + 1
        page_obj = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_width} {page_height}] "
            f"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> "
            f"/Contents {content_obj_id} 0 R >>"
        )
        objects.append(page_obj.encode("ascii"))

        content_lines = [
            "BT",
            f"/F1 10 Tf {margin_x} {top_y} Td {line_height} TL",
        ]
        for line in page_lines:
            content_lines.append(f"({_pdf_text(line)}) Tj T*")
        content_lines.append("ET")
        stream = "\n".join(content_lines).encode("latin-1")
        stream_obj = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
            + stream
            + b"\nendstream"
        )
        objects.append(stream_obj)

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for obj_num, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{obj_num} 0 obj\n".encode("ascii"))
        out.extend(obj)
        out.extend(b"\nendobj\n")

    xref_start = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        out.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    out.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_start}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(out)


def render_research_pdf(
    result: ResearchResponseBody,
    *,
    labels: dict[str, str] | None = None,
) -> bytes:
    """Render research as PDF: prefer WeasyPrint (CSS), then fpdf2 + HTML, then legacy PDF.

    ``labels`` controls localized PDF section headings (Question, Executive
    Summary, ...). Use :func:`report_localize.get_pdf_labels` to obtain one,
    or call :func:`report_localize.render_localized_research_pdf` which
    auto-detects the user's language.
    """
    lbl = labels or DEFAULT_LABELS
    body_html = research_pdf_body_html(result, internal_cite_links=True, labels=lbl)

    pdf = _try_render_pdf_weasyprint(_html_document_for_weasy(body_html))
    if pdf:
        return pdf

    logger.info("PDF: WeasyPrint unavailable or failed; trying fpdf2 HTML (layout may be simpler than WeasyPrint).")
    body_html_fpdf = research_pdf_body_html(
        result, internal_cite_links=False, labels=lbl
    )
    font_set = _resolve_fpdf_font_set()
    if font_set is not None:
        logger.info(
            "PDF: fpdf2 using regular=%s bold=%s italic=%s",
            font_set.regular,
            font_set.bold,
            font_set.italic,
        )
        try:
            return _render_pdf_fpdf2(_fpdf_html_document(body_html_fpdf), font_set)
        except Exception as exc:  # noqa: BLE001
            logger.warning("fpdf2 HTML render failed: %s", exc)
    else:
        logger.info("no Unicode TTF found for fpdf2; using legacy PDF renderer")

    md = research_result_to_markdown(result, labels=lbl)
    return render_pdf_from_markdown(md)
