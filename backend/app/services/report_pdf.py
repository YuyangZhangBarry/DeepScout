from __future__ import annotations

import textwrap

from backend.app.schemas_research import ResearchResponseBody


def _pdf_text(text: str) -> str:
    cleaned = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    # The built-in Helvetica font is WinAnsi/Latin-1. Keep generation dependency-free
    # and replace unsupported glyphs instead of failing a completed research job.
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
            text = text.lstrip("#").strip().upper()
        chunks = textwrap.wrap(text, width=width, replace_whitespace=False) or [""]
        lines.extend(chunks)
    return lines


def research_result_to_markdown(result: ResearchResponseBody) -> str:
    parts = [
        "# DeepScout Research Report",
        "",
        "## Question",
        result.question,
        "",
        "## Executive Summary",
        result.answer.executive_summary,
        "",
        "## Key Points",
    ]
    if result.answer.key_points:
        for point in result.answer.key_points:
            refs = ", ".join(point.source_ids)
            suffix = f" [{refs}]" if refs else ""
            parts.append(f"- {point.text}{suffix}")
    else:
        parts.append("- No key points returned.")

    if result.answer.report_markdown:
        parts.extend(["", "## Report", result.answer.report_markdown])
    if result.answer.limitations:
        parts.extend(["", "## Limitations", result.answer.limitations])

    parts.extend(["", "## Citations"])
    if result.answer.citations:
        for citation in result.answer.citations:
            title = citation.title or citation.url
            parts.append(f"- [{citation.source_id}] {title} {citation.url}")
    else:
        parts.append("- No citations.")

    return "\n".join(parts).strip() + "\n"


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


def render_research_pdf(result: ResearchResponseBody) -> bytes:
    return render_pdf_from_markdown(research_result_to_markdown(result))
