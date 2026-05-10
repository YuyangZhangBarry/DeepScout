"""Map evidence ``s0``…``sN`` markers in answers to sequential ``[1]``…``[n]`` for display."""

from __future__ import annotations

import re

from backend.app.schemas_research import CitationEntry, KeyPoint, ResearchAnswerPayload


def apply_numbered_citations(payload: ResearchAnswerPayload) -> ResearchAnswerPayload:
    """
    Assign ``citation_label`` 1..n on citations (list order) and rewrite body text / key_points
    so ``[s0]``, ``s3``, etc. become ``[1]``, ``[2]``… for the cited sources only.
    """
    citations = list(payload.citations)
    if not citations:
        return payload

    sid_to_label: dict[str, str] = {c.source_id: str(i) for i, c in enumerate(citations, start=1)}
    sid_to_int: dict[str, int] = {c.source_id: i for i, c in enumerate(citations, start=1)}

    def rewrite_text(text: str) -> str:
        if not text:
            return text
        out = text

        def bracket_repl(m: re.Match[str]) -> str:
            sid = f"s{m.group(1)}"
            lab = sid_to_label.get(sid)
            return f"[{lab}]" if lab is not None else m.group(0)

        out = re.sub(r"\[s(\d+)\]", bracket_repl, out, flags=re.IGNORECASE)

        def bare_repl(m: re.Match[str]) -> str:
            sid = f"s{m.group(1)}"
            lab = sid_to_label.get(sid)
            return f"[{lab}]" if lab is not None else m.group(0)

        out = re.sub(r"\bs(\d+)\b", bare_repl, out, flags=re.IGNORECASE)
        return out

    new_citations = [
        CitationEntry(
            source_id=c.source_id,
            citation_label=sid_to_int[c.source_id],
            paper_id=c.paper_id,
            url=c.url,
            title=c.title,
        )
        for c in citations
    ]

    new_key_points = [
        KeyPoint(
            text=rewrite_text(kp.text),
            source_ids=[sid_to_label[s] for s in kp.source_ids if s in sid_to_label],
        )
        for kp in payload.key_points
    ]

    return ResearchAnswerPayload(
        executive_summary=rewrite_text(payload.executive_summary),
        key_points=new_key_points,
        limitations=rewrite_text(payload.limitations),
        report_markdown=rewrite_text(payload.report_markdown),
        citations=new_citations,
    )
