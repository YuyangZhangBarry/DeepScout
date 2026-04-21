import logging
import re
from typing import NamedTuple

from pydantic import BaseModel, Field

from backend.app.config import Settings, get_settings
from backend.app.schemas_research import (
    CitationEntry,
    EvidenceSourceOut,
    KeyPoint,
    ResearchAnswerPayload,
    ResearchRequestBody,
    ResearchResponseBody,
)
from backend.app.services.fetch import fetch_urls
from backend.app.services.llm import (
    DeepseekClient,
    completion_message_text,
    parse_json_object_from_content,
)
from backend.app.services.search import search_literature
from backend.app.services.urlnorm import url_dedup_key

logger = logging.getLogger(__name__)


class _EvidenceRow(NamedTuple):
    source_id: str
    paper_id: str | None
    url: str
    title: str
    text: str


class _PlanningOut(BaseModel):
    queries: list[str] = Field(default_factory=list)


class _SynthOut(BaseModel):
    executive_summary: str = ""
    key_points: list[dict] = Field(default_factory=list)
    limitations: str = ""
    report_markdown: str = ""
    used_source_ids: list[str] = Field(default_factory=list)


def _clamp_queries(
    raw: list[str],
    *,
    question: str,
    lo: int,
    hi: int,
) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for q in raw:
        q = (q or "").strip()
        if len(q) < 2 or q in seen:
            continue
        seen.add(q)
        out.append(q)
        if len(out) >= hi:
            break
    if len(out) < lo:
        base = question.strip().replace("\n", " ")[:400]
        if not base:
            base = "literature survey"
        k = 0
        while len(out) < lo:
            candidate = f"{base} (subquery {k + 1})"
            k += 1
            if candidate in seen:
                candidate = f"{base} angle {k}"
            seen.add(candidate)
            out.append(candidate)
    return out[:hi]


async def _llm_plan_queries(
    client: DeepseekClient,
    *,
    question: str,
    min_q: int,
    max_q: int,
) -> list[str]:
    system = (
        "You plan academic literature search. Given the user's research question, "
        f"produce between {min_q} and {max_q} short English search queries for Semantic Scholar "
        "(keywords and topical phrases; avoid prose). "
        'Return JSON only with shape {"queries": ["...", "..."]}.'
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    data = await client.chat_completion(
        messages,
        temperature=0.35,
        max_tokens=700,
        response_format={"type": "json_object"},
        timeout=90.0,
    )
    text = completion_message_text(data)
    obj = parse_json_object_from_content(text)
    parsed = _PlanningOut.model_validate(obj)
    return _clamp_queries(parsed.queries, question=question, lo=min_q, hi=max_q)


def _trim_rows(rows: list[_EvidenceRow], max_chars: int) -> list[_EvidenceRow]:
    total = 0
    kept: list[_EvidenceRow] = []
    for row in rows:
        chunk = len(row.text) + 80
        if kept and total + chunk > max_chars:
            break
        kept.append(row)
        total += chunk
    return kept


def _collect_source_id_refs(text: str) -> set[str]:
    return set(re.findall(r"\bs\d+\b", text or ""))


async def _llm_synthesize(
    client: DeepseekClient,
    *,
    question: str,
    rows: list[_EvidenceRow],
    cfg: Settings,
) -> ResearchAnswerPayload:
    parts: list[str] = [f"Question:\n{question}\n\nSources:\n"]
    for row in rows:
        parts.append(
            f"\n---\nsource_id: {row.source_id}\n"
            f"paper_id: {row.paper_id or ''}\n"
            f"url: {row.url}\n"
            f"title: {row.title}\n"
            f"text:\n{row.text}\n"
        )
    user_content = "".join(parts)
    if len(user_content) > cfg.research_context_max_chars:
        user_content = user_content[: cfg.research_context_max_chars] + "\n...[truncated]\n"

    system = (
        "You synthesize academic literature for a research report. Use ONLY the Sources block. "
        "Do not invent papers, metrics, or citations that are not supported by the given text. "
        "Every substantive claim in executive_summary must be traceable to source_id values listed in Sources. "
        "Each key point must include at least one valid source_id from Sources. "
        "report_markdown should be a structured mini-report (### headings) and may use inline markers like [s0] "
        "that match source_id values.\n"
        "Return JSON only with keys: "
        "executive_summary (string), "
        "key_points (array of objects with text and source_ids), "
        "limitations (string), "
        "report_markdown (string), "
        "used_source_ids (array of strings like \"s0\" listing every source you relied on)."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    data = await client.chat_completion(
        messages,
        temperature=0.45,
        max_tokens=6000,
        response_format={"type": "json_object"},
        timeout=180.0,
    )
    text = completion_message_text(data)
    obj = parse_json_object_from_content(text)
    parsed = _SynthOut.model_validate(obj)

    id_set = {r.source_id for r in rows}

    key_points: list[KeyPoint] = []
    for item in parsed.key_points:
        if not isinstance(item, dict):
            continue
        t = str(item.get("text", "")).strip()
        if not t:
            continue
        sids = [str(s).strip() for s in (item.get("source_ids") or []) if str(s).strip() in id_set]
        key_points.append(KeyPoint(text=t, source_ids=sids))

    used: set[str] = {s for s in parsed.used_source_ids if s in id_set}
    for kp in key_points:
        used.update(kp.source_ids)
    used.update(_collect_source_id_refs(parsed.executive_summary))
    used.update(_collect_source_id_refs(parsed.report_markdown))

    by_id = {r.source_id: r for r in rows}
    citations: list[CitationEntry] = []
    seen_c: set[str] = set()
    for sid in sorted(used, key=lambda x: int(x[1:]) if len(x) > 1 and x[1:].isdigit() else 0):
        if sid not in by_id or sid in seen_c:
            continue
        seen_c.add(sid)
        r = by_id[sid]
        citations.append(
            CitationEntry(
                source_id=sid,
                paper_id=r.paper_id,
                url=r.url,
                title=r.title,
            )
        )

    return ResearchAnswerPayload(
        executive_summary=parsed.executive_summary.strip(),
        key_points=key_points,
        limitations=parsed.limitations.strip(),
        report_markdown=parsed.report_markdown.strip(),
        citations=citations,
    )


async def run_research_v0(
    body: ResearchRequestBody,
    *,
    settings: Settings | None = None,
) -> ResearchResponseBody:
    """
    End-to-end v0: plan queries → literature search → optional fetch → Deepseek JSON synthesis.
    """
    cfg = settings or get_settings()
    llm = DeepseekClient(cfg)

    max_sub = body.max_subqueries or cfg.research_max_subqueries
    max_sub = max(cfg.research_min_subqueries, min(max_sub, 12))
    min_sub = min(cfg.research_min_subqueries, max_sub)

    planning_queries = await _llm_plan_queries(
        llm,
        question=body.question,
        min_q=min_sub,
        max_q=max_sub,
    )

    max_papers = body.max_papers or cfg.research_max_papers
    hits = await search_literature(planning_queries, settings=cfg)
    hits = hits[:max_papers]

    max_fetch = body.max_fetch_urls if body.max_fetch_urls is not None else cfg.research_max_fetch_urls
    fetch_text_by_key: dict[str, str] = {}
    if max_fetch > 0 and hits:
        urls = [h.url for h in hits[:max_fetch]]
        docs = await fetch_urls(urls, settings=cfg)
        for d in docs:
            if d.error or not (d.text or "").strip():
                continue
            key = url_dedup_key(d.final_url or d.url)
            fetch_text_by_key[key] = (d.text or "")[:4000]

    rows: list[_EvidenceRow] = []
    for i, hit in enumerate(hits):
        sid = f"s{i}"
        text = (hit.snippet or "").strip()
        fk = url_dedup_key(hit.url)
        extra = fetch_text_by_key.get(fk)
        if extra:
            text = f"{text}\n\n### Page excerpt\n{extra}".strip()
        text = text[: cfg.research_excerpt_chars]
        rows.append(
            _EvidenceRow(
                source_id=sid,
                paper_id=hit.paper_id,
                url=hit.url,
                title=hit.title,
                text=text,
            )
        )

    rows = _trim_rows(rows, cfg.research_context_max_chars)

    if not rows:
        answer = ResearchAnswerPayload(
            executive_summary=(
                "No retrievable papers were found for this question. "
                "Try broader keywords, enable Semantic Scholar API key for higher rate limits, or switch provider."
            ),
            key_points=[],
            limitations="Empty evidence after search/fetch; refine the question or retry later.",
            report_markdown="# Report\n\n_No sources retrieved._\n",
            citations=[],
        )
        return ResearchResponseBody(
            question=body.question,
            planning_queries=planning_queries,
            evidence=[],
            answer=answer,
        )

    answer = await _llm_synthesize(llm, question=body.question, rows=rows, cfg=cfg)

    evidence_out = [
        EvidenceSourceOut(
            source_id=r.source_id,
            paper_id=r.paper_id,
            url=r.url,
            title=r.title,
            excerpt=r.text[:1500],
        )
        for r in rows
    ]

    return ResearchResponseBody(
        question=body.question,
        planning_queries=planning_queries,
        evidence=evidence_out,
        answer=answer,
    )
