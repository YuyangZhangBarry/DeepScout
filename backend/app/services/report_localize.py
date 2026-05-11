"""Just-in-time localization for the research report.

The research pipeline (planning, search, RAG, synthesis) always runs in English
because the literature itself and Semantic Scholar/arXiv queries are English.
We only translate the final answer for the PDF, so retrieval quality is never
impacted by the user's display language.

Public surface:
    detect_user_language(question)         # "en" | "zh"
    get_pdf_labels(lang)                   # localized PDF section labels
    translate_research_answer(answer, ...) # async LLM-backed translator
    localize_research_response(result, ...)# detect + (maybe) translate
    render_localized_research_pdf(result)  # one-shot: localize then render
"""
from __future__ import annotations

import json
import logging
from typing import Final

from pydantic import BaseModel, Field

from backend.app.schemas_research import (
    KeyPoint,
    ResearchAnswerPayload,
    ResearchResponseBody,
)
from backend.app.services.llm import (
    DeepseekClient,
    completion_message_text,
    parse_json_object_from_content,
)

logger = logging.getLogger(__name__)


SUPPORTED_LANGS: Final[tuple[str, ...]] = ("en", "zh")


def detect_user_language(question: str) -> str:
    """Return ``"zh"`` if the question contains any CJK ideograph, else ``"en"``.

    Heuristic only: covers the common case (Chinese users asking in Chinese) and
    avoids pulling a heavy language-detect dependency. Anything outside the CJK
    Unified Ideographs (incl. Extension A) falls back to English so we never
    silently translate ambiguous input.
    """
    if not question:
        return "en"
    for ch in question:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF:
            return "zh"
        if 0x3400 <= code <= 0x4DBF:
            return "zh"
    return "en"


# Section labels rendered in the PDF. Keys mirror the English strings that
# ``research_result_to_markdown`` used to hard-code.
PDF_LABELS: Final[dict[str, dict[str, str]]] = {
    "en": {
        "report_title": "DeepScout Research Report",
        "question": "Question",
        "executive_summary": "Executive Summary",
        "key_points": "Key Points",
        "report": "Report",
        "limitations": "Limitations",
        "citations": "Citations",
        "no_key_points": "No key points returned.",
        "no_citations": "No citations.",
    },
    "zh": {
        "report_title": "DeepScout 研究报告",
        "question": "研究问题",
        "executive_summary": "摘要",
        "key_points": "关键要点",
        "report": "报告正文",
        "limitations": "局限性",
        "citations": "参考文献",
        "no_key_points": "未返回关键要点。",
        "no_citations": "无参考文献。",
    },
}


def get_pdf_labels(lang: str) -> dict[str, str]:
    """Return PDF section labels for ``lang``, falling back to English."""
    return PDF_LABELS.get(lang, PDF_LABELS["en"])


class _TranslationOut(BaseModel):
    executive_summary: str = ""
    key_points: list[str] = Field(default_factory=list)
    limitations: str = ""
    report_markdown: str = ""


_LANG_DISPLAY: Final[dict[str, str]] = {
    "zh": "Simplified Chinese (Mandarin, 简体中文)",
}


_TRANSLATE_SYSTEM_TMPL = (
    "You translate a synthesized research report from English to {lang}. "
    "Translate the meaning naturally and faithfully. Strict rules:\n"
    "- Keep ALL citation markers untouched, including numeric markers like [1], [12] "
    "and source-id markers like [s0], [s12].\n"
    "- Keep every URL unchanged.\n"
    "- Preserve markdown structure: headings (### ...), bullet lists, blockquotes.\n"
    "- Keep inline code spans `like_this` and fenced ``` blocks unchanged.\n"
    "- Do NOT introduce new citations, facts, or claims.\n"
    "Return JSON only with the same keys as the input: executive_summary (string), "
    "key_points (array of strings, same length and order as input), limitations (string), "
    "report_markdown (string). Each value must be the translated text only."
)


async def translate_research_answer(
    answer: ResearchAnswerPayload,
    *,
    target_lang: str,
    client: DeepseekClient | None = None,
) -> ResearchAnswerPayload:
    """Translate answer text fields into ``target_lang`` while preserving citations.

    Returns the original ``answer`` unchanged when ``target_lang == "en"``,
    when the language is unsupported, when every text field is empty, or when
    the LLM call/parse fails. The PDF will still render — just in English.
    Citations (``url``, ``title``, ``source_id``) are never modified.
    """
    if target_lang == "en":
        return answer

    lang_display = _LANG_DISPLAY.get(target_lang)
    if lang_display is None:
        logger.info("translate skipped: unsupported lang=%s", target_lang)
        return answer

    payload = {
        "executive_summary": answer.executive_summary,
        "key_points": [kp.text for kp in answer.key_points],
        "limitations": answer.limitations,
        "report_markdown": answer.report_markdown,
    }
    if not any(payload.values()):
        return answer

    cli = client or DeepseekClient()
    system = _TRANSLATE_SYSTEM_TMPL.format(lang=lang_display)
    user = json.dumps(payload, ensure_ascii=False)

    try:
        data = await cli.chat_completion(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=6000,
            response_format={"type": "json_object"},
            timeout=180.0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("translate LLM call failed (%s); keeping English report", exc)
        return answer

    try:
        parsed = _TranslationOut.model_validate(
            parse_json_object_from_content(completion_message_text(data))
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("translate response parse failed (%s); keeping English report", exc)
        return answer

    if len(parsed.key_points) == len(answer.key_points):
        translated_points = [
            KeyPoint(
                text=(translated_text or src.text),
                source_ids=list(src.source_ids),
            )
            for src, translated_text in zip(answer.key_points, parsed.key_points)
        ]
    else:
        logger.warning(
            "translate key_points count mismatch (got=%s want=%s); keeping originals",
            len(parsed.key_points),
            len(answer.key_points),
        )
        translated_points = [
            KeyPoint(text=kp.text, source_ids=list(kp.source_ids))
            for kp in answer.key_points
        ]

    return ResearchAnswerPayload(
        executive_summary=parsed.executive_summary or answer.executive_summary,
        key_points=translated_points,
        limitations=parsed.limitations or answer.limitations,
        report_markdown=parsed.report_markdown or answer.report_markdown,
        citations=list(answer.citations),
    )


async def localize_research_response(
    result: ResearchResponseBody,
    *,
    client: DeepseekClient | None = None,
) -> tuple[ResearchResponseBody, str]:
    """Detect ``result.question`` language and translate the answer when needed.

    Returns ``(possibly_translated_result, detected_lang)``. The detected
    language is also useful for picking localized section labels.
    """
    lang = detect_user_language(result.question)
    if lang == "en":
        return result, lang

    translated_answer = await translate_research_answer(
        result.answer, target_lang=lang, client=client
    )
    if translated_answer is result.answer:
        return result, lang
    new_result = result.model_copy(update={"answer": translated_answer})
    logger.info("PDF: localized answer to lang=%s", lang)
    return new_result, lang


async def render_localized_research_pdf(
    result: ResearchResponseBody,
    *,
    client: DeepseekClient | None = None,
) -> bytes:
    """End-to-end: detect language, translate if needed, render with localized labels."""
    from backend.app.services.report_pdf import render_research_pdf

    new_result, lang = await localize_research_response(result, client=client)
    return render_research_pdf(new_result, labels=get_pdf_labels(lang))
