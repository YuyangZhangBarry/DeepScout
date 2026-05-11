"""Tests for the just-in-time PDF localization pipeline.

We never hit a real LLM here: ``DeepseekClient.chat_completion`` is replaced by
an in-memory fake that returns whatever the test scripts. Async helpers are
exercised through ``asyncio.run`` to avoid adding a pytest-asyncio dependency.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from backend.app.schemas_research import (
    CitationEntry,
    KeyPoint,
    ResearchAnswerPayload,
    ResearchResponseBody,
)
from backend.app.services import report_localize
from backend.app.services.report_localize import (
    PDF_LABELS,
    detect_user_language,
    get_pdf_labels,
    localize_research_response,
    render_localized_research_pdf,
    translate_research_answer,
)
from backend.app.services.report_pdf import (
    research_pdf_body_html,
    research_result_to_markdown,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_answer() -> ResearchAnswerPayload:
    return ResearchAnswerPayload(
        executive_summary="Diffusion models give strong perceptual quality at low bitrates [1].",
        key_points=[
            KeyPoint(text="Better perceptual quality than autoencoders [1].", source_ids=["s1"]),
            KeyPoint(text="Sampling cost is the main trade-off [s2].", source_ids=["s2"]),
        ],
        limitations="Sampling is slow; rate-distortion behavior varies.",
        report_markdown="### Background\nDiffusion models compress images well [1].\n",
        citations=[
            CitationEntry(source_id="s1", citation_label=1, url="https://example.com/a", title="Paper A"),
            CitationEntry(source_id="s2", citation_label=2, url="https://example.com/b", title="Paper B"),
        ],
    )


def _make_result(question: str) -> ResearchResponseBody:
    return ResearchResponseBody(
        question=question,
        planning_queries=["q"],
        evidence=[],
        answer=_make_answer(),
    )


class _FakeDeepseek:
    """Stand-in DeepseekClient that returns a scripted JSON payload.

    Use ``.last_messages`` to assert what the production code sent.
    """

    def __init__(self, payload: dict[str, Any] | None = None, *, raise_exc: Exception | None = None) -> None:
        self.payload = payload or {}
        self.raise_exc = raise_exc
        self.last_messages: list[dict[str, Any]] | None = None
        self.calls: int = 0

    async def chat_completion(self, messages, **kwargs):  # noqa: ANN001
        self.calls += 1
        self.last_messages = list(messages)
        if self.raise_exc is not None:
            raise self.raise_exc
        return {
            "choices": [
                {"message": {"content": json.dumps(self.payload, ensure_ascii=False)}}
            ]
        }


# ---------------------------------------------------------------------------
# Language detection + labels
# ---------------------------------------------------------------------------


def test_detect_user_language_english() -> None:
    assert detect_user_language("What are the benefits of diffusion models?") == "en"


def test_detect_user_language_chinese() -> None:
    assert detect_user_language("扩散模型在图像压缩上的优势是什么？") == "zh"


def test_detect_user_language_mixed_picks_chinese() -> None:
    # Mixed CJK + Latin still counts as zh — any CJK ideograph wins.
    assert detect_user_language("Diffusion 模型 advantages?") == "zh"


def test_detect_user_language_empty_or_punctuation() -> None:
    assert detect_user_language("") == "en"
    assert detect_user_language("???") == "en"


def test_get_pdf_labels_known_and_fallback() -> None:
    zh = get_pdf_labels("zh")
    assert zh["report_title"].startswith("DeepScout")
    assert zh["question"] == "研究问题"
    assert zh["citations"] == "参考文献"
    # Unknown lang falls back to English.
    en = get_pdf_labels("ja")
    assert en is PDF_LABELS["en"]


# ---------------------------------------------------------------------------
# research_result_to_markdown / body_html honour labels
# ---------------------------------------------------------------------------


def test_research_result_to_markdown_chinese_labels() -> None:
    md = research_result_to_markdown(_make_result("扩散模型?"), labels=get_pdf_labels("zh"))
    assert "# DeepScout 研究报告" in md
    assert "## 研究问题" in md
    assert "## 摘要" in md
    assert "## 关键要点" in md
    assert "## 参考文献" in md
    assert "DeepScout Research Report" not in md


def test_research_pdf_body_html_chinese_citations_section_split() -> None:
    result = _make_result("扩散模型?")
    html = research_pdf_body_html(result, labels=get_pdf_labels("zh"))
    # Localized "Citations" heading appears once in the references block. Any
    # English "Citations" heading would mean the markdown split missed.
    assert "参考文献" in html
    assert "Citations" not in html
    # Anchor ids for cite-1/cite-2 should still come from the references block.
    assert 'id="cite-1"' in html
    assert 'id="cite-2"' in html


def test_research_result_to_markdown_defaults_to_english() -> None:
    md = research_result_to_markdown(_make_result("English question"))
    assert "## Question" in md
    assert "## Executive Summary" in md
    assert "## Citations" in md


# ---------------------------------------------------------------------------
# translate_research_answer (async helpers run via asyncio.run to avoid
# pytest-asyncio as a hard dependency)
# ---------------------------------------------------------------------------


def test_translate_returns_input_when_target_is_english() -> None:
    answer = _make_answer()
    fake = _FakeDeepseek({"executive_summary": "should-not-be-used"})
    out = asyncio.run(translate_research_answer(answer, target_lang="en", client=fake))
    assert out is answer
    assert fake.calls == 0


def test_translate_unsupported_lang_returns_input() -> None:
    answer = _make_answer()
    fake = _FakeDeepseek({"executive_summary": "should-not-be-used"})
    out = asyncio.run(translate_research_answer(answer, target_lang="ja", client=fake))
    assert out is answer
    assert fake.calls == 0


def test_translate_zh_uses_llm_and_preserves_citations() -> None:
    answer = _make_answer()
    fake = _FakeDeepseek(
        {
            "executive_summary": "扩散模型在低比特率下感知质量优秀 [1]。",
            "key_points": [
                "比自编码器具有更好的感知质量 [1]。",
                "采样开销是主要权衡 [s2]。",
            ],
            "limitations": "采样较慢；率失真表现各异。",
            "report_markdown": "### 背景\n扩散模型在图像压缩上表现良好 [1]。\n",
        }
    )
    out = asyncio.run(translate_research_answer(answer, target_lang="zh", client=fake))

    assert fake.calls == 1
    assert fake.last_messages is not None
    sys_msg = fake.last_messages[0]["content"]
    assert "Simplified Chinese" in sys_msg
    # System prompt must teach the model to keep citation/source-id markers.
    assert "[1]" in sys_msg and "[s0]" in sys_msg

    assert out is not answer
    assert "扩散模型" in out.executive_summary
    assert "[1]" in out.executive_summary
    assert "[s2]" in out.key_points[1].text
    # source_ids on key points are preserved as-is.
    assert out.key_points[0].source_ids == ["s1"]
    assert out.key_points[1].source_ids == ["s2"]
    # Citations themselves (titles, urls) are never translated.
    assert out.citations[0].title == "Paper A"
    assert out.citations[1].url == "https://example.com/b"
    assert len(out.key_points) == len(answer.key_points)


def test_translate_falls_back_on_llm_failure() -> None:
    answer = _make_answer()
    fake = _FakeDeepseek(raise_exc=RuntimeError("boom"))
    out = asyncio.run(translate_research_answer(answer, target_lang="zh", client=fake))
    assert out is answer  # graceful fallback to English original


def test_translate_falls_back_on_bad_json() -> None:
    answer = _make_answer()

    class BrokenJsonFake:
        calls = 0
        last_messages = None

        async def chat_completion(self, messages, **kwargs):  # noqa: ANN001
            BrokenJsonFake.calls += 1
            return {"choices": [{"message": {"content": "not json"}}]}

    out = asyncio.run(
        translate_research_answer(answer, target_lang="zh", client=BrokenJsonFake())
    )
    assert out is answer


def test_translate_keeps_originals_on_length_mismatch() -> None:
    answer = _make_answer()
    # Returns only ONE key point — production code must NOT silently drop the other.
    fake = _FakeDeepseek(
        {
            "executive_summary": "摘要 [1]。",
            "key_points": ["只有一条 [1]。"],
            "limitations": "局限。",
            "report_markdown": "正文。",
        }
    )
    out = asyncio.run(translate_research_answer(answer, target_lang="zh", client=fake))
    assert out is not answer
    assert len(out.key_points) == 2
    # Both points fall back to their English source text.
    assert out.key_points[0].text == answer.key_points[0].text
    assert out.key_points[1].text == answer.key_points[1].text


# ---------------------------------------------------------------------------
# localize_research_response + end-to-end PDF wrapper
# ---------------------------------------------------------------------------


def test_localize_research_response_english_is_passthrough() -> None:
    result = _make_result("What about diffusion?")
    out, lang = asyncio.run(localize_research_response(result))
    assert lang == "en"
    assert out is result  # no copy, no LLM call


def test_localize_research_response_chinese_calls_llm(monkeypatch) -> None:
    fake = _FakeDeepseek(
        {
            "executive_summary": "扩散模型 [1]",
            "key_points": ["要点A [1]", "要点B [s2]"],
            "limitations": "局限",
            "report_markdown": "### 背景\n正文 [1]\n",
        }
    )
    monkeypatch.setattr(report_localize, "DeepseekClient", lambda *a, **kw: fake)

    result = _make_result("扩散模型?")
    out, lang = asyncio.run(localize_research_response(result))
    assert lang == "zh"
    assert out is not result
    assert out.answer.executive_summary.startswith("扩散模型")
    assert fake.calls == 1


def test_render_localized_research_pdf_chinese(monkeypatch) -> None:
    fake = _FakeDeepseek(
        {
            "executive_summary": "扩散模型 [1] 在低比特率有优势。",
            "key_points": ["感知质量更佳 [1]。", "采样开销 [s2]。"],
            "limitations": "局限。",
            "report_markdown": "### 背景\n扩散模型 [1]。\n",
        }
    )
    monkeypatch.setattr(report_localize, "DeepseekClient", lambda *a, **kw: fake)

    pdf = asyncio.run(render_localized_research_pdf(_make_result("扩散模型的优势?")))
    assert pdf.startswith(b"%PDF")
    assert b"%%EOF" in pdf
    assert fake.calls == 1


def test_render_localized_research_pdf_english_skips_llm() -> None:
    # No monkeypatch needed — translate_research_answer short-circuits for "en"
    # and never instantiates a DeepseekClient.
    pdf = asyncio.run(render_localized_research_pdf(_make_result("What about diffusion?")))
    assert pdf.startswith(b"%PDF")
    assert b"%%EOF" in pdf
