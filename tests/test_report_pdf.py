from backend.app.schemas_research import ResearchAnswerPayload, ResearchResponseBody
from backend.app.services.report_pdf import render_research_pdf, research_result_to_markdown


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
    assert pdf.startswith(b"%PDF-1.4")
    assert b"%%EOF" in pdf
