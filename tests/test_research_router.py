from unittest.mock import AsyncMock, patch

from backend.app.schemas_research import (
    ResearchAnswerPayload,
    ResearchResponseBody,
)


@patch("backend.app.routers.research.run_research_v0", new_callable=AsyncMock)
def test_post_research_returns_payload(mock_run, client) -> None:
    mock_run.return_value = ResearchResponseBody(
        question="What is test?",
        planning_queries=["t1", "t2"],
        evidence=[],
        answer=ResearchAnswerPayload(
            executive_summary="Summary.",
            key_points=[],
            limitations="None.",
            report_markdown="# Report\n\nBody.\n",
            citations=[],
        ),
    )
    r = client.post("/v1/research/", json={"question": "What is test?"})
    assert r.status_code == 200
    data = r.json()
    assert data["question"] == "What is test?"
    assert data["planning_queries"] == ["t1", "t2"]
    assert data["answer"]["executive_summary"] == "Summary."
    mock_run.assert_awaited_once()


@patch("backend.app.routers.research.run_research_v0", new_callable=AsyncMock)
def test_post_research_runtime_error_503(mock_run, client) -> None:
    mock_run.side_effect = RuntimeError("DEEPSEEK_API_KEY is not set")
    r = client.post("/v1/research/", json={"question": "What is test?"})
    assert r.status_code == 503
    assert "DEEPSEEK" in r.json()["error"]["message"]
