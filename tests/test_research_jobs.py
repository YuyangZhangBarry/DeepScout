import uuid
from unittest.mock import patch

import pytest

from backend.app.schemas_research import (
    CitationEntry,
    EvidenceSourceOut,
    ResearchAnswerPayload,
    ResearchRequestBody,
    ResearchResponseBody,
)
from backend.app.services.research_jobs import reset_research_job_store


@pytest.fixture(autouse=True)
def _clear_job_store() -> None:
    reset_research_job_store()
    yield
    reset_research_job_store()


async def _complete_job_with_stub_result(job_id: str, body: ResearchRequestBody) -> None:
    from backend.app.services.research_jobs import get_research_job_store

    store = get_research_job_store()
    result = ResearchResponseBody(
        question=body.question,
        planning_queries=["q1"],
        evidence=[],
        answer=ResearchAnswerPayload(
            executive_summary="Done.",
            key_points=[],
            limitations="",
            report_markdown="# R\n",
            citations=[],
        ),
    )
    await store.set_result(job_id, result)


@patch(
    "backend.app.routers.research.execute_research_job",
    side_effect=_complete_job_with_stub_result,
)
def test_post_research_job_then_poll_completed(_mock_exec, client) -> None:
    r = client.post("/v1/research/jobs", json={"question": "What is test?"})
    assert r.status_code == 202
    data = r.json()
    assert data["status"] == "pending"
    assert "job_id" in data
    assert data["poll_url"].endswith(f"/v1/research/jobs/{data['job_id']}")
    assert "/events" in data["events_url"]

    pr = client.get(f"/v1/research/jobs/{data['job_id']}")
    assert pr.status_code == 200
    body = pr.json()
    assert body["status"] == "completed"
    assert body["result"] is not None
    assert body["result"]["question"] == "What is test?"


def test_get_unknown_job_404(client) -> None:
    rid = str(uuid.uuid4())
    r = client.get(f"/v1/research/jobs/{rid}")
    assert r.status_code == 404


@patch(
    "backend.app.routers.research.execute_research_job",
    side_effect=_complete_job_with_stub_result,
)
def test_job_events_stream_ends_with_done(_mock_exec, client) -> None:
    r = client.post("/v1/research/jobs", json={"question": "Stream test question here"})
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    with client.stream("GET", f"/v1/research/jobs/{job_id}/events") as stream:
        assert stream.status_code == 200
        buf = ""
        for chunk in stream.iter_bytes():
            buf += chunk.decode()
            if '"done": true' in buf:
                break
        assert '"done": true' in buf
        assert "completed" in buf


@patch(
    "backend.app.routers.research.execute_research_job",
    side_effect=_complete_job_with_stub_result,
)
def test_completed_job_pdf_download(_mock_exec, client) -> None:
    r = client.post("/v1/research/jobs", json={"question": "PDF test question here"})
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    pdf = client.get(f"/v1/research/jobs/{job_id}/pdf")
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.content.startswith(b"%PDF-1.4")
