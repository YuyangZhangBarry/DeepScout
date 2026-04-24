import asyncio
import json
import logging

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from backend.app.schemas_research import (
    ResearchJobCreatedResponse,
    ResearchJobStatusResponse,
    ResearchRequestBody,
    ResearchResponseBody,
)
from backend.app.services.research_jobs import execute_research_job, get_research_job_store
from backend.app.services.research_v0 import run_research_v0

logger = logging.getLogger(__name__)

router = APIRouter()


def _job_status_response(job_id: str, snap: dict) -> ResearchJobStatusResponse:
    result = None
    if snap.get("result"):
        result = ResearchResponseBody.model_validate(snap["result"])
    events = snap.get("events") or []
    return ResearchJobStatusResponse(
        job_id=job_id,
        status=snap["status"],
        phase=snap.get("phase"),
        error=snap.get("error"),
        events=events[-50:],
        result=result,
    )


@router.post(
    "/",
    response_model=ResearchResponseBody,
    summary="Run literature research v0 (plan → search → optional fetch → synthesis)",
)
async def post_research(request: Request, body: ResearchRequestBody) -> ResearchResponseBody:
    rid = getattr(request.state, "request_id", None)
    logger.info(
        "[research] request_start request_id=%s question_chars=%s",
        rid,
        len(body.question),
    )
    try:
        return await run_research_v0(body)
    except RuntimeError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except httpx.HTTPError as exc:
        logger.warning("research upstream HTTP error: %s", exc)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail="LLM or network request failed",
        ) from exc
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("research parse/validation error: %s", exc)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail="LLM returned invalid JSON or schema",
        ) from exc


@router.post(
    "/jobs",
    response_model=ResearchJobCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enqueue async research (poll GET /jobs/{job_id} or SSE GET /jobs/{job_id}/events)",
)
async def post_research_job(
    request: Request,
    body: ResearchRequestBody,
    background_tasks: BackgroundTasks,
) -> ResearchJobCreatedResponse:
    store = get_research_job_store()
    job_id = await store.create(body)
    background_tasks.add_task(execute_research_job, job_id, body)
    base = str(request.base_url).rstrip("/")
    return ResearchJobCreatedResponse(
        job_id=job_id,
        status="pending",
        poll_url=f"{base}/v1/research/jobs/{job_id}",
        events_url=f"{base}/v1/research/jobs/{job_id}/events",
    )


@router.get(
    "/jobs/{job_id}",
    response_model=ResearchJobStatusResponse,
    summary="Poll research job: status, recent phase events, and result when completed",
)
async def get_research_job(job_id: str) -> ResearchJobStatusResponse:
    store = get_research_job_store()
    snap = await store.snapshot(job_id)
    if not snap:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Unknown job_id")
    return _job_status_response(job_id, snap)


@router.get(
    "/jobs/{job_id}/events",
    summary="SSE stream of phase events until the job completes or fails",
)
async def stream_research_job_events(job_id: str) -> StreamingResponse:
    store = get_research_job_store()
    if await store.snapshot(job_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Unknown job_id")

    async def event_generator():
        last_idx = 0
        while True:
            snap = await store.snapshot(job_id)
            if snap is None:
                yield f"data: {json.dumps({'error': 'job_missing'})}\n\n"
                break
            events = snap.get("events") or []
            while last_idx < len(events):
                yield f"data: {json.dumps(events[last_idx], ensure_ascii=False)}\n\n"
                last_idx += 1
            st = snap.get("status")
            if st in ("completed", "failed"):
                done_payload: dict = {"status": st, "done": True}
                if st == "failed" and snap.get("error"):
                    done_payload["error"] = snap["error"]
                yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"
                break
            await asyncio.sleep(0.35)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
