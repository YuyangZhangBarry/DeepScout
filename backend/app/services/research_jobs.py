from __future__ import annotations

import asyncio
import copy
import logging
import time
import uuid
from typing import Any

from backend.app.logging_context import trace_scope
from backend.app.schemas_research import ResearchRequestBody, ResearchResponseBody
from backend.app.services.research_phases import ResearchPhase

logger = logging.getLogger(__name__)


class ResearchJobStore:
    """In-memory async research jobs (Day 13–14: poll + SSE). Not durable across restarts."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    async def create(self, body: ResearchRequestBody) -> str:
        job_id = str(uuid.uuid4())
        async with self._lock:
            self._jobs[job_id] = {
                "status": "pending",
                "created_at": time.time(),
                "updated_at": time.time(),
                "body": body.model_dump(),
                "result": None,
                "error": None,
                "events": [],
                "phase": None,
            }
        return job_id

    async def push_event(self, job_id: str, phase: str, detail: str = "") -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job["events"].append(
                {"phase": phase, "detail": detail, "ts": time.time()},
            )
            job["phase"] = phase
            job["status"] = "running"
            job["updated_at"] = time.time()

    async def set_result(self, job_id: str, result: ResearchResponseBody) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job["status"] = "completed"
            job["result"] = result.model_dump()
            job["error"] = None
            job["updated_at"] = time.time()
            # Phase DONE is already recorded via on_progress from run_research_v0.

    async def set_failed(self, job_id: str, message: str) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job["status"] = "failed"
            job["phase"] = ResearchPhase.FAILED.value
            job["error"] = message[:4000]
            job["updated_at"] = time.time()
            job["events"].append(
                {"phase": ResearchPhase.FAILED.value, "detail": message[:500], "ts": time.time()},
            )

    async def snapshot(self, job_id: str) -> dict[str, Any] | None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            return copy.deepcopy(job)


_store: ResearchJobStore | None = None


def get_research_job_store() -> ResearchJobStore:
    global _store
    if _store is None:
        _store = ResearchJobStore()
    return _store


def reset_research_job_store() -> None:
    """Clear in-memory jobs (tests only; not safe under concurrent requests)."""
    global _store
    _store = None


async def execute_research_job(job_id: str, body: ResearchRequestBody) -> None:
    from backend.app.services.research_v0 import run_research_v0

    store = get_research_job_store()

    async def on_progress(phase: ResearchPhase, detail: str = "") -> None:
        await store.push_event(job_id, phase.value, detail)

    with trace_scope(f"job:{job_id}"):
        try:
            result = await run_research_v0(body, on_progress=on_progress)
            await store.set_result(job_id, result)
        except Exception as exc:  # noqa: BLE001
            logger.exception("research job %s failed", job_id)
            await store.set_failed(job_id, str(exc))
