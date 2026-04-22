import json
import logging

import httpx
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import ValidationError

from backend.app.schemas_research import ResearchRequestBody, ResearchResponseBody
from backend.app.services.research_v0 import run_research_v0

logger = logging.getLogger(__name__)

router = APIRouter()


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
