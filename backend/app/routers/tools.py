import logging

import httpx
from fastapi import APIRouter, HTTPException, status

from backend.app.schemas_tools import (
    FetchRequestBody,
    FetchResponseBody,
    SearchRequestBody,
    SearchResponseBody,
)
from backend.app.services.fetch import fetch_urls
from backend.app.services.search import search_web

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/search", response_model=SearchResponseBody)
async def tools_search(body: SearchRequestBody) -> SearchResponseBody:
    try:
        items = await search_web(
            body.queries,
            max_results_per_query=body.max_results_per_query,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except httpx.HTTPError as exc:
        logger.warning("search provider HTTP error: %s", exc)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail="Search provider request failed",
        ) from exc
    return SearchResponseBody(items=items)


@router.post("/fetch", response_model=FetchResponseBody)
async def tools_fetch(body: FetchRequestBody) -> FetchResponseBody:
    documents = await fetch_urls(body.urls)
    return FetchResponseBody(documents=documents)
