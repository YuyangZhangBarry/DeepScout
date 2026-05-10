"""Log each HTTP request as explicit steps (start + end with latency)."""

from __future__ import annotations

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("backend.http")


class HttpStepLoggingMiddleware(BaseHTTPMiddleware):
    """Emit [step=http] lines for request start and completion (status + duration)."""

    async def dispatch(self, request: Request, call_next) -> Response:
        rid = getattr(request.state, "request_id", None)
        method = request.method
        path = request.url.path
        client = request.client.host if request.client else "-"
        logger.info(
            "[step=http] action=request_start method=%s path=%s client=%s request_id=%s",
            method,
            path,
            client,
            rid or "-",
        )
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.exception(
                "[step=http] action=request_error method=%s path=%s elapsed_ms=%.1f request_id=%s",
                method,
                path,
                elapsed_ms,
                rid or "-",
            )
            raise
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "[step=http] action=request_end method=%s path=%s status=%s elapsed_ms=%.1f request_id=%s",
            method,
            path,
            response.status_code,
            elapsed_ms,
            rid or "-",
        )
        return response
