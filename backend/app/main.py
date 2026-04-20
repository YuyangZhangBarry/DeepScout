import logging
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.app.config import get_settings
from backend.app.middleware.request_id import REQUEST_ID_HEADER, RequestIDMiddleware
from backend.app.routers import tools as tools_router
from backend.app.schemas import ErrorDetail, ErrorResponse, HealthResponse

logger = logging.getLogger(__name__)


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _error_payload(
    *,
    code: str,
    message: str,
    request: Request | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rid = _request_id(request) if request else None
    body = ErrorResponse(
        error=ErrorDetail(code=code, message=message, request_id=rid, details=details)
    )
    return body.model_dump()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        debug=settings.debug,
    )

    app.add_middleware(RequestIDMiddleware)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        code = f"http_{exc.status_code}"
        message = str(exc.detail) if exc.detail else "HTTP error"
        payload = _error_payload(code=code, message=message, request=request)
        headers = {}
        rid = _request_id(request)
        if rid:
            headers[REQUEST_ID_HEADER] = rid
        return JSONResponse(status_code=exc.status_code, content=payload, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        payload = _error_payload(
            code="validation_error",
            message="Request validation failed",
            request=request,
            details={"errors": exc.errors()},
        )
        headers = {}
        rid = _request_id(request)
        if rid:
            headers[REQUEST_ID_HEADER] = rid
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=payload,
            headers=headers,
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = _request_id(request) or str(uuid.uuid4())
        logger.exception("unhandled error request_id=%s", request_id)
        payload = _error_payload(
            code="internal_error",
            message="An unexpected error occurred",
            request=request,
            details=None if not settings.debug else {"type": type(exc).__name__},
        )
        headers = {REQUEST_ID_HEADER: request_id}
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=payload,
            headers=headers,
        )

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse(app=settings.app_name)

    app.include_router(tools_router.router, prefix="/v1/tools", tags=["tools"])

    return app


app = create_app()
