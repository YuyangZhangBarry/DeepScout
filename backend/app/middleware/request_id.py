import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


REQUEST_ID_HEADER = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a request id to each request (header or generated) for tracing."""

    async def dispatch(self, request: Request, call_next) -> Response:
        raw = request.headers.get(REQUEST_ID_HEADER.lower())
        candidate = (raw or "").strip()
        request_id = candidate or str(uuid.uuid4())
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
