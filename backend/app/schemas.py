from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    app: str
    version: str = "0.1.0"


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail = Field(..., description="Machine-readable error envelope")
