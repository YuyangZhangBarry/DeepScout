"""Per-request / per-job trace id for structured stderr logs (see main._configure_app_logging)."""

from __future__ import annotations

import contextvars
import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)


class TraceIdFilter(logging.Filter):
    """Inject ``record.trace_id`` for formatters (HTTP request_id or ``job:…``)."""

    def filter(self, record: logging.LogRecord) -> bool:
        tid = _trace_id.get()
        record.trace_id = tid if tid else "-"
        return True


def get_trace_id() -> str | None:
    return _trace_id.get()


def bind_trace_id(value: str | None) -> contextvars.Token[Any]:
    return _trace_id.set(value)


def reset_trace_id(token: contextvars.Token[Any]) -> None:
    _trace_id.reset(token)


@contextmanager
def trace_scope(trace_id: str | None) -> Generator[None, None, None]:
    token = bind_trace_id(trace_id)
    try:
        yield
    finally:
        reset_trace_id(token)
