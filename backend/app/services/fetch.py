import asyncio
import hashlib
import logging
import time
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Iterable
from urllib.parse import urlparse

import httpx
import trafilatura

from backend.app.config import Settings, get_settings
from backend.app.schemas_tools import FetchedDocument
from backend.app.services.urlnorm import normalize_http_url

logger = logging.getLogger(__name__)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _content_hash_from_text(text: str) -> str:
    return _sha256_hex(text.encode("utf-8"))


def _host_key(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc.lower() or "__invalid__"


class _HostRateLimiter:
    """Bound concurrent fetches per host and keep a small delay between host hits."""

    def __init__(self, *, per_host_concurrent: int, per_host_delay_seconds: float) -> None:
        self._per_host_concurrent = max(1, per_host_concurrent)
        self._delay = max(0.0, per_host_delay_seconds)
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._last_started: dict[str, float] = {}
        self._state_lock = asyncio.Lock()

    async def _state_for_host(self, host: str) -> tuple[asyncio.Semaphore, asyncio.Lock]:
        async with self._state_lock:
            sem = self._semaphores.get(host)
            if sem is None:
                sem = asyncio.Semaphore(self._per_host_concurrent)
                self._semaphores[host] = sem
            lock = self._locks.get(host)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[host] = lock
            return sem, lock

    @asynccontextmanager
    async def slot(self, url: str) -> AsyncIterator[None]:
        host = _host_key(url)
        sem, lock = await self._state_for_host(host)
        async with sem:
            async with lock:
                if self._delay > 0:
                    now = time.monotonic()
                    wait_for = self._last_started.get(host, 0.0) + self._delay - now
                    if wait_for > 0:
                        await asyncio.sleep(wait_for)
                self._last_started[host] = time.monotonic()
            yield


async def fetch_url(
    url: str,
    *,
    settings: Settings | None = None,
    client: httpx.AsyncClient | None = None,
) -> FetchedDocument:
    """
    Fetch a single URL, extract main text with trafilatura, compute content_hash of extracted text.
    On failure returns FetchedDocument with error set (does not raise).
    """
    cfg = settings or get_settings()
    normalized = normalize_http_url(url)
    if not normalized:
        return FetchedDocument(
            url=url,
            error="invalid or unsupported URL",
        )

    headers = {"User-Agent": cfg.http_user_agent}
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(
            follow_redirects=True,
            headers=headers,
        )

    try:
        response = await client.get(normalized, timeout=cfg.fetch_timeout_seconds)
        final_url = str(response.url)
        status = response.status_code
        body = response.content

        if status >= 400:
            return FetchedDocument(
                url=normalized,
                final_url=final_url,
                status_code=status,
                content_hash=_sha256_hex(body),
                error=f"HTTP {status}",
            )

        extracted = trafilatura.extract(body, url=final_url, include_comments=False, include_tables=False)
        text = (extracted or "").strip()

        meta = trafilatura.extract_metadata(body, default_url=final_url)
        title = (meta.title or "").strip() if meta is not None else ""

        if text:
            digest = _content_hash_from_text(text)
        else:
            digest = _sha256_hex(body)

        return FetchedDocument(
            url=normalized,
            final_url=final_url,
            title=title,
            text=text,
            content_hash=digest,
            status_code=status,
        )
    except httpx.TimeoutException as exc:
        logger.info("fetch timeout url=%s err=%s", normalized, exc)
        return FetchedDocument(url=normalized, error="timeout")
    except httpx.HTTPError as exc:
        logger.info("fetch http error url=%s err=%s", normalized, exc)
        return FetchedDocument(url=normalized, error=str(exc) or "http error")
    except Exception as exc:  # noqa: BLE001 — return envelope for batch safety
        logger.exception("fetch unexpected error url=%s", normalized)
        return FetchedDocument(url=normalized, error=str(exc))
    finally:
        if own_client and client is not None:
            await client.aclose()


async def fetch_urls(
    urls: Iterable[str],
    *,
    settings: Settings | None = None,
) -> list[FetchedDocument]:
    """
    Fetch many URLs with bounded concurrency (polite parallel fetch).
    """
    cfg = settings or get_settings()
    url_list = [u for u in urls if (u or "").strip()]
    if not url_list:
        return []

    sem = asyncio.Semaphore(cfg.fetch_max_concurrent)
    host_limiter = _HostRateLimiter(
        per_host_concurrent=cfg.fetch_per_host_max_concurrent,
        per_host_delay_seconds=cfg.fetch_per_host_delay_seconds,
    )
    shared_client: httpx.AsyncClient | None = httpx.AsyncClient(
        follow_redirects=True,
        headers={"User-Agent": cfg.http_user_agent},
    )

    async def _one(u: str) -> FetchedDocument:
        async with sem:
            async with host_limiter.slot(u):
                return await fetch_url(u, settings=cfg, client=shared_client)

    try:
        return list(await asyncio.gather(*[_one(u) for u in url_list]))
    finally:
        if shared_client is not None:
            await shared_client.aclose()


async def fetch_urls_with_retry(
    urls: Iterable[str],
    *,
    settings: Settings | None = None,
    max_rounds: int = 2,
) -> list[FetchedDocument]:
    """
    Like :func:`fetch_urls`, but optionally runs a second round only for URLs that
    failed or returned empty extracted text (Day 11–12 fetch resilience).
    """
    cfg = settings or get_settings()
    url_list = [u for u in urls if (u or "").strip()]
    if not url_list:
        return []
    rounds = max(1, min(int(max_rounds), 3))
    first = await fetch_urls(url_list, settings=cfg)
    if rounds < 2:
        return first

    merged = list(first)
    retry_idx = [
        i
        for i, d in enumerate(first)
        if d.error or not (d.text or "").strip()
    ]
    if not retry_idx:
        return merged

    to_retry = [url_list[i] for i in retry_idx]
    second = await fetch_urls(to_retry, settings=cfg)
    for j, i in enumerate(retry_idx):
        if j < len(second):
            nd = second[j]
            if not nd.error and (nd.text or "").strip():
                merged[i] = nd
    return merged
