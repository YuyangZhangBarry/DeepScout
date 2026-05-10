from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from backend.app.config import Settings
from backend.app.services.embeddings import embed_texts, embeddings_post_url


@pytest.mark.parametrize(
    ("base", "want"),
    [
        ("https://api.openai.com/v1", "https://api.openai.com/v1/embeddings"),
        ("https://api.openai.com", "https://api.openai.com/v1/embeddings"),
        ("https://api.openai.com/", "https://api.openai.com/v1/embeddings"),
        ("https://proxy.example/v1/", "https://proxy.example/v1/embeddings"),
    ],
)
def test_embeddings_post_url(base: str, want: str) -> None:
    assert embeddings_post_url(base) == want


def test_embed_texts_retries_429_then_ok() -> None:
    class MC:
        def __init__(self) -> None:
            self.n = 0

        async def post(
            self,
            url: str,
            *,
            headers: dict[str, str] | None = None,
            json: dict | None = None,
            timeout: float | None = None,
        ) -> httpx.Response:
            self.n += 1
            req = httpx.Request("POST", url)
            if self.n == 1:
                return httpx.Response(429, headers={"Retry-After": "0"}, request=req)
            return httpx.Response(
                200,
                json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]},
                request=req,
            )

        async def aclose(self) -> None:
            return None

    mc = MC()
    settings = Settings(
        embedding_api_key="sk-test",
        embedding_base_url="https://api.openai.com",
        embedding_batch_size=1,
        embedding_max_retries=3,
        embedding_min_seconds_between_batches=0.0,
        embedding_retry_base_seconds=0.01,
    )

    async def run() -> None:
        with patch("backend.app.services.embeddings.asyncio.sleep", new_callable=AsyncMock):
            out = await embed_texts(["a"], settings=settings, client=mc)
        assert out == [[0.1, 0.2]]
        assert mc.n == 2

    asyncio.run(run())


def test_embed_texts_two_batches_sleep_between() -> None:
    sleeps: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    class MC:
        def __init__(self) -> None:
            self.n = 0

        async def post(
            self,
            url: str,
            *,
            headers: dict[str, str] | None = None,
            json: dict | None = None,
            timeout: float | None = None,
        ) -> httpx.Response:
            self.n += 1
            req = httpx.Request("POST", url)
            vec = [[0.1], [0.2]][self.n - 1]
            return httpx.Response(
                200,
                json={"data": [{"index": 0, "embedding": vec}]},
                request=req,
            )

        async def aclose(self) -> None:
            return None

    mc = MC()
    settings = Settings(
        embedding_api_key="sk-test",
        embedding_base_url="https://api.openai.com",
        embedding_batch_size=1,
        embedding_max_retries=2,
        embedding_min_seconds_between_batches=0.5,
        embedding_retry_base_seconds=0.01,
    )

    async def run() -> None:
        with patch("backend.app.services.embeddings.asyncio.sleep", side_effect=record_sleep):
            out = await embed_texts(["a", "b"], settings=settings, client=mc)
        assert out == [[0.1], [0.2]]
        assert mc.n == 2
        assert sleeps == [0.5]

    asyncio.run(run())
