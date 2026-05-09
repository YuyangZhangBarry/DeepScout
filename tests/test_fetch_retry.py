import asyncio
from unittest.mock import patch

from backend.app.config import Settings
from backend.app.schemas_tools import FetchedDocument
from backend.app.services.fetch import fetch_urls, fetch_urls_with_retry


def test_fetch_urls_with_retry_replaces_failed_slot() -> None:
    bad = FetchedDocument(url="https://a.example/", error="timeout")
    good = FetchedDocument(
        url="https://a.example/",
        final_url="https://a.example/",
        title="T",
        text="recovered body",
        content_hash="h",
        status_code=200,
    )

    async def fake_fetch(urls, settings=None):
        u = list(urls)
        if len(u) == 1 and "a.example" in u[0]:
            return [good]
        return [
            bad,
            FetchedDocument(
                url="https://b.example/",
                title="",
                text="ok",
                content_hash="x",
                status_code=200,
            ),
        ]

    async def main():
        with patch("backend.app.services.fetch.fetch_urls", side_effect=fake_fetch):
            return await fetch_urls_with_retry(
                ["https://a.example/", "https://b.example/"],
                max_rounds=2,
            )

    out = asyncio.run(main())
    assert out[0].text == "recovered body"
    assert out[1].text == "ok"


def test_fetch_urls_limits_concurrency_per_host() -> None:
    active_by_host: dict[str, int] = {}
    max_active_by_host: dict[str, int] = {}

    async def fake_fetch(url, settings=None, client=None):
        host = url.split("/")[2]
        active_by_host[host] = active_by_host.get(host, 0) + 1
        max_active_by_host[host] = max(max_active_by_host.get(host, 0), active_by_host[host])
        await asyncio.sleep(0.01)
        active_by_host[host] -= 1
        return FetchedDocument(
            url=url,
            final_url=url,
            title="",
            text="ok",
            content_hash="x",
            status_code=200,
        )

    async def main():
        settings = Settings(
            fetch_max_concurrent=4,
            fetch_per_host_max_concurrent=1,
            fetch_per_host_delay_seconds=0,
        )
        with patch("backend.app.services.fetch.fetch_url", side_effect=fake_fetch):
            return await fetch_urls(
                [
                    "https://a.example/1",
                    "https://a.example/2",
                    "https://b.example/1",
                    "https://b.example/2",
                ],
                settings=settings,
            )

    out = asyncio.run(main())
    assert len(out) == 4
    assert max_active_by_host == {"a.example": 1, "b.example": 1}
