import asyncio
from unittest.mock import patch

from backend.app.schemas_tools import FetchedDocument
from backend.app.services.fetch import fetch_urls_with_retry


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
