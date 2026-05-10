import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx

from backend.app.config import Settings
from backend.app.services.search import (
    _arxiv_entry_to_hit,
    _dedupe_hits_preserving_order,
    _search_arxiv,
    search_literature,
)


ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.01234v2</id>
    <updated>2024-01-03T00:00:00Z</updated>
    <published>2024-01-01T00:00:00Z</published>
    <title> Diffusion Models for Compression </title>
    <summary> A survey of compression with diffusion models. </summary>
    <author><name>Alice Example</name></author>
    <author><name>Bob Example</name></author>
    <link href="http://arxiv.org/abs/2401.01234v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.01234v2" rel="related" type="application/pdf"/>
  </entry>
</feed>
"""


def test_arxiv_entry_maps_to_search_hit() -> None:
    from xml.etree import ElementTree as ET

    root = ET.fromstring(ARXIV_FEED)
    entry = next(iter(root))
    hit = _arxiv_entry_to_hit(entry, "diffusion compression")
    assert hit is not None
    assert hit.paper_id == "arxiv:2401.01234v2"
    assert hit.url == "http://arxiv.org/pdf/2401.01234v2"
    assert hit.title == "Diffusion Models for Compression"
    assert "Alice Example" in hit.snippet
    assert hit.source_query == "diffusion compression"


def test_search_arxiv_uses_official_api_and_parses_feed() -> None:
    response = MagicMock()
    response.status_code = 200
    response.text = ARXIV_FEED
    response.raise_for_status.return_value = None
    client = MagicMock(spec=httpx.AsyncClient)
    client.get = AsyncMock(return_value=response)
    settings = Settings(
        search_provider="arxiv",
        arxiv_min_seconds_between_requests=0,
    )

    async def main():
        return await _search_arxiv(
            ["diffusion compression"],
            max_results_per_query=3,
            settings=settings,
            client=client,
        )

    out = asyncio.run(main())
    assert len(out) == 1
    assert out[0].paper_id == "arxiv:2401.01234v2"
    _, kwargs = client.get.await_args
    assert kwargs["params"]["search_query"] == "all:diffusion compression"
    assert kwargs["params"]["max_results"] == 3


def test_search_literature_merges_semantic_scholar_and_arxiv(monkeypatch) -> None:
    from backend.app.schemas_tools import SearchHit
    from backend.app.services import search as search_mod

    async def fake_provider(provider, queries, *, max_results_per_query, settings, client):
        if provider == "semantic_scholar":
            return [
                SearchHit(
                    url="https://s2.example/paper",
                    title="S2",
                    snippet="",
                    paper_id="paper-1",
                )
            ]
        if provider == "arxiv":
            return [
                SearchHit(
                    url="https://arxiv.org/pdf/2401.01234",
                    title="arXiv",
                    snippet="",
                    paper_id="arxiv:2401.01234",
                )
            ]
        raise AssertionError(provider)

    monkeypatch.setattr(search_mod, "_search_provider", fake_provider)

    async def main():
        return await search_literature(
            ["query"],
            settings=Settings(search_provider="semantic_scholar,arxiv"),
        )

    out = asyncio.run(main())
    assert [h.paper_id for h in out] == ["paper-1", "arxiv:2401.01234"]


def test_dedupe_treats_arxiv_abs_and_pdf_urls_as_same_paper() -> None:
    from backend.app.schemas_tools import SearchHit

    out = _dedupe_hits_preserving_order(
        [
            SearchHit(
                url="https://arxiv.org/abs/2401.01234v2",
                title="abs",
                paper_id="s2-paper",
            ),
            SearchHit(
                url="https://arxiv.org/pdf/2401.01234v2",
                title="pdf",
                paper_id="arxiv:2401.01234v2",
            ),
        ]
    )
    assert len(out) == 1
    assert out[0].title == "abs"
