from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app.config import get_settings
from backend.app.main import create_app
from backend.app.schemas_tools import FetchedDocument, SearchHit


def test_search_tavily_without_key_returns_503(monkeypatch, clear_settings_cache) -> None:
    monkeypatch.setenv("SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("TAVILY_API_KEY", "")
    get_settings.cache_clear()
    client = TestClient(create_app())
    r = client.post("/v1/tools/search", json={"queries": ["hello"]})
    assert r.status_code == 503
    body = r.json()
    assert body["error"]["code"] == "http_503"
    assert "TAVILY_API_KEY" in body["error"]["message"]


def test_search_provider_rejects_tavily_combined_with_academic(
    monkeypatch, clear_settings_cache
) -> None:
    monkeypatch.setenv("SEARCH_PROVIDER", "semantic_scholar,tavily")
    get_settings.cache_clear()
    with pytest.raises(ValidationError, match="tavily cannot be combined"):
        create_app()


@patch("backend.app.routers.tools.search_literature", new_callable=AsyncMock)
def test_search_success_uses_mocked_search(mock_search, client) -> None:
    mock_search.return_value = [
        SearchHit(
            url="https://docs.example/page",
            title="T",
            snippet="S",
            source_query="hello",
        )
    ]
    r = client.post("/v1/tools/search", json={"queries": ["hello"]})
    assert r.status_code == 200
    data = r.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["url"] == "https://docs.example/page"
    mock_search.assert_awaited_once()


@patch("backend.app.routers.tools.fetch_urls", new_callable=AsyncMock)
def test_fetch_success_uses_mocked_fetch(mock_fetch, client) -> None:
    mock_fetch.return_value = [
        FetchedDocument(
            url="https://example.com/",
            final_url="https://example.com/",
            title="Ex",
            text="body",
            content_hash="abc",
            status_code=200,
        )
    ]
    r = client.post("/v1/tools/fetch", json={"urls": ["https://example.com/"]})
    assert r.status_code == 200
    docs = r.json()["documents"]
    assert len(docs) == 1
    assert docs[0]["text"] == "body"
    mock_fetch.assert_awaited_once()
