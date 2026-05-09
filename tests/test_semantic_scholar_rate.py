import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import httpx

from backend.app.config import Settings
from backend.app.services import search as search_mod
from backend.app.services.search import _semantic_scholar_search_one


def _reset_s2_rate_state() -> None:
    search_mod._s2_next_available_monotonic = 0.0


async def _two_searches() -> tuple[float, float]:
    _reset_s2_rate_state()
    times: list[float] = []

    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"data": []}

    async def get_side_effect(*args, **kwargs):
        times.append(time.monotonic())
        return response

    client = MagicMock(spec=httpx.AsyncClient)
    client.get = AsyncMock(side_effect=get_side_effect)
    settings = Settings(
        semantic_scholar_min_seconds_between_requests=0.12,
        http_user_agent="test-agent",
    )
    await _semantic_scholar_search_one(
        client, query="q1", max_results=5, settings=settings
    )
    await _semantic_scholar_search_one(
        client, query="q2", max_results=5, settings=settings
    )
    assert len(times) == 2
    return times[0], times[1]


def test_semantic_scholar_search_respects_min_interval_between_http_gets() -> None:
    t0, t1 = asyncio.run(_two_searches())
    assert t1 - t0 >= 0.11
