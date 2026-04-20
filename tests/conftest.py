import pytest
from fastapi.testclient import TestClient

from backend.app.config import get_settings
from backend.app.main import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture
def clear_settings_cache():
    """Reset settings singleton between tests that change env."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
