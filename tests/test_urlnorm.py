import pytest

from backend.app.services.urlnorm import normalize_http_url, url_dedup_key


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", None),
        ("   ", None),
        ("example.com", "https://example.com/"),
        ("HTTPS://ExAmPlE.com/foo", "https://example.com/foo"),
        ("http://127.0.0.1:8000", "http://127.0.0.1:8000/"),
        ("https://a.example/path?q=1#frag", "https://a.example/path?q=1#frag"),
        ("ftp://files.example/file", None),
        ("https://", None),
    ],
)
def test_normalize_http_url(raw: str, expected: str | None) -> None:
    assert normalize_http_url(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected_key"),
    [
        ("", ""),
        ("  ", ""),
        ("example.com", "https://example.com/"),
        ("HTTPS://ExAmPlE.com/foo/", "https://example.com/foo"),
        ("https://a.example/x?y=1#z", "https://a.example/x"),
    ],
)
def test_url_dedup_key(raw: str, expected_key: str) -> None:
    assert url_dedup_key(raw) == expected_key
