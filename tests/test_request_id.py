import re

from backend.app.middleware.request_id import REQUEST_ID_HEADER

_UUID36 = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)


def test_request_id_generated_when_absent(client) -> None:
    r = client.get("/health")
    h = r.headers.get(REQUEST_ID_HEADER.lower())
    assert h and _UUID36.match(h)


def test_request_id_echoed_when_valid(client) -> None:
    rid = "aaaaaaaa-bbbb-4ccc-dddd-eeeeeeeeeeee"
    r = client.get("/health", headers={REQUEST_ID_HEADER: rid})
    assert r.headers.get(REQUEST_ID_HEADER.lower()) == rid


def test_request_id_whitespace_only_falls_back_to_uuid(client) -> None:
    r = client.get("/health", headers={REQUEST_ID_HEADER: "   "})
    h = r.headers.get(REQUEST_ID_HEADER.lower())
    assert h and _UUID36.match(h)
