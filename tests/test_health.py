from backend.app.middleware.request_id import REQUEST_ID_HEADER


def test_health_ok(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert isinstance(data["app"], str) and data["app"]
    assert data["version"] == "0.1.0"
    rid = r.headers.get(REQUEST_ID_HEADER.lower())
    assert rid
    assert len(rid) == 36
