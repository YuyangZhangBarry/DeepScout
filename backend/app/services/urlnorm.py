from urllib.parse import urlparse, urlunparse


def url_dedup_key(url: str) -> str:
    """Stable key for de-duplicating URLs (scheme + host + path, no query/fragment)."""
    raw = (url or "").strip()
    if not raw:
        return ""
    try:
        p = urlparse(raw if "://" in raw else f"https://{raw}")
        scheme = (p.scheme or "https").lower()
        netloc = p.netloc.lower()
        path = p.path or "/"
        if len(path) > 1 and path.endswith("/"):
            path = path[:-1]
        return urlunparse((scheme, netloc, path, "", "", ""))
    except Exception:
        return raw.lower()


def normalize_http_url(url: str) -> str | None:
    """Return a usable http(s) URL or None if invalid."""
    raw = (url or "").strip()
    if not raw:
        return None
    p = urlparse(raw if "://" in raw else f"https://{raw}")
    scheme = (p.scheme or "https").lower()
    if scheme not in ("http", "https"):
        return None
    netloc = (p.netloc or "").lower()
    if not netloc:
        return None
    path = p.path if p.path else "/"
    return urlunparse((scheme, netloc, path, p.params, p.query, p.fragment))
