"""Reject cross-origin browser writes before a route can mutate local state."""

from urllib.parse import urlsplit

from fastapi import HTTPException, Request

from core import settings

UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _origin(value: str, *, allow_path: bool) -> str | None:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or (not allow_path and parsed.query)
        or parsed.fragment
        or (not allow_path and parsed.path not in {"", "/"})
    ):
        return None
    try:
        parsed.port
    except ValueError:
        return None
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"


def verify_csrf(request: Request) -> None:
    """Require a configured same-origin source for every unsafe browser request."""
    if request.method not in UNSAFE_METHODS:
        return
    trusted = {
        origin
        for value in settings.EVIDENCEGRAPH_TRUSTED_ORIGINS.split(",")
        if (origin := _origin(value.strip(), allow_path=False)) is not None
    }
    source = request.headers.get("origin")
    if source is not None:
        source_origin = _origin(source, allow_path=False)
    else:
        referer = request.headers.get("referer")
        source_origin = _origin(referer, allow_path=True) if referer is not None else None
    if source_origin is None or source_origin not in trusted:
        raise HTTPException(status_code=403, detail="Untrusted request origin")
