"""Browser writes require an exact trusted source; public reads still work."""

import pytest
from fastapi.testclient import TestClient

from service import app


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/auth/login"),
        ("POST", "/papers"),
        ("POST", "/history"),
        ("PUT", "/conversations/thread"),
        ("DELETE", "/conversations/thread"),
    ],
)
def test_cross_site_browser_write_is_rejected(method: str, path: str) -> None:
    client = TestClient(app)
    response = client.request(method, path, headers={"Origin": "https://evil.example"})
    assert response.status_code == 403
    assert response.json()["detail"] == "Untrusted request origin"


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Origin": "null"},
        {"Origin": "http://127.0.0.1:3000.evil.example"},
        {"Origin": "http://127.0.0.1:3000/path"},
        {"Origin": "http://[broken"},
        {"Referer": "http://127.0.0.1:3000.evil.example/path"},
        {"Origin": "null", "Referer": "http://127.0.0.1:3000/"},
    ],
)
def test_missing_or_malformed_source_is_rejected(headers: dict[str, str]) -> None:
    client = TestClient(app)
    response = client.post("/auth/logout", headers=headers)
    assert response.status_code == 403


def test_same_origin_and_referer_fallback_are_accepted(test_client) -> None:
    assert test_client.post("/auth/logout").status_code == 204
    client = TestClient(app)
    response = client.post(
        "/auth/logout",
        headers={"Referer": "http://127.0.0.1:3000/?view=tutorial"},
    )
    assert response.status_code == 204


def test_public_tutorial_reads_do_not_require_source() -> None:
    client = TestClient(app)
    assert client.get("/showcase").status_code == 200
