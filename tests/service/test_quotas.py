"""Expensive routes return a usable 429 without trusting spoofed client IPs."""

import pytest
from fastapi import HTTPException, Request

from core import settings
from evidencegraph.accounts import AccountStore
from evidencegraph.quotas import QuotaStore
from service import app
from service.auth import get_account_store
from service.papers import graph_extraction_is_configured
from service.quotas import POLICIES, client_ip, enforce_quota, get_quota_store


def test_registration_is_limited_before_account_creation(
    test_client, tmp_path, monkeypatch
) -> None:
    accounts = AccountStore(tmp_path / "accounts")
    quotas = QuotaStore(tmp_path / "quotas")
    app.dependency_overrides[get_account_store] = lambda: accounts
    app.dependency_overrides[get_quota_store] = lambda: quotas
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_QUOTAS_ENABLED", True)
    monkeypatch.setitem(POLICIES, "register", (3600, 1, None))
    try:
        first = test_client.post(
            "/auth/register",
            json={"username": "first-user", "password": "long-enough-password"},
        )
        assert first.status_code == 201
        second = test_client.post(
            "/auth/register",
            json={"username": "second-user", "password": "long-enough-password"},
        )
        assert second.status_code == 429
        assert int(second.headers["Retry-After"]) > 0
        assert "try again" in second.json()["detail"]
        assert accounts.authenticate("second-user", "long-enough-password") is None
    finally:
        app.dependency_overrides.pop(get_account_store, None)
        app.dependency_overrides.pop(get_quota_store, None)


def test_upload_limit_applies_before_reading_pdf(test_client, tmp_path, monkeypatch) -> None:
    app.dependency_overrides[get_quota_store] = lambda: QuotaStore(tmp_path)
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_QUOTAS_ENABLED", True)
    monkeypatch.setitem(POLICIES, "upload", (3600, 1, None))
    try:
        first = test_client.post("/papers", content=b"bad", headers={"Content-Type": "text/plain"})
        assert first.status_code == 415
        second = test_client.post("/papers", content=b"bad", headers={"Content-Type": "text/plain"})
        assert second.status_code == 429
    finally:
        app.dependency_overrides.pop(get_quota_store, None)


def test_graph_and_chat_routes_share_their_action_budget(
    test_client, mock_agent, tmp_path, monkeypatch
) -> None:
    app.dependency_overrides[get_quota_store] = lambda: QuotaStore(tmp_path)
    app.dependency_overrides[graph_extraction_is_configured] = lambda: False
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_QUOTAS_ENABLED", True)
    monkeypatch.setitem(POLICIES, "graph", (3600, 1, None))
    monkeypatch.setitem(POLICIES, "chat", (3600, 1, None))
    try:
        document_id = "a" * 64  # The isolated access fixture owns this test document.
        first_graph = test_client.post(f"/papers/{document_id}/graph")
        assert first_graph.status_code == 503
        second_graph = test_client.post(f"/papers/{document_id}/graph/rebuild")
        assert second_graph.status_code == 429

        first_chat = test_client.post("/invoke", json={"message": "First"})
        assert first_chat.status_code == 200
        second_chat = test_client.post("/stream", json={"message": "Second"})
        assert second_chat.status_code == 429
        assert mock_agent.ainvoke.await_count == 1
    finally:
        app.dependency_overrides.pop(graph_extraction_is_configured, None)
        app.dependency_overrides.pop(get_quota_store, None)


def test_untrusted_proxy_header_cannot_change_client_ip(monkeypatch) -> None:
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/papers",
            "headers": [(b"x-real-ip", b"203.0.113.8")],
            "client": ("192.0.2.10", 1234),
        }
    )
    assert client_ip(request) == "192.0.2.10"
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_TRUSTED_PROXY_CIDRS", "192.0.2.0/24")
    assert client_ip(request) == "203.0.113.8"


def test_global_upload_budget_cannot_be_evaded_by_changing_ip(tmp_path, monkeypatch) -> None:
    store = QuotaStore(tmp_path)
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_QUOTAS_ENABLED", True)
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_GLOBAL_UPLOADS_PER_DAY", 1)

    def request(ip: str) -> Request:
        return Request(
            {"type": "http", "method": "POST", "path": "/papers", "headers": [], "client": (ip, 1234)}
        )

    enforce_quota(request("192.0.2.10"), store, "upload")
    with pytest.raises(HTTPException) as error:
        enforce_quota(request("198.51.100.10"), store, "upload")
    assert error.value.status_code == 429
    assert int(error.value.headers["Retry-After"]) > 0
