"""The operations summary belongs only to the configured existing account."""

from uuid import UUID

from core import settings
from evidencegraph.accounts import ACCOUNT_COOKIE, AccountStore
from evidencegraph.quotas import QuotaStore
from service import app
from service.auth import get_account_store


def test_admin_summary_rejects_guests_and_other_accounts(test_client, tmp_path, monkeypatch) -> None:
    store = AccountStore(tmp_path)
    owner_id = store.register("don", "long-owner-test-password")
    visitor_id = store.register("visitor", "long-visitor-test-password")
    assert owner_id is not None and visitor_id is not None
    app.dependency_overrides[get_account_store] = lambda: store
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_ADMIN_ACCOUNT_ID", UUID(owner_id))
    try:
        assert test_client.get("/admin/summary").status_code == 401
        test_client.cookies.set(ACCOUNT_COOKIE, store.issue_session(visitor_id))
        assert test_client.get("/admin/summary").status_code == 403
        test_client.cookies.set(ACCOUNT_COOKIE, store.issue_session(owner_id))
        monkeypatch.setattr(settings, "EVIDENCEGRAPH_ADMIN_ACCOUNT_ID", None)
        assert test_client.get("/admin/summary").status_code == 403
    finally:
        app.dependency_overrides.pop(get_account_store, None)


def test_admin_summary_returns_only_aggregate_counts(test_client, tmp_path, monkeypatch) -> None:
    store = AccountStore(tmp_path)
    owner_id = store.register("don", "long-owner-test-password")
    assert owner_id is not None
    code = store.begin_email_registration("reader", "reader@example.com", "long-reader-password")
    assert code is not None
    assert store.complete_email_registration("reader", code) is not None
    quotas = QuotaStore(tmp_path)
    assert quotas.consume("upload", [("ip:visitor", 10), ("site:all", 30)], 86400) is None
    app.dependency_overrides[get_account_store] = lambda: store
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_ADMIN_ACCOUNT_ID", UUID(owner_id))
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_QUOTAS_ENABLED", True)
    test_client.cookies.set(ACCOUNT_COOKIE, store.issue_session(owner_id))
    try:
        response = test_client.get("/admin/summary")
        assert response.status_code == 200
        assert response.json() == {
            "accounts_total": 2,
            "verified_email_accounts": 1,
            "uploads_allowed_today": 1,
            "graphs_allowed_today": 0,
            "chats_allowed_this_hour": 0,
            "verification_emails_allowed_today": 0,
            "quotas_enabled": True,
        }
        assert "reader@example.com" not in response.text
        assert "don" not in response.text
    finally:
        app.dependency_overrides.pop(get_account_store, None)


def test_admin_can_review_and_reversibly_ban_an_account(test_client, tmp_path, monkeypatch) -> None:
    store = AccountStore(tmp_path)
    owner_id = store.register("don", "long-owner-test-password")
    visitor_id = store.register("reader", "long-reader-test-password")
    assert owner_id is not None and visitor_id is not None
    visitor_token = store.issue_session(visitor_id)
    owner_token = store.issue_session(owner_id)
    quotas = QuotaStore(tmp_path)
    assert quotas.consume(
        "upload", [(f"account:{visitor_id}", 20), ("site:all", 30)], 86400
    ) is None
    app.dependency_overrides[get_account_store] = lambda: store
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_ADMIN_ACCOUNT_ID", UUID(owner_id))
    try:
        assert test_client.get("/admin/accounts").status_code == 401
        test_client.cookies.set(ACCOUNT_COOKIE, visitor_token)
        assert test_client.get("/admin/accounts").status_code == 403
        assert test_client.put(f"/admin/accounts/{owner_id}/ban", json={}).status_code == 403

        test_client.cookies.set(ACCOUNT_COOKIE, owner_token)
        listing = test_client.get("/admin/accounts")
        assert listing.status_code == 200
        items = {item["username"]: item for item in listing.json()["accounts"]}
        assert items["don"]["is_self"] is True
        assert items["reader"]["uploads_allowed_today"] == 1
        assert items["reader"]["banned"] is False
        assert "password_hash" not in listing.text
        assert test_client.get("/session").json()["is_admin"] is True
        assert test_client.put(f"/admin/accounts/{owner_id}/ban", json={}).status_code == 409
        assert test_client.put(
            f"/admin/accounts/{visitor_id}/ban",
            json={},
            headers={"Origin": "https://untrusted.example"},
        ).status_code == 403
        assert store.resolve_session(visitor_token) == visitor_id

        assert test_client.put(
            f"/admin/accounts/{visitor_id}/ban", json={"reason": "Manual test"}
        ).status_code == 204
        assert store.resolve_session(visitor_token) is None
        assert store.authenticate("reader", "long-reader-test-password") is None
        assert next(
            item for item in test_client.get("/admin/accounts").json()["accounts"]
            if item["account_id"] == visitor_id
        )["banned"] is True
        assert test_client.delete(f"/admin/accounts/{visitor_id}/ban").status_code == 204
        assert store.authenticate("reader", "long-reader-test-password") == visitor_id
        assert store.resolve_session(visitor_token) is None
    finally:
        app.dependency_overrides.pop(get_account_store, None)
