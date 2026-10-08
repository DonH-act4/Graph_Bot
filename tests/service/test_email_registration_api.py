"""The registration API does not authenticate until email possession is proven."""

import sqlite3

from core import settings
from evidencegraph.accounts import ACCOUNT_COOKIE, AccountStore
from evidencegraph.email_delivery import EmailDeliveryError
from evidencegraph.quotas import QuotaStore
from service import app
from service.auth import get_account_store, get_verification_sender
from service.quotas import get_quota_store


class FakeSender:
    def __init__(self, *, fail: bool = False) -> None:
        self.sent: list[tuple[str, str]] = []
        self.fail = fail

    def send_verification(self, email: str, code: str) -> None:
        if self.fail:
            raise EmailDeliveryError("test delivery failed")
        self.sent.append((email, code))


def test_registration_waits_for_email_verification(test_client, tmp_path, monkeypatch) -> None:
    store = AccountStore(tmp_path)
    sender = FakeSender()
    app.dependency_overrides[get_account_store] = lambda: store
    app.dependency_overrides[get_verification_sender] = lambda: sender
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_EMAIL_VERIFICATION_REQUIRED", True)
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_REQUIRE_LOGIN_FOR_CHAT", True)
    credentials = {
        "username": "Researcher", "email": "reader@example.com", "password": "long-password-123"
    }
    try:
        assert test_client.post("/auth/register", json={**credentials, "email": "not-an-email"}).status_code == 422
        assert test_client.post("/auth/register", json={"username": "Researcher", "password": "long-password-123"}).status_code == 422
        pending = test_client.post("/auth/register", json=credentials)
        assert pending.status_code == 202
        assert pending.json() == {"status": "verification_required"}
        assert ACCOUNT_COOKIE not in pending.cookies
        assert store.authenticate("Researcher", credentials["password"]) is None
        assert test_client.post("/auth/login", json=credentials).status_code == 401
        assert sender.sent[0][0] == "reader@example.com"
        assert test_client.post("/auth/verify-email", json={"username": "Researcher", "code": "wrong"}).status_code == 400
        verified = test_client.post(
            "/auth/verify-email", json={"username": "Researcher", "code": sender.sent[0][1]}
        )
        assert verified.status_code == 200
        assert ACCOUNT_COOKIE in verified.cookies
        assert test_client.get("/session").json()["authenticated"] is True
        assert test_client.post(
            "/auth/verify-email", json={"username": "Researcher", "code": sender.sent[0][1]}
        ).status_code == 400
    finally:
        app.dependency_overrides.pop(get_account_store, None)
        app.dependency_overrides.pop(get_verification_sender, None)


def test_email_delivery_failure_does_not_leave_a_pending_registration(
    test_client, tmp_path, monkeypatch
) -> None:
    store = AccountStore(tmp_path)
    app.dependency_overrides[get_account_store] = lambda: store
    app.dependency_overrides[get_verification_sender] = lambda: FakeSender(fail=True)
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_EMAIL_VERIFICATION_REQUIRED", True)
    try:
        response = test_client.post(
            "/auth/register",
            json={
                "username": "mail-failed", "email": "failed@example.com",
                "password": "long-password-123",
            },
        )
        assert response.status_code == 503
        assert "test delivery failed" not in response.text
        with sqlite3.connect(store.database_path) as connection:
            assert connection.execute("SELECT COUNT(*) FROM pending_registrations").fetchone() == (0,)
        assert store.authenticate("mail-failed", "long-password-123") is None
    finally:
        app.dependency_overrides.pop(get_account_store, None)
        app.dependency_overrides.pop(get_verification_sender, None)


def test_global_mail_budget_blocks_second_provider_call(test_client, tmp_path, monkeypatch) -> None:
    store = AccountStore(tmp_path / "accounts")
    sender = FakeSender()
    app.dependency_overrides[get_account_store] = lambda: store
    app.dependency_overrides[get_verification_sender] = lambda: sender
    app.dependency_overrides[get_quota_store] = lambda: QuotaStore(tmp_path / "quotas")
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_EMAIL_VERIFICATION_REQUIRED", True)
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_QUOTAS_ENABLED", True)
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_GLOBAL_VERIFICATION_EMAILS_PER_DAY", 1)
    try:
        first = test_client.post(
            "/auth/register",
            json={"username": "first", "email": "first@example.com", "password": "long-password-123"},
        )
        second = test_client.post(
            "/auth/register",
            json={"username": "second", "email": "second@example.com", "password": "long-password-123"},
        )
        assert first.status_code == 202
        assert second.status_code == 429
        assert int(second.headers["Retry-After"]) > 0
        assert len(sender.sent) == 1
        with sqlite3.connect(store.database_path) as connection:
            assert connection.execute("SELECT COUNT(*) FROM pending_registrations").fetchone() == (1,)
    finally:
        app.dependency_overrides.pop(get_account_store, None)
        app.dependency_overrides.pop(get_verification_sender, None)
        app.dependency_overrides.pop(get_quota_store, None)
