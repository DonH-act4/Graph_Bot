"""Email codes create accounts only after a single successful verification."""

import json
import sqlite3
from datetime import UTC, datetime, timedelta

import httpx

from evidencegraph.accounts import AccountStore
from evidencegraph.email_delivery import EmailDeliveryError, ResendVerificationSender


def test_email_registration_requires_a_single_valid_code(tmp_path) -> None:
    store = AccountStore(tmp_path)
    code = store.begin_email_registration("Researcher", "researcher@example.test", "long-password-123")
    assert code is not None
    assert store.authenticate("Researcher", "long-password-123") is None
    assert store.complete_email_registration("Researcher", "wrong") is None
    account_id = store.complete_email_registration("researcher", code)
    assert account_id is not None
    assert store.authenticate("Researcher", "long-password-123") == account_id
    assert store.complete_email_registration("Researcher", code) is None
    assert store.begin_email_registration("Other", "researcher@example.test", "long-password-123") is None


def test_email_code_expires_and_five_failed_attempts_invalidate_it(tmp_path) -> None:
    store = AccountStore(tmp_path)
    code = store.begin_email_registration("expired", "expired@example.test", "long-password-123")
    assert code is not None
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "UPDATE pending_registrations SET expires_at = ? WHERE username = ?",
            ((datetime.now(UTC) - timedelta(seconds=1)).isoformat(), "expired"),
        )
    assert store.complete_email_registration("expired", code) is None
    assert store.authenticate("expired", "long-password-123") is None

    code = store.begin_email_registration("limited", "limited@example.test", "long-password-123")
    assert code is not None
    for _ in range(5):
        assert store.complete_email_registration("limited", "wrong") is None
    assert store.complete_email_registration("limited", code) is None


def test_retry_invalidates_previous_code_and_failed_delivery_cleans_only_its_own(tmp_path) -> None:
    store = AccountStore(tmp_path)
    first = store.begin_email_registration("again", "again@example.test", "long-password-123")
    second = store.begin_email_registration("again", "again@example.test", "long-password-123")
    assert first is not None and second is not None and first != second
    store.discard_email_registration("again", first)
    assert store.complete_email_registration("again", first) is None
    assert store.complete_email_registration("again", second) is not None


def test_resend_sender_uses_server_key_and_never_returns_provider_body() -> None:
    captured: list[httpx.Request] = []

    def accept(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"id": "accepted"})

    with httpx.Client(transport=httpx.MockTransport(accept)) as client:
        sender = ResendVerificationSender("private-test-key", "verify@example.test", client=client)
        sender.send_verification("reader@example.test", "test-code")
    assert len(captured) == 1
    assert captured[0].url == "https://api.resend.com/emails"
    assert captured[0].headers["Authorization"] == "Bearer private-test-key"
    body = json.loads(captured[0].content)
    assert body["to"] == ["reader@example.test"]
    assert "test-code" in body["text"]

    def reject(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="provider-private-detail")

    with httpx.Client(transport=httpx.MockTransport(reject)) as client:
        sender = ResendVerificationSender("private-test-key", "verify@example.test", client=client)
        try:
            sender.send_verification("reader@example.test", "test-code")
        except EmailDeliveryError as exc:
            assert "provider-private-detail" not in str(exc)
        else:
            raise AssertionError("Provider rejection must fail")
