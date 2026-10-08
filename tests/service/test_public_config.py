"""Public mode refuses known unsafe combinations before starting the API."""

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from core import settings
from service import app
from service.public_config import validate_public_configuration


def test_local_mode_needs_no_public_settings(monkeypatch) -> None:
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_COOKIE_SECURE", False)
    validate_public_configuration()


def test_public_mode_rejects_http_and_insecure_cookies(monkeypatch) -> None:
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_PUBLIC_MODE", True)
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_COOKIE_SECURE", False)
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_REQUIRE_LOGIN_FOR_CHAT", False)
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_QUOTAS_ENABLED", False)
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_TRUSTED_ORIGINS", "http://example.test")
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_TRUSTED_PROXY_CIDRS", "")
    with pytest.raises(RuntimeError, match="COOKIE_SECURE.*REQUIRE_LOGIN.*QUOTAS.*HTTPS.*PROXY"):
        with TestClient(app):
            pass


def test_public_mode_accepts_explicit_https_proxy_configuration(monkeypatch) -> None:
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_PUBLIC_MODE", True)
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_COOKIE_SECURE", True)
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_REQUIRE_LOGIN_FOR_CHAT", True)
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_QUOTAS_ENABLED", True)
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_TRUSTED_ORIGINS", "https://papers.example.test")
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_TRUSTED_PROXY_CIDRS", "192.0.2.4/32")
    monkeypatch.setattr(settings, "AUTH_SECRET", None)
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_EMAIL_VERIFICATION_REQUIRED", True)
    monkeypatch.setattr(settings, "RESEND_API_KEY", SecretStr("test-key"))
    monkeypatch.setattr(settings, "RESEND_FROM_EMAIL", "verify@example.com")
    validate_public_configuration()


def test_public_mode_rejects_missing_email_delivery(monkeypatch) -> None:
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_PUBLIC_MODE", True)
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_COOKIE_SECURE", True)
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_REQUIRE_LOGIN_FOR_CHAT", True)
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_QUOTAS_ENABLED", True)
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_TRUSTED_ORIGINS", "https://papers.example.test")
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_TRUSTED_PROXY_CIDRS", "192.0.2.4/32")
    monkeypatch.setattr(settings, "AUTH_SECRET", None)
    monkeypatch.setattr(settings, "RESEND_API_KEY", None)
    monkeypatch.setattr(settings, "RESEND_FROM_EMAIL", None)
    with pytest.raises(RuntimeError, match="EMAIL_VERIFICATION_REQUIRED.*RESEND_API_KEY"):
        validate_public_configuration()
