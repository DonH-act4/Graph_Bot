import os
from pathlib import Path
from unittest.mock import patch

import pytest

from core import settings


@pytest.fixture(autouse=True)
def isolate_browser_security_settings(monkeypatch):
    """Keep local demo .env switches from changing unrelated test expectations."""
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_REQUIRE_LOGIN_FOR_CHAT", False)
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_ENFORCE_SESSION_IDENTITY", False)
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_TRUSTED_ORIGINS", "http://127.0.0.1:3000")
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_QUOTAS_ENABLED", False)
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_PUBLIC_MODE", False)
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_ADMIN_ACCOUNT_ID", None)
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_EMAIL_VERIFICATION_REQUIRED", False)


def pytest_addoption(parser):
    parser.addoption(
        "--run-docker", action="store_true", default=False, help="run docker integration tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "docker: mark test as requiring docker containers")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-docker"):
        skip_docker = pytest.mark.skip(reason="need --run-docker option to run")
        for item in items:
            if "docker" in item.keywords:
                item.add_marker(skip_docker)


@pytest.fixture
def mock_env():
    """Isolate app settings while keeping Windows home-directory lookup available."""
    home_env = {"USERPROFILE": str(Path.home())} if os.name == "nt" else {}
    with patch.dict(os.environ, home_env, clear=True):
        yield
