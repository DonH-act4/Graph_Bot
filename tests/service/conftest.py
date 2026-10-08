import hashlib
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.types import StateSnapshot

from evidencegraph.access import PaperAccessStore
from evidencegraph.conversations import ConversationStore
from service import app
from service.conversations import get_conversation_store
from service.papers import get_guest_identity, get_paper_access_store


@pytest.fixture(autouse=True)
def isolated_conversation_store(tmp_path):
    """Keep chat API tests from writing workspace metadata into the developer's data."""
    store = ConversationStore(tmp_path / "conversations")
    app.dependency_overrides[get_conversation_store] = lambda: store
    yield store
    app.dependency_overrides.pop(get_conversation_store, None)


@pytest.fixture(autouse=True)
def isolated_test_paper_access(tmp_path):
    """Give legacy service fixtures explicit ownership of their fake papers."""
    access = PaperAccessStore(tmp_path / "access")
    test_guest = "test-service-guest"
    access.grant(test_guest, "a" * 64)
    private_pdf = b"%PDF-1.4\nprivate deletion fixture"
    access.grant(test_guest, hashlib.sha256(private_pdf).hexdigest())
    app.dependency_overrides[get_paper_access_store] = lambda: access
    app.dependency_overrides[get_guest_identity] = lambda: test_guest
    yield access
    app.dependency_overrides.pop(get_guest_identity, None)
    app.dependency_overrides.pop(get_paper_access_store, None)


@pytest.fixture
def test_client():
    """Fixture to create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_agent():
    """Fixture to create a mock agent that can be configured for different test scenarios."""
    agent_mock = AsyncMock()
    agent_mock.ainvoke = AsyncMock(
        return_value=[("values", {"messages": [AIMessage(content="Test response")]})]
    )
    agent_mock.aget_state = AsyncMock(
        return_value=StateSnapshot(
            values={},
            next=(),
            config={},
            metadata=None,
            created_at=None,
            parent_config=None,
            tasks=(),
            interrupts=(),
        )
    )
    # Tests that exercise checkpointer reads set this explicitly; otherwise AsyncMock would
    # auto-create an attribute that looks like a working checkpointer.
    agent_mock.checkpointer = None
    with patch("service.service.get_agent", Mock(return_value=agent_mock)):
        yield agent_mock


@pytest.fixture
def mock_settings(mock_env):
    """Fixture to ensure settings are clean for each test."""
    with patch("service.service.settings") as mock_settings:
        yield mock_settings


@pytest.fixture
def mock_httpx():
    """Patch httpx.stream and httpx.get to use our test client."""

    with TestClient(app) as client:

        def mock_stream(method: str, url: str, **kwargs):
            # Strip the base URL since TestClient expects just the path
            path = url.replace("http://0.0.0.0", "")
            return client.stream(method, path, **kwargs)

        def mock_get(url: str, **kwargs):
            # Strip the base URL since TestClient expects just the path
            path = url.replace("http://0.0.0.0", "")
            return client.get(path, **kwargs)

        with patch("httpx.stream", mock_stream):
            with patch("httpx.get", mock_get):
                yield
