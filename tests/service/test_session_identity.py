"""Strict mode binds private conversations to the server-issued guest session."""

from core import settings
from schema import ConversationUpdate


def test_session_reports_server_identity_and_enforcement(test_client, monkeypatch):
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_ENFORCE_SESSION_IDENTITY", True)
    response = test_client.get("/session")
    assert response.status_code == 200
    assert response.json() == {
        "user_id": "test-service-guest",
        "identity_enforced": True,
        "chat_requires_login": False,
        "authenticated": False,
        "is_admin": False,
        "email_verification_required": False,
    }


def test_strict_mode_rejects_spoofed_conversation_identity(
    test_client, mock_agent, isolated_conversation_store, monkeypatch
):
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_ENFORCE_SESSION_IDENTITY", True)
    isolated_conversation_store.save("victim-thread", ConversationUpdate(user_id="victim"))

    assert test_client.get(
        "/conversations/victim-thread", params={"user_id": "victim"}
    ).status_code == 404
    assert test_client.put(
        "/conversations/new-thread", json={"user_id": "victim"}
    ).status_code == 404
    assert test_client.post(
        "/history", json={"thread_id": "victim-thread", "user_id": "victim"}
    ).status_code == 404
    assert test_client.get("/threads", params={"user_id": "victim"}).status_code == 404
    assert test_client.delete(
        "/conversations/victim-thread", params={"user_id": "victim"}
    ).status_code == 404
    assert test_client.post(
        "/invoke", json={"thread_id": "victim-thread", "user_id": "victim", "message": "Read"}
    ).status_code == 404
    assert test_client.post(
        "/stream", json={"thread_id": "victim-thread", "user_id": "victim", "message": "Read"}
    ).status_code == 404
    assert isolated_conversation_store.get("victim-thread") is not None
    mock_agent.ainvoke.assert_not_awaited()


def test_strict_mode_rejects_foreign_thread_even_with_own_user_id(
    test_client, mock_agent, isolated_conversation_store, monkeypatch
):
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_ENFORCE_SESSION_IDENTITY", True)
    isolated_conversation_store.save("victim-thread", ConversationUpdate(user_id="victim"))

    response = test_client.post(
        "/invoke",
        json={
            "thread_id": "victim-thread",
            "user_id": "test-service-guest",
            "message": "Continue their conversation",
        },
    )
    assert response.status_code == 404
    mock_agent.ainvoke.assert_not_awaited()


def test_strict_mode_keeps_published_tutorial_history_readable(
    test_client, mock_agent, isolated_conversation_store, monkeypatch
):
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_ENFORCE_SESSION_IDENTITY", True)
    isolated_conversation_store.save(
        "tutorial", ConversationUpdate(user_id="author", document_id="a" * 64)
    )
    isolated_conversation_store.publish_showcase(
        "tutorial", "author", "Research tutorial", "Read-only example", "test-model"
    )

    response = test_client.post(
        "/history", json={"thread_id": "tutorial", "user_id": "author"}
    )
    assert response.status_code == 200
    assert response.json()["conversation"]["user_id"] == "author"
    assert test_client.post(
        "/invoke", json={"thread_id": "tutorial", "user_id": "author", "message": "Edit"}
    ).status_code == 404


def test_strict_mode_does_not_read_agent_state_for_unknown_thread(
    test_client, mock_agent, monkeypatch
):
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_ENFORCE_SESSION_IDENTITY", True)
    response = test_client.post(
        "/history", json={"thread_id": "unknown", "user_id": "test-service-guest"}
    )
    assert response.status_code == 200
    assert response.json() == {"messages": [], "conversation": None}
    mock_agent.aget_state.assert_not_awaited()
