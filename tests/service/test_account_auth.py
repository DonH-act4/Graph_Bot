"""Account-mode access is enforced by the API, not only by the React shell."""

from core import settings
from evidencegraph.accounts import ACCOUNT_COOKIE, AccountStore
from service import app
from service.auth import get_account_store
from service.papers import get_guest_identity


def test_register_login_logout_and_guest_chat_gate(
    test_client, mock_agent, isolated_test_paper_access, tmp_path, monkeypatch
):
    account_store = AccountStore(tmp_path / "accounts")
    app.dependency_overrides[get_account_store] = lambda: account_store
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_REQUIRE_LOGIN_FOR_CHAT", True)
    try:
        guest = test_client.get("/session")
        assert guest.status_code == 200
        assert guest.json() == {
            "user_id": "test-service-guest",
            "identity_enforced": True,
            "chat_requires_login": True,
            "authenticated": False,
            "email_verification_required": False,
        }
        assert test_client.get("/papers/configuration").status_code == 200
        assert test_client.get("/papers/" + "a" * 64).status_code != 401
        assert test_client.get("/threads", params={"user_id": "test-service-guest"}).status_code == 401
        assert test_client.post(
            "/stream", json={"user_id": "test-service-guest", "message": "Ask"}
        ).status_code == 401
        assert test_client.put(
            "/conversations/new", json={"user_id": "test-service-guest"}
        ).status_code == 401
        mock_agent.ainvoke.assert_not_awaited()

        credentials = {"username": "Researcher_1", "password": "a-long-test-password"}
        registered = test_client.post("/auth/register", json=credentials)
        assert registered.status_code == 201
        assert ACCOUNT_COOKIE in registered.cookies
        assert "httponly" in registered.headers["set-cookie"].lower()
        account_id = registered.json()["user_id"]
        assert test_client.get("/session").json()["user_id"] == account_id
        assert test_client.get("/session").json()["authenticated"] is True
        assert test_client.get("/threads", params={"user_id": account_id}).status_code == 200
        assert test_client.get("/threads", params={"user_id": "another-person"}).status_code == 404
        assert test_client.put(
            "/conversations/account-thread",
            json={"user_id": account_id, "draft_message": "Ask the graph"},
        ).status_code == 200
        assert test_client.get(
            "/conversations/account-thread", params={"user_id": account_id}
        ).json()["draft_message"] == "Ask the graph"
        assert test_client.post(
            "/history", json={"thread_id": "account-thread", "user_id": account_id}
        ).status_code == 200

        assert test_client.post("/auth/logout").status_code == 204
        assert account_store.resolve_session(registered.cookies[ACCOUNT_COOKIE]) is None
        assert test_client.get("/threads", params={"user_id": account_id}).status_code == 401
        wrong = test_client.post(
            "/auth/login", json={**credentials, "password": "incorrect-password"}
        )
        assert wrong.status_code == 401
        logged_in = test_client.post("/auth/login", json=credentials)
        assert logged_in.status_code == 200
        assert logged_in.json()["user_id"] == account_id
    finally:
        app.dependency_overrides.pop(get_account_store, None)


def test_published_tutorial_remains_readable_without_login(
    test_client, isolated_conversation_store, mock_agent, monkeypatch
):
    from schema import ConversationUpdate

    monkeypatch.setattr(settings, "EVIDENCEGRAPH_REQUIRE_LOGIN_FOR_CHAT", True)
    isolated_conversation_store.save(
        "tutorial-thread", ConversationUpdate(user_id="sample-author", document_id="a" * 64)
    )
    isolated_conversation_store.publish_showcase(
        "tutorial-thread", "sample-author", "Research tutorial", "Public sample", "test-model"
    )
    assert test_client.get("/showcase").status_code == 200
    assert test_client.post(
        "/history", json={"thread_id": "tutorial-thread", "user_id": "sample-author"}
    ).status_code == 200
    assert test_client.post(
        "/stream", json={"thread_id": "tutorial-thread", "user_id": "sample-author", "message": "Edit"}
    ).status_code == 401


def test_duplicate_username_and_login_throttle(test_client, tmp_path, monkeypatch):
    account_store = AccountStore(tmp_path / "accounts")
    app.dependency_overrides[get_account_store] = lambda: account_store
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_REQUIRE_LOGIN_FOR_CHAT", True)
    credentials = {"username": "Reviewer", "password": "another-long-password"}
    try:
        assert test_client.post("/auth/register", json=credentials).status_code == 201
        assert test_client.post(
            "/auth/register", json={**credentials, "username": "reviewer"}
        ).status_code == 409
        test_client.post("/auth/logout")
        for _ in range(5):
            assert test_client.post(
                "/auth/login", json={**credentials, "password": "wrong-password"}
            ).status_code == 401
        assert test_client.post("/auth/login", json=credentials).status_code == 401
    finally:
        app.dependency_overrides.pop(get_account_store, None)


def test_registration_transfers_browser_papers_to_account(
    test_client, isolated_test_paper_access, tmp_path, monkeypatch
):
    access = isolated_test_paper_access
    account_store = AccountStore(tmp_path / "accounts")
    app.dependency_overrides.pop(get_guest_identity, None)
    app.dependency_overrides[get_account_store] = lambda: account_store
    monkeypatch.setattr(settings, "EVIDENCEGRAPH_REQUIRE_LOGIN_FOR_CHAT", True)
    try:
        guest_id = test_client.get("/session").json()["user_id"]
        document_id = "c" * 64
        access.grant(guest_id, document_id)
        assert access.owns(guest_id, document_id)
        registered = test_client.post(
            "/auth/register",
            json={"username": "paper-owner", "password": "long-enough-password"},
        )
        assert registered.status_code == 201
        account_id = registered.json()["user_id"]
        assert access.owns(account_id, document_id)
        assert not access.owns(guest_id, document_id)
        access.grant(guest_id, "d" * 64)
        assert test_client.get("/session").json()["user_id"] == account_id
        assert access.owns(guest_id, "d" * 64)
        assert not access.owns(account_id, "d" * 64)
        assert test_client.post("/auth/logout").status_code == 204
        assert test_client.get("/session").json()["user_id"] == guest_id
        assert not access.can_read(guest_id, document_id)
    finally:
        app.dependency_overrides.pop(get_account_store, None)
