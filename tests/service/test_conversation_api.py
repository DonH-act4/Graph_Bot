"""Conversation workspace APIs and checkpoint durability."""

import asyncio
import hashlib
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph

from evidencegraph.conversations import ConversationStore
from evidencegraph.papers import PaperStore
from schema import ConversationUpdate
from service import app
from service.papers import get_paper_store
from service.service import _checkpoint_document_ids

DOCUMENT_ID = "a" * 64
BLOCK_ID = "blk_" + "b" * 24
PDF = b"%PDF-1.4\nprivate deletion fixture"
PRIVATE_DOCUMENT_ID = hashlib.sha256(PDF).hexdigest()


class PaperFixture:
    def get(self, document_id: str):
        return object() if document_id == DOCUMENT_ID else None

    def get_block(self, document_id: str, block_id: str):
        if document_id == DOCUMENT_ID and block_id == BLOCK_ID:
            return SimpleNamespace(block_id=block_id, text="Source passage", locations=[])
        return None

    def deletion_state(self, _document_id: str):
        return None

    def request_deletion(self, _document_id: str) -> None:
        pass

    def finish_deletion(self, _document_id: str) -> None:
        pass


@pytest.fixture
def papers():
    app.dependency_overrides[get_paper_store] = PaperFixture
    yield
    app.dependency_overrides.pop(get_paper_store, None)


def test_save_restore_upload_only_conversation(
    test_client, mock_agent, isolated_conversation_store, papers
) -> None:
    saved = test_client.put(
        "/conversations/upload-thread",
        json={
            "user_id": "user",
            "document_id": DOCUMENT_ID,
            "document_name": "My paper.pdf",
            "selected_block_ids": [BLOCK_ID],
            "draft_message": "Explain this passage",
        },
    )
    assert saved.status_code == 200
    reopened = ConversationStore(isolated_conversation_store.root)
    assert reopened.get("upload-thread").draft_message == "Explain this passage"
    loaded = test_client.get("/conversations/upload-thread", params={"user_id": "user"})
    assert loaded.json() == saved.json()
    history = test_client.post("/history", json={"thread_id": "upload-thread", "user_id": "user"})
    assert history.status_code == 200
    assert history.json()["messages"] == []
    assert history.json()["conversation"] == saved.json()
    threads = test_client.get("/threads", params={"user_id": "user"}).json()["threads"]
    assert len(threads) == 1
    assert threads[0]["document_id"] == DOCUMENT_ID
    assert threads[0]["title"] == "My paper.pdf"


def test_unknown_history_is_empty_not_server_error(test_client, mock_agent) -> None:
    response = test_client.post("/history", json={"thread_id": "new-thread", "user_id": "user"})
    assert response.status_code == 200
    assert response.json() == {"messages": [], "conversation": None}


def test_legacy_checkpoint_paper_references_are_conservative() -> None:
    checkpoint = SimpleNamespace(checkpoint={"channel_values": {"messages": [
        HumanMessage(content="Question", additional_kwargs={"document_id": PRIVATE_DOCUMENT_ID}),
    ]}})
    assert _checkpoint_document_ids(checkpoint) == {PRIVATE_DOCUMENT_ID}
    checkpoint.checkpoint["channel_values"]["messages"].append(
        HumanMessage(content="Second paper", additional_kwargs={"evidence_context": {
            "document_id": DOCUMENT_ID,
        }})
    )
    assert _checkpoint_document_ids(checkpoint) == {PRIVATE_DOCUMENT_ID, DOCUMENT_ID}


def test_rejects_missing_paper_and_invalid_selection(test_client, papers) -> None:
    for payload, code in (
        ({"user_id": "user", "document_id": "c" * 64}, 404),
        (
            {"user_id": "user", "document_id": DOCUMENT_ID, "selected_block_ids": ["blk_missing"]},
            404,
        ),
        ({"user_id": "user", "selected_block_ids": [BLOCK_ID]}, 422),
        (
            {
                "user_id": "user",
                "document_id": DOCUMENT_ID,
                "selected_block_ids": [BLOCK_ID, BLOCK_ID],
            },
            422,
        ),
    ):
        response = test_client.put("/conversations/thread", json=payload)
        assert response.status_code == code


def test_rejects_wrong_user_for_workspace_and_history(test_client, mock_agent) -> None:
    assert test_client.put("/conversations/thread", json={"user_id": "owner"}).status_code == 200
    assert test_client.get("/conversations/thread", params={"user_id": "other"}).status_code == 404
    assert test_client.put("/conversations/thread", json={"user_id": "other"}).status_code == 404
    assert (
        test_client.post("/history", json={"thread_id": "thread", "user_id": "other"}).status_code
        == 404
    )
    assert test_client.get("/threads", params={"user_id": "other"}).json() == {"threads": []}
    assert test_client.delete("/conversations/thread", params={"user_id": "other"}).status_code == 404


def test_delete_upload_only_conversation_and_reject_late_save(
    test_client, mock_agent, isolated_conversation_store, papers
) -> None:
    assert test_client.put(
        "/conversations/upload-thread",
        json={"user_id": "user", "document_id": DOCUMENT_ID, "draft_message": "Draft"},
    ).status_code == 200
    deleted = test_client.delete("/conversations/upload-thread", params={"user_id": "user"})
    assert deleted.status_code == 204
    assert isolated_conversation_store.is_deleted("upload-thread")
    assert isolated_conversation_store.get("upload-thread") is None
    assert test_client.get("/threads", params={"user_id": "user"}).json() == {"threads": []}
    assert test_client.post(
        "/history", json={"thread_id": "upload-thread", "user_id": "user"}
    ).status_code == 404
    assert test_client.put(
        "/conversations/upload-thread", json={"user_id": "user", "document_id": DOCUMENT_ID}
    ).status_code == 404
    assert test_client.delete(
        "/conversations/upload-thread", params={"user_id": "user"}
    ).status_code == 404


def test_delete_private_paper_removes_pdf_graph_and_workspace(
    test_client, mock_agent, isolated_conversation_store, tmp_path
) -> None:
    store = PaperStore(tmp_path / "papers")
    store.accept(PDF)
    folder = store.source_path(PRIVATE_DOCUMENT_ID).parent
    (folder / "parsed.json").write_text("parsed")
    (folder / "graph.json").write_text("graph")
    (folder / "graphs").mkdir()
    (folder / "graphs" / "000001.json").write_text("version")
    app.dependency_overrides[get_paper_store] = lambda: store
    try:
        assert test_client.put(
            "/conversations/private",
            json={"user_id": "owner", "document_id": PRIVATE_DOCUMENT_ID},
        ).status_code == 200
        assert test_client.delete(
            "/conversations/private", params={"user_id": "owner"}
        ).status_code == 204
        assert not folder.exists()
        assert store.deletion_state(PRIVATE_DOCUMENT_ID) == "deleted"
        assert isolated_conversation_store.get("private") is None
        assert test_client.get(f"/papers/{PRIVATE_DOCUMENT_ID}").status_code == 404
        assert test_client.get(f"/papers/{PRIVATE_DOCUMENT_ID}/source").status_code == 404
        assert test_client.get(f"/papers/{PRIVATE_DOCUMENT_ID}/graph/status").status_code == 404
        assert test_client.get(f"/papers/{PRIVATE_DOCUMENT_ID}/graph/versions").status_code == 404
        assert test_client.get(f"/papers/{PRIVATE_DOCUMENT_ID}/graph/reviews").status_code == 404
        # The same exact PDF can be uploaded again in a new conversation.
        assert test_client.post(
            "/papers", content=PDF, headers={"Content-Type": "application/pdf"}
        ).status_code == 202
    finally:
        app.dependency_overrides.pop(get_paper_store, None)


def test_delete_refuses_shared_paper_without_removing_anything(
    test_client, mock_agent, isolated_conversation_store, tmp_path
) -> None:
    store = PaperStore(tmp_path / "papers")
    store.accept(PDF)
    app.dependency_overrides[get_paper_store] = lambda: store
    try:
        for thread in ("first", "second"):
            assert test_client.put(
                f"/conversations/{thread}",
                json={"user_id": "owner", "document_id": PRIVATE_DOCUMENT_ID},
            ).status_code == 200
        response = test_client.delete("/conversations/first", params={"user_id": "owner"})
        assert response.status_code == 409
        assert "another conversation" in response.json()["detail"]
        assert store.source_path(PRIVATE_DOCUMENT_ID).exists()
        assert isolated_conversation_store.get("first") is not None
        assert isolated_conversation_store.get("second") is not None
    finally:
        app.dependency_overrides.pop(get_paper_store, None)


@pytest.mark.asyncio
async def test_delete_cancels_active_chat_request(
    mock_agent, isolated_conversation_store
) -> None:
    started = asyncio.Event()

    async def slow_answer(**_kwargs):
        started.set()
        await asyncio.sleep(30)

    mock_agent.ainvoke.side_effect = slow_answer
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Origin": "http://127.0.0.1:3000"},
    ) as client:
        pending = asyncio.create_task(client.post(
            "/invoke", json={"thread_id": "active", "user_id": "owner", "message": "Wait"}
        ))
        await asyncio.wait_for(started.wait(), timeout=3)
        deleted = await client.delete("/conversations/active", params={"user_id": "owner"})
        assert deleted.status_code == 204
        with pytest.raises(asyncio.CancelledError):
            await pending
    assert isolated_conversation_store.is_deleted("active")


def test_published_showcase_is_read_only(
    test_client, mock_agent, isolated_conversation_store, papers
) -> None:
    assert test_client.get("/showcase").status_code == 200
    assert test_client.get("/showcase").json() is None
    isolated_conversation_store.save(
        "sample", ConversationUpdate(user_id="sample-user", document_id=DOCUMENT_ID)
    )
    published = isolated_conversation_store.publish_showcase(
        "sample", "sample-user", "Research demo", "Real model conversation", "gpt-oss:20b"
    )
    assert test_client.get("/showcase").json() == published.model_dump(mode="json")
    for path in ("/invoke", "/stream"):
        for identity in ({"user_id": "sample-user"}, {}):
            response = test_client.post(
                path, json={"thread_id": "sample", "message": "Overwrite", **identity}
            )
            assert response.status_code == 403
    assert (
        test_client.put("/conversations/sample", json={"user_id": "sample-user"}).status_code == 403
    )
    assert (
        test_client.delete("/conversations/sample", params={"user_id": "sample-user"}).status_code
        == 403
    )
    mock_agent.ainvoke.assert_not_awaited()
    # A user's separate conversation can still reference the demonstration paper.
    assert (
        test_client.put(
            "/conversations/personal", json={"user_id": "person", "document_id": DOCUMENT_ID}
        ).status_code
        == 200
    )
    conflict = test_client.delete("/conversations/personal", params={"user_id": "person"})
    assert conflict.status_code == 409
    assert "tutorial" in conflict.json()["detail"]
    assert isolated_conversation_store.get("personal") is not None


@pytest.mark.asyncio
async def test_real_checkpoints_and_workspace_survive_database_reopen(
    tmp_path, isolated_conversation_store
) -> None:
    database_path = str(tmp_path / "messages.sqlite3")
    builder = StateGraph(MessagesState)
    builder.add_node("reply", lambda state: {"messages": [AIMessage(content="Saved reply")]})
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    isolated_conversation_store.save(
        "persisted",
        ConversationUpdate(user_id="user", document_id=DOCUMENT_ID, document_name="My paper.pdf"),
    )
    async with AsyncSqliteSaver.from_conn_string(database_path) as saver:
        agent = builder.compile(checkpointer=saver)
        with patch("service.service.get_agent", return_value=agent):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
                headers={"Origin": "http://127.0.0.1:3000"},
            ) as client:
                response = await client.post(
                    "/invoke",
                    json={
                        "thread_id": "persisted",
                        "user_id": "user",
                        "message": "What did it find?",
                    },
                )
                assert response.status_code == 200
                assert response.json()["content"] == "Saved reply"
    async with AsyncSqliteSaver.from_conn_string(database_path) as reopened:
        agent = builder.compile(checkpointer=reopened)
        with patch("service.service.get_agent", return_value=agent):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
                headers={"Origin": "http://127.0.0.1:3000"},
            ) as client:
                history = await client.post(
                    "/history", json={"thread_id": "persisted", "user_id": "user"}
                )
                assert history.status_code == 200
                assert [m["content"] for m in history.json()["messages"]] == [
                    "What did it find?",
                    "Saved reply",
                ]
                assert history.json()["conversation"]["document_id"] == DOCUMENT_ID
                assert history.json()["conversation"]["title"] == "What did it find?"
                threads = (await client.get("/threads", params={"user_id": "user"})).json()[
                    "threads"
                ]
                assert len(threads) == 1
                assert threads[0]["thread_id"] == "persisted"
                assert threads[0]["document_id"] == DOCUMENT_ID
                deleted = await client.delete(
                    "/conversations/persisted", params={"user_id": "user"}
                )
                assert deleted.status_code == 204
                assert await reopened.aget_tuple({"configurable": {"thread_id": "persisted"}}) is None
                assert (await client.get("/threads", params={"user_id": "user"})).json() == {
                    "threads": []
                }
                assert (await client.post(
                    "/history", json={"thread_id": "persisted", "user_id": "user"}
                )).status_code == 404
