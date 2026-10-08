from datetime import UTC, datetime, timedelta

import pytest

from evidencegraph.conversations import ConversationStore, merge_thread_summaries
from schema import ConversationState, ConversationUpdate, ThreadSummary

DOCUMENT_ID = "a" * 64
BLOCK_ID = "blk_" + "b" * 24


def test_read_missing_store_does_not_create_files(tmp_path) -> None:
    root = tmp_path / "not-yet-created"
    store = ConversationStore(root)
    assert store.get("thread") is None
    assert store.list("user", "research-assistant", 30) == []
    assert store.get_showcase() is None
    assert not root.exists()


def test_workspace_survives_new_store_and_consumes_draft_on_turn(tmp_path) -> None:
    store = ConversationStore(tmp_path)
    saved = store.save(
        "thread",
        ConversationUpdate(
            user_id="user",
            document_id=DOCUMENT_ID,
            document_name="My paper.pdf",
            selected_block_ids=[BLOCK_ID],
            draft_message="What did this paper find?",
        ),
    )
    reopened = ConversationStore(tmp_path)
    assert reopened.get("thread") == saved
    turned = reopened.record_turn(
        "thread", "user", "research-assistant", "What did this paper find?", None
    )
    assert turned.created_at == saved.created_at
    assert turned.document_id == DOCUMENT_ID
    assert turned.document_name == "My paper.pdf"
    assert turned.title == "What did this paper find?"
    assert turned.selected_block_ids == []
    assert turned.draft_message == ""


def test_workspace_cannot_be_reassigned_to_another_user_or_agent(tmp_path) -> None:
    store = ConversationStore(tmp_path)
    original = store.save("thread", ConversationUpdate(user_id="owner"))
    for update in (
        ConversationUpdate(user_id="another-user"),
        ConversationUpdate(user_id="owner", agent_id="another-agent"),
    ):
        with pytest.raises(PermissionError, match="not found"):
            store.save("thread", update)
    assert store.get("thread") == original


def test_workspace_lists_keep_scope_and_checkpoint_time(tmp_path) -> None:
    now = datetime.now(UTC)
    workspace = ConversationState(
        thread_id="saved",
        user_id="user",
        title=None,
        document_id=DOCUMENT_ID,
        document_name="Uploaded paper.pdf",
        selected_block_ids=[BLOCK_ID],
        created_at=now,
        updated_at=now,
    )
    checkpoint = ThreadSummary(
        thread_id="saved",
        agent_id="research-assistant",
        title="First question",
        updated_at=now + timedelta(seconds=1),
    )
    upload_only = workspace.model_copy(update={"thread_id": "upload-only"})
    merged = merge_thread_summaries([checkpoint], [workspace, upload_only], 30)
    assert [summary.thread_id for summary in merged] == ["saved", "upload-only"]
    assert merged[0].title == "First question"
    assert merged[0].updated_at == checkpoint.updated_at
    assert merged[0].document_id == DOCUMENT_ID
    assert merged[1].title == "Uploaded paper.pdf"


def test_showcase_requires_existing_paper_conversation_and_survives_reopen(tmp_path) -> None:
    store = ConversationStore(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        store.publish_showcase("missing", "user", "Demo", "A sample", "gpt-oss:20b")
    store.save("thread", ConversationUpdate(user_id="user"))
    with pytest.raises(ValueError, match="requires a paper"):
        store.publish_showcase("thread", "user", "Demo", "A sample", "gpt-oss:20b")
    store.save("thread", ConversationUpdate(user_id="user", document_id=DOCUMENT_ID))
    published = store.publish_showcase(
        "thread", "user", "Demo", "A real sample conversation", "ollama/gpt-oss:20b"
    )
    assert ConversationStore(tmp_path).get_showcase() == published
    assert published.document_id == DOCUMENT_ID
    assert list(tmp_path.glob("*.tmp")) == []
