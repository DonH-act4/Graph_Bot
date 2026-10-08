"""Small persistent workspace index alongside LangGraph's message checkpoints."""

import os
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from schema import ConversationState, ConversationUpdate, Showcase, ThreadSummary


class ConversationStore:
    """Persist paper selection and drafts without duplicating conversation messages."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.database_path = root / "conversations.sqlite3"
        self.showcase_path = root / "showcase.json"

    def get(self, thread_id: str) -> ConversationState | None:
        if not self.database_path.exists():
            return None
        with closing(sqlite3.connect(self.database_path)) as connection:
            row = connection.execute(
                "SELECT state FROM conversations WHERE thread_id = ?", (thread_id,)
            ).fetchone()
        return ConversationState.model_validate_json(row[0]) if row else None

    def list(self, user_id: str, agent_id: str, limit: int) -> list[ConversationState]:
        if not self.database_path.exists():
            return []
        with closing(sqlite3.connect(self.database_path)) as connection:
            rows = connection.execute(
                "SELECT state FROM conversations WHERE user_id = ? AND agent_id = ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (user_id, agent_id, limit),
            ).fetchall()
        return [ConversationState.model_validate_json(row[0]) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        self.root.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS conversations ("
            "thread_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, agent_id TEXT NOT NULL, "
            "updated_at TEXT NOT NULL, state TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS conversations_user_agent_updated "
            "ON conversations(user_id, agent_id, updated_at DESC)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS deleted_threads ("
            "thread_id TEXT PRIMARY KEY, deleted_at TEXT NOT NULL)"
        )
        return connection

    def is_deleted(self, thread_id: str) -> bool:
        if not self.database_path.exists():
            return False
        with closing(self._connect()) as connection:
            return connection.execute(
                "SELECT 1 FROM deleted_threads WHERE thread_id = ?", (thread_id,)
            ).fetchone() is not None

    def deleted_thread_ids(self, thread_ids: Sequence[str]) -> set[str]:
        if not thread_ids or not self.database_path.exists():
            return set()
        placeholders = ", ".join("?" for _ in thread_ids)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"SELECT thread_id FROM deleted_threads WHERE thread_id IN ({placeholders})",
                thread_ids,
            ).fetchall()
        return {row[0] for row in rows}

    def other_threads_using_document(self, document_id: str, thread_id: str) -> tuple[str, ...]:
        """Conservatively retain a PDF if another workspace references its hash."""
        if not self.database_path.exists():
            return ()
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT thread_id, state FROM conversations WHERE thread_id != ?", (thread_id,)
            ).fetchall()
        return tuple(
            other_id
            for other_id, state in rows
            if ConversationState.model_validate_json(state).document_id == document_id
        )

    def delete(self, thread_id: str, user_id: str, agent_id: str) -> None:
        """Remove workspace data and prevent a delayed save from recreating this thread."""
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT user_id, agent_id FROM conversations WHERE thread_id = ?", (thread_id,)
            ).fetchone()
            if row is not None and row != (user_id, agent_id):
                raise PermissionError("Conversation not found")
            connection.execute(
                "INSERT OR IGNORE INTO deleted_threads(thread_id, deleted_at) VALUES (?, ?)",
                (thread_id, datetime.now(UTC).isoformat()),
            )
            connection.execute(
                "DELETE FROM conversations WHERE thread_id = ? AND user_id = ? AND agent_id = ?",
                (thread_id, user_id, agent_id),
            )

    def save(self, thread_id: str, update: ConversationUpdate) -> ConversationState:
        """Create or replace workspace state, keeping creation time and inferred title."""
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM deleted_threads WHERE thread_id = ?", (thread_id,)
            ).fetchone():
                raise PermissionError("Conversation not found")
            row = connection.execute(
                "SELECT state FROM conversations WHERE thread_id = ?", (thread_id,)
            ).fetchone()
            existing = ConversationState.model_validate_json(row[0]) if row else None
            if existing and (
                existing.user_id != update.user_id or existing.agent_id != update.agent_id
            ):
                raise PermissionError("Conversation not found")
            now = datetime.now(UTC)
            values = update.model_dump()
            if update.title is None and existing:
                values["title"] = existing.title
            state = ConversationState(
                **values,
                thread_id=thread_id,
                created_at=existing.created_at if existing else now,
                updated_at=now,
            )
            connection.execute(
                "INSERT INTO conversations(thread_id, user_id, agent_id, updated_at, state) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(thread_id) DO UPDATE SET "
                "updated_at=excluded.updated_at, state=excluded.state",
                (
                    thread_id,
                    state.user_id,
                    state.agent_id,
                    state.updated_at.isoformat(),
                    state.model_dump_json(),
                ),
            )
        return state

    def record_turn(
        self,
        thread_id: str,
        user_id: str,
        agent_id: str,
        message: str,
        document_id: str | None,
    ) -> ConversationState:
        """Associate a real chat turn with its paper and consume the saved draft."""
        existing = self.get(thread_id)
        values = existing.model_dump() if existing else {}
        current_document = document_id or (existing.document_id if existing else None)
        return self.save(
            thread_id,
            ConversationUpdate(
                user_id=user_id,
                agent_id=agent_id,
                title=(existing.title if existing else None) or message[:120],
                document_id=current_document,
                document_name=values.get("document_name")
                if current_document == values.get("document_id")
                else None,
                selected_block_ids=[],
                draft_message="",
            ),
        )

    def get_showcase(self) -> Showcase | None:
        if not self.showcase_path.exists():
            return None
        return Showcase.model_validate_json(self.showcase_path.read_text(encoding="utf-8"))

    def publish_showcase(
        self,
        thread_id: str,
        user_id: str,
        title: str,
        description: str,
        model: str,
    ) -> Showcase:
        """Publish an existing paper conversation; the caller creates real chat turns first."""
        conversation = self.get(thread_id)
        if conversation is None or conversation.user_id != user_id:
            raise ValueError("Conversation not found")
        if conversation.document_id is None:
            raise ValueError("Showcase conversation requires a paper")
        showcase = Showcase(
            title=title,
            description=description,
            thread_id=thread_id,
            user_id=user_id,
            document_id=conversation.document_id,
            model=model,
        )
        self.root.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.root, suffix=".tmp", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(showcase.model_dump_json(indent=2))
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, self.showcase_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return showcase


def merge_thread_summaries(
    checkpoints: list[ThreadSummary], workspaces: list[ConversationState], limit: int
) -> list[ThreadSummary]:
    """Include upload-only conversations and enrich legacy checkpoint summaries."""
    summaries = {summary.thread_id: summary for summary in checkpoints}
    for workspace in workspaces:
        previous = summaries.get(workspace.thread_id)
        updated_at = workspace.updated_at
        if previous and previous.updated_at and previous.updated_at > updated_at:
            updated_at = previous.updated_at
        summaries[workspace.thread_id] = ThreadSummary(
            thread_id=workspace.thread_id,
            agent_id=workspace.agent_id,
            title=workspace.title
            or (previous.title if previous else None)
            or workspace.document_name
            or "New research conversation",
            updated_at=updated_at,
            document_id=workspace.document_id,
            document_name=workspace.document_name,
            selected_block_ids=workspace.selected_block_ids,
        )
    return sorted(
        summaries.values(),
        key=lambda summary: summary.updated_at or datetime.min.replace(tzinfo=UTC),
        reverse=True,
    )[:limit]
