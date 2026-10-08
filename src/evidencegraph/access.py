"""Opaque guest sessions and per-paper access for the local research workspace."""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from evidencegraph.conversations import ConversationStore

GUEST_COOKIE = "evidencegraph_guest"
GUEST_LIFETIME = timedelta(hours=24)


class PaperAccessStore:
    """Store only token hashes; paper hashes are identifiers, not access grants."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.database_path = root / "paper-access.sqlite3"

    def _connect(self) -> sqlite3.Connection:
        self.root.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS guest_sessions ("
            "token_hash TEXT PRIMARY KEY, guest_id TEXT NOT NULL, expires_at TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS paper_access ("
            "guest_id TEXT NOT NULL, document_id TEXT NOT NULL, created_at TEXT NOT NULL, "
            "PRIMARY KEY(guest_id, document_id))"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS paper_access_document ON paper_access(document_id)"
        )
        return connection

    def resolve_guest(self, token: str) -> str | None:
        if not token or not self.database_path.exists():
            return None
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT guest_id, expires_at FROM guest_sessions WHERE token_hash = ?",
                (token_hash,),
            ).fetchone()
        if row is None or datetime.fromisoformat(row[1]) <= datetime.now(UTC):
            return None
        return str(row[0])

    def create_guest(self) -> tuple[str, str]:
        guest_id = str(uuid4())
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        expires_at = datetime.now(UTC) + GUEST_LIFETIME
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT INTO guest_sessions(token_hash, guest_id, expires_at) VALUES (?, ?, ?)",
                (token_hash, guest_id, expires_at.isoformat()),
            )
        return guest_id, token

    def grant(self, guest_id: str, document_id: str) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT OR IGNORE INTO paper_access(guest_id, document_id, created_at) "
                "VALUES (?, ?, ?)",
                (guest_id, document_id, datetime.now(UTC).isoformat()),
            )

    def transfer_grants(self, guest_id: str, account_id: str) -> None:
        """Atomically move this browser's PDF access to its signed-in account."""
        if guest_id == account_id or not self.database_path.exists():
            return
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT OR IGNORE INTO paper_access(guest_id, document_id, created_at) "
                "SELECT ?, document_id, created_at FROM paper_access WHERE guest_id = ?",
                (account_id, guest_id),
            )
            connection.execute("DELETE FROM paper_access WHERE guest_id = ?", (guest_id,))

    def owns(self, guest_id: str, document_id: str) -> bool:
        if not self.database_path.exists():
            return False
        with closing(self._connect()) as connection:
            return connection.execute(
                "SELECT 1 FROM paper_access WHERE guest_id = ? AND document_id = ?",
                (guest_id, document_id),
            ).fetchone() is not None

    def has_other_owner(self, guest_id: str, document_id: str) -> bool:
        if not self.database_path.exists():
            return False
        with closing(self._connect()) as connection:
            return connection.execute(
                "SELECT 1 FROM paper_access WHERE guest_id != ? AND document_id = ? LIMIT 1",
                (guest_id, document_id),
            ).fetchone() is not None

    def revoke(self, guest_id: str, document_id: str) -> None:
        if not self.database_path.exists():
            return
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "DELETE FROM paper_access WHERE guest_id = ? AND document_id = ?",
                (guest_id, document_id),
            )

    def is_public_tutorial(self, document_id: str) -> bool:
        showcase = ConversationStore(self.root).get_showcase()
        return showcase is not None and showcase.document_id == document_id

    def can_read(self, guest_id: str, document_id: str) -> bool:
        return self.owns(guest_id, document_id) or self.is_public_tutorial(document_id)
