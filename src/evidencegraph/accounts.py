"""Local account credentials and revocable, opaque login sessions."""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError, VerifyMismatchError

ACCOUNT_COOKIE = "evidencegraph_account"
ACCOUNT_LIFETIME = timedelta(days=7)
LOGIN_WINDOW = timedelta(minutes=15)
MAX_LOGIN_FAILURES = 5
_hasher = PasswordHasher(time_cost=2, memory_cost=19456, parallelism=1)
# Verify an unknown username against a real hash to avoid a cheap account-existence timing signal.
_dummy_hash = _hasher.hash("evidencegraph-invalid-account-password")


class AccountStore:
    """Persist account IDs separately from guest access grants and client-selected IDs."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.database_path = root / "accounts.sqlite3"

    def _connect(self) -> sqlite3.Connection:
        self.root.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS accounts ("
            "account_id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE, "
            "password_hash TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS account_sessions ("
            "token_hash TEXT PRIMARY KEY, account_id TEXT NOT NULL, expires_at TEXT NOT NULL, "
            "FOREIGN KEY(account_id) REFERENCES accounts(account_id))"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS login_failures ("
            "username TEXT PRIMARY KEY, failure_count INTEGER NOT NULL, first_failure_at TEXT NOT NULL)"
        )
        return connection

    def register(self, username: str, password: str) -> str | None:
        normalized = username.casefold()
        password_hash = _hasher.hash(password)
        account_id = str(uuid4())
        with closing(self._connect()) as connection, connection:
            try:
                connection.execute(
                    "INSERT INTO accounts(account_id, username, password_hash, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (account_id, normalized, password_hash, datetime.now(UTC).isoformat()),
                )
            except sqlite3.IntegrityError:
                return None
        return account_id

    def authenticate(self, username: str, password: str) -> str | None:
        normalized = username.casefold()
        now = datetime.now(UTC)
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT account_id, password_hash FROM accounts WHERE username = ?", (normalized,)
            ).fetchone()
            failures = connection.execute(
                "SELECT failure_count, first_failure_at FROM login_failures WHERE username = ?",
                (normalized,),
            ).fetchone()
            locked = bool(
                failures
                and failures[0] >= MAX_LOGIN_FAILURES
                and now - datetime.fromisoformat(failures[1]) < LOGIN_WINDOW
            )
            try:
                valid = _hasher.verify(row[1] if row else _dummy_hash, password)
            except (VerifyMismatchError, VerificationError):
                valid = False
            if locked or row is None or not valid:
                if not failures or now - datetime.fromisoformat(failures[1]) >= LOGIN_WINDOW:
                    connection.execute(
                        "INSERT OR REPLACE INTO login_failures VALUES (?, ?, ?)",
                        (normalized, 1, now.isoformat()),
                    )
                else:
                    connection.execute(
                        "UPDATE login_failures SET failure_count = failure_count + 1 "
                        "WHERE username = ?",
                        (normalized,),
                    )
                return None
            connection.execute("DELETE FROM login_failures WHERE username = ?", (normalized,))
            return str(row[0])

    def issue_session(self, account_id: str) -> str:
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now(UTC) + ACCOUNT_LIFETIME
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT INTO account_sessions(token_hash, account_id, expires_at) VALUES (?, ?, ?)",
                (hashlib.sha256(token.encode()).hexdigest(), account_id, expires_at.isoformat()),
            )
        return token

    def resolve_session(self, token: str) -> str | None:
        if not token or not self.database_path.exists():
            return None
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT account_id, expires_at FROM account_sessions WHERE token_hash = ?",
                (hashlib.sha256(token.encode()).hexdigest(),),
            ).fetchone()
        if row is None or datetime.fromisoformat(row[1]) <= datetime.now(UTC):
            return None
        return str(row[0])

    def revoke_session(self, token: str) -> None:
        if not token or not self.database_path.exists():
            return
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "DELETE FROM account_sessions WHERE token_hash = ?",
                (hashlib.sha256(token.encode()).hexdigest(),),
            )
