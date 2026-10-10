"""Local account credentials and revocable, opaque login sessions."""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from secrets import compare_digest
from uuid import uuid4

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError, VerifyMismatchError

ACCOUNT_COOKIE = "evidencegraph_account"
ACCOUNT_LIFETIME = timedelta(days=7)
EMAIL_VERIFICATION_LIFETIME = timedelta(minutes=20)
MAX_EMAIL_VERIFICATION_ATTEMPTS = 5
LOGIN_WINDOW = timedelta(minutes=15)
MAX_LOGIN_FAILURES = 5
_hasher = PasswordHasher(time_cost=2, memory_cost=19456, parallelism=1)
# Verify an unknown username against a real hash to avoid a cheap account-existence timing signal.
_dummy_hash = _hasher.hash("evidencegraph-invalid-account-password")


@dataclass(frozen=True)
class ManagedAccount:
    account_id: str
    username: str
    created_at: str
    email_verified: bool
    banned: bool


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
        connection.execute(
            "CREATE TABLE IF NOT EXISTS account_emails ("
            "account_id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE, verified_at TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS pending_registrations ("
            "username TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL, "
            "code_hash TEXT NOT NULL, expires_at TEXT NOT NULL, attempts INTEGER NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS account_bans ("
            "account_id TEXT PRIMARY KEY, banned_at TEXT NOT NULL, "
            "FOREIGN KEY(account_id) REFERENCES accounts(account_id))"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS account_moderation_events ("
            "event_id INTEGER PRIMARY KEY AUTOINCREMENT, actor_id TEXT NOT NULL, "
            "target_id TEXT NOT NULL, action TEXT NOT NULL, reason TEXT NOT NULL, "
            "created_at TEXT NOT NULL)"
        )
        return connection

    def list_managed_accounts(self, *, limit: int = 100) -> list[ManagedAccount]:
        """Return a bounded owner view without email addresses or password hashes."""
        if not 1 <= limit <= 100:
            raise ValueError("Account list limit must be between 1 and 100")
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT a.account_id, a.username, a.created_at, "
                "EXISTS(SELECT 1 FROM account_emails e WHERE e.account_id = a.account_id), "
                "EXISTS(SELECT 1 FROM account_bans b WHERE b.account_id = a.account_id) "
                "FROM accounts a ORDER BY a.created_at DESC, a.account_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            ManagedAccount(str(row[0]), str(row[1]), str(row[2]), bool(row[3]), bool(row[4]))
            for row in rows
        ]

    def ban_account(self, target_id: str, actor_id: str, reason: str = "") -> bool:
        """Suspend an existing account and revoke every active session atomically."""
        if target_id == actor_id:
            raise ValueError("An administrator cannot ban their own account")
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM accounts WHERE account_id = ?", (target_id,)
            ).fetchone() is None:
                return False
            now = datetime.now(UTC).isoformat()
            changed = connection.execute(
                "INSERT OR IGNORE INTO account_bans(account_id, banned_at) VALUES (?, ?)",
                (target_id, now),
            ).rowcount
            connection.execute("DELETE FROM account_sessions WHERE account_id = ?", (target_id,))
            if changed:
                connection.execute(
                    "INSERT INTO account_moderation_events"
                    "(actor_id, target_id, action, reason, created_at) VALUES (?, ?, ?, ?, ?)",
                    (actor_id, target_id, "ban", reason, now),
                )
        return True

    def unban_account(self, target_id: str, actor_id: str) -> bool:
        """Restore login eligibility without restoring revoked sessions."""
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM accounts WHERE account_id = ?", (target_id,)
            ).fetchone() is None:
                return False
            changed = connection.execute(
                "DELETE FROM account_bans WHERE account_id = ?", (target_id,)
            ).rowcount
            if changed:
                connection.execute(
                    "INSERT INTO account_moderation_events"
                    "(actor_id, target_id, action, reason, created_at) VALUES (?, ?, ?, ?, ?)",
                    (actor_id, target_id, "unban", "", datetime.now(UTC).isoformat()),
                )
        return True

    def begin_email_registration(self, username: str, email: str, password: str) -> str | None:
        """Reserve a new identity and return a high-entropy, short-lived email code."""
        normalized = username.casefold()
        password_hash = _hasher.hash(password)
        code = secrets.token_urlsafe(18)
        now = datetime.now(UTC)
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM pending_registrations WHERE expires_at <= ?", (now.isoformat(),)
            )
            if connection.execute(
                "SELECT 1 FROM accounts WHERE username = ?", (normalized,)
            ).fetchone() or connection.execute(
                "SELECT 1 FROM account_emails WHERE email = ?", (email,)
            ).fetchone():
                return None
            # A retry with the same username/email replaces its previous, now invalid code.
            connection.execute(
                "DELETE FROM pending_registrations WHERE username = ? AND email = ?",
                (normalized, email),
            )
            try:
                connection.execute(
                    "INSERT INTO pending_registrations VALUES (?, ?, ?, ?, ?, 0)",
                    (
                        normalized,
                        email,
                        password_hash,
                        hashlib.sha256(code.encode()).hexdigest(),
                        (now + EMAIL_VERIFICATION_LIFETIME).isoformat(),
                    ),
                )
            except sqlite3.IntegrityError:
                return None
        return code

    def discard_email_registration(self, username: str, code: str) -> None:
        """Release only the pending code whose delivery failed."""
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "DELETE FROM pending_registrations WHERE username = ? AND code_hash = ?",
                (username.casefold(), hashlib.sha256(code.encode()).hexdigest()),
            )

    def complete_email_registration(self, username: str, code: str) -> str | None:
        """Verify once, then create an account and bind its verified email atomically."""
        normalized = username.casefold()
        now = datetime.now(UTC)
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT email, password_hash, code_hash, expires_at, attempts "
                    "FROM pending_registrations WHERE username = ?",
                    (normalized,),
                ).fetchone()
                if row is None:
                    return None
                email, password_hash, code_hash, expires_at, attempts = row
                if expires_at <= now.isoformat() or attempts >= MAX_EMAIL_VERIFICATION_ATTEMPTS:
                    connection.execute(
                        "DELETE FROM pending_registrations WHERE username = ?", (normalized,)
                    )
                    return None
                if not compare_digest(code_hash, hashlib.sha256(code.encode()).hexdigest()):
                    connection.execute(
                        "UPDATE pending_registrations SET attempts = attempts + 1 WHERE username = ?",
                        (normalized,),
                    )
                    return None
                account_id = str(uuid4())
                connection.execute(
                    "INSERT INTO accounts(account_id, username, password_hash, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (account_id, normalized, password_hash, now.isoformat()),
                )
                connection.execute(
                    "INSERT INTO account_emails(account_id, email, verified_at) VALUES (?, ?, ?)",
                    (account_id, email, now.isoformat()),
                )
                connection.execute(
                    "DELETE FROM pending_registrations WHERE username = ?", (normalized,)
                )
                return account_id
        except sqlite3.IntegrityError:
            return None

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
                "SELECT a.account_id, a.password_hash, "
                "EXISTS(SELECT 1 FROM account_bans b WHERE b.account_id = a.account_id) "
                "FROM accounts a WHERE a.username = ?",
                (normalized,),
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
            if locked or row is None or not valid or bool(row[2]):
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
                "SELECT s.account_id, s.expires_at FROM account_sessions s "
                "WHERE s.token_hash = ? AND NOT EXISTS "
                "(SELECT 1 FROM account_bans b WHERE b.account_id = s.account_id)",
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
