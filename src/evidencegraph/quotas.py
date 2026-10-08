"""Persistent, atomic fixed-window request counters for expensive API operations."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path


class QuotaStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.database_path = root / "request-quotas.sqlite3"

    def _connect(self) -> sqlite3.Connection:
        self.root.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS quota_counters ("
            "category TEXT NOT NULL, scope_hash TEXT NOT NULL, window_start INTEGER NOT NULL, "
            "expires_at INTEGER NOT NULL, request_count INTEGER NOT NULL, "
            "PRIMARY KEY(category, scope_hash, window_start))"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS quota_expiry ON quota_counters(expires_at)"
        )
        return connection

    def consume(
        self,
        category: str,
        scopes: Sequence[tuple[str, int]],
        period_seconds: int,
        *,
        now: int | None = None,
    ) -> int | None:
        """Return seconds until retry when any scope is full; otherwise count once."""
        if not scopes or period_seconds < 1 or any(limit < 1 for _, limit in scopes):
            raise ValueError("Quota scopes and period must be positive")
        current = int(time.time()) if now is None else now
        window_start = current - current % period_seconds
        expires_at = window_start + period_seconds
        hashed_scopes = [
            (hashlib.sha256(scope.encode()).hexdigest(), limit) for scope, limit in scopes
        ]
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM quota_counters WHERE expires_at <= ?", (current,))
            for scope_hash, limit in hashed_scopes:
                row = connection.execute(
                    "SELECT request_count FROM quota_counters "
                    "WHERE category = ? AND scope_hash = ? AND window_start = ?",
                    (category, scope_hash, window_start),
                ).fetchone()
                if row is not None and row[0] >= limit:
                    return expires_at - current
            for scope_hash, _limit in hashed_scopes:
                connection.execute(
                    "INSERT INTO quota_counters VALUES (?, ?, ?, ?, 1) "
                    "ON CONFLICT(category, scope_hash, window_start) "
                    "DO UPDATE SET request_count = request_count + 1",
                    (category, scope_hash, window_start, expires_at),
                )
        return None
