"""Read aggregate operations counts without modifying live databases."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from contextlib import closing
from pathlib import Path


def _read_only_connection(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=5)


def account_counts(root: Path) -> tuple[int, int]:
    """Return total accounts and accounts with a verified email."""
    path = root / "accounts.sqlite3"
    if not path.is_file():
        return 0, 0
    with closing(_read_only_connection(path)) as connection:
        total = connection.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]
        verified = connection.execute("SELECT COUNT(*) FROM account_emails").fetchone()[0]
    return int(total), int(verified)


def current_global_request_counts(root: Path, *, now: int | None = None) -> dict[str, int]:
    """Count accepted site-wide quota requests in the current UTC windows."""
    categories = {
        "upload": 86400,
        "graph": 86400,
        "chat": 3600,
        "verification-email": 86400,
    }
    counts = dict.fromkeys(categories, 0)
    path = root / "request-quotas.sqlite3"
    if not path.is_file():
        return counts
    current = int(time.time()) if now is None else now
    site_hash = hashlib.sha256(b"site:all").hexdigest()
    with closing(_read_only_connection(path)) as connection:
        for category, period in categories.items():
            window_start = current - current % period
            row = connection.execute(
                "SELECT request_count FROM quota_counters "
                "WHERE category = ? AND scope_hash = ? AND window_start = ? "
                "AND expires_at > ?",
                (category, site_hash, window_start, current),
            ).fetchone()
            if row is not None:
                counts[category] = int(row[0])
    return counts


def current_account_request_counts(
    root: Path, account_ids: list[str], *, now: int | None = None
) -> dict[str, dict[str, int]]:
    """Read current-window account quota counters; guests are not attributed."""
    categories = {"upload": 86400, "graph": 86400, "chat": 3600}
    counts = {account_id: dict.fromkeys(categories, 0) for account_id in account_ids}
    path = root / "request-quotas.sqlite3"
    if not path.is_file() or not account_ids:
        return counts
    current = int(time.time()) if now is None else now
    with closing(_read_only_connection(path)) as connection:
        for account_id in account_ids:
            scope_hash = hashlib.sha256(f"account:{account_id}".encode()).hexdigest()
            for category, period in categories.items():
                window_start = current - current % period
                row = connection.execute(
                    "SELECT request_count FROM quota_counters "
                    "WHERE category = ? AND scope_hash = ? AND window_start = ? "
                    "AND expires_at > ?",
                    (category, scope_hash, window_start, current),
                ).fetchone()
                if row is not None:
                    counts[account_id][category] = int(row[0])
    return counts
