"""Bans revoke old sessions and remain reversible without deleting content."""

import sqlite3

import pytest

from evidencegraph.accounts import AccountStore


def test_ban_revokes_sessions_and_unban_requires_new_login(tmp_path) -> None:
    store = AccountStore(tmp_path)
    owner_id = store.register("don", "long-owner-test-password")
    visitor_id = store.register("visitor", "long-visitor-test-password")
    assert owner_id is not None and visitor_id is not None
    token = store.issue_session(visitor_id)
    assert store.resolve_session(token) == visitor_id

    with pytest.raises(ValueError, match="own account"):
        store.ban_account(owner_id, owner_id)
    assert store.ban_account(visitor_id, owner_id, "Abuse report") is True
    assert store.ban_account(visitor_id, owner_id, "Duplicate") is True
    assert store.resolve_session(token) is None
    assert store.authenticate("visitor", "long-visitor-test-password") is None
    assert next(item for item in store.list_managed_accounts() if item.account_id == visitor_id).banned

    assert store.unban_account(visitor_id, owner_id) is True
    assert store.resolve_session(token) is None
    assert store.authenticate("visitor", "long-visitor-test-password") == visitor_id
    assert store.issue_session(visitor_id) != token
    with sqlite3.connect(store.database_path) as connection:
        actions = connection.execute(
            "SELECT action FROM account_moderation_events WHERE target_id = ? ORDER BY event_id",
            (visitor_id,),
        ).fetchall()
    assert actions == [("ban",), ("unban",)]


def test_banning_missing_account_does_not_create_a_record(tmp_path) -> None:
    store = AccountStore(tmp_path)
    assert store.ban_account("missing", "owner") is False
    assert store.unban_account("missing", "owner") is False
    assert store.list_managed_accounts() == []
