"""Guest sessions must not turn a public paper hash into an access credential."""

import hashlib
import sqlite3
from datetime import UTC, datetime, timedelta

from evidencegraph.access import PaperAccessStore

PAPER_ID = "a" * 64


def test_guest_token_is_opaque_and_only_its_hash_is_stored(tmp_path):
    access = PaperAccessStore(tmp_path)
    guest_id, token = access.create_guest()

    assert access.resolve_guest(token) == guest_id
    assert access.resolve_guest("wrong-token") is None
    with sqlite3.connect(access.database_path) as connection:
        stored = connection.execute("SELECT token_hash FROM guest_sessions").fetchone()
    assert stored == (hashlib.sha256(token.encode()).hexdigest(),)
    assert token not in access.database_path.read_bytes().decode(errors="ignore")


def test_paper_access_is_granted_to_the_uploading_guest_only(tmp_path):
    access = PaperAccessStore(tmp_path)
    first, _token = access.create_guest()
    second, _other_token = access.create_guest()

    access.grant(first, PAPER_ID)
    assert access.can_read(first, PAPER_ID)
    assert not access.can_read(second, PAPER_ID)
    assert access.has_other_owner(second, PAPER_ID)
    access.revoke(first, PAPER_ID)
    assert not access.can_read(first, PAPER_ID)


def test_sign_in_moves_existing_paper_grants_without_leaving_guest_access(tmp_path):
    access = PaperAccessStore(tmp_path)
    guest_id, _token = access.create_guest()
    account_id = "registered-account"
    access.grant(guest_id, PAPER_ID)

    access.transfer_grants(guest_id, account_id)
    access.transfer_grants(guest_id, account_id)

    assert access.owns(account_id, PAPER_ID)
    assert not access.owns(guest_id, PAPER_ID)
    assert not access.has_other_owner(account_id, PAPER_ID)


def test_expired_guest_token_is_rejected(tmp_path):
    access = PaperAccessStore(tmp_path)
    _guest_id, token = access.create_guest()
    expired = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    with sqlite3.connect(access.database_path) as connection:
        connection.execute("UPDATE guest_sessions SET expires_at = ?", (expired,))
    assert access.resolve_guest(token) is None
