"""Owner statistics read aggregate SQLite state without altering it."""

from evidencegraph.ops_summary import (
    account_counts,
    current_account_request_counts,
    current_global_request_counts,
)
from evidencegraph.quotas import QuotaStore


def test_missing_databases_return_zero_without_creating_files(tmp_path) -> None:
    assert account_counts(tmp_path) == (0, 0)
    assert current_global_request_counts(tmp_path, now=1_700_000_000) == {
        "upload": 0,
        "graph": 0,
        "chat": 0,
        "verification-email": 0,
    }
    assert list(tmp_path.iterdir()) == []


def test_only_site_wide_current_window_counters_are_reported(tmp_path) -> None:
    now = 1_700_000_000
    store = QuotaStore(tmp_path)
    assert store.consume("upload", [("ip:one", 10), ("site:all", 30)], 86400, now=now) is None
    assert store.consume("upload", [("ip:two", 10), ("site:all", 30)], 86400, now=now) is None
    assert store.consume("chat", [("ip:one", 10), ("site:all", 240)], 3600, now=now) is None
    assert store.consume("verification-email", [("site:all", 50)], 86400, now=now) is None

    counts = current_global_request_counts(tmp_path, now=now)
    assert counts == {"upload": 2, "graph": 0, "chat": 1, "verification-email": 1}
    assert current_global_request_counts(tmp_path, now=now + 86400) == dict.fromkeys(counts, 0)


def test_account_counts_exclude_guest_and_other_account_scopes(tmp_path) -> None:
    now = 1_700_000_000
    store = QuotaStore(tmp_path)
    assert store.consume("upload", [("ip:guest", 10), ("site:all", 30)], 86400, now=now) is None
    assert store.consume("upload", [("account:owner", 20), ("site:all", 30)], 86400, now=now) is None
    assert store.consume("chat", [("account:reader", 120), ("site:all", 240)], 3600, now=now) is None
    assert current_account_request_counts(tmp_path, ["owner", "reader"], now=now) == {
        "owner": {"upload": 1, "graph": 0, "chat": 0},
        "reader": {"upload": 0, "graph": 0, "chat": 1},
    }
    assert current_account_request_counts(tmp_path, ["owner"], now=now + 86400) == {
        "owner": {"upload": 0, "graph": 0, "chat": 0}
    }
