"""Persistent quota counters remain atomic across callers and store instances."""

from concurrent.futures import ThreadPoolExecutor

from evidencegraph.quotas import QuotaStore


def test_quota_restarts_and_resets_at_window_boundary(tmp_path) -> None:
    store = QuotaStore(tmp_path)
    scopes = [("ip:192.0.2.10", 2), ("account:owner", 3)]
    assert store.consume("chat", scopes, 60, now=120) is None
    assert QuotaStore(tmp_path).consume("chat", scopes, 60, now=121) is None
    assert store.consume("chat", scopes, 60, now=122) == 58
    assert store.consume("chat", scopes, 60, now=180) is None


def test_rejected_scope_does_not_charge_other_scope(tmp_path) -> None:
    store = QuotaStore(tmp_path)
    assert store.consume("upload", [("account:a", 1)], 60, now=120) is None
    assert store.consume(
        "upload", [("ip:new", 1), ("account:a", 1)], 60, now=121
    ) == 59
    assert store.consume("upload", [("ip:new", 1)], 60, now=122) is None


def test_concurrent_requests_cannot_exceed_limit(tmp_path) -> None:
    store = QuotaStore(tmp_path)

    def attempt(_: int) -> int | None:
        return store.consume("graph", [("ip:192.0.2.20", 3)], 60, now=120)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(attempt, range(16)))
    assert results.count(None) == 3
    assert results.count(60) == 13
