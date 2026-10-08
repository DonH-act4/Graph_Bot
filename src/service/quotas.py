"""Map costly API requests to persistent IP and account request budgets."""

from __future__ import annotations

import ipaddress

from fastapi import HTTPException, Request

from core import settings
from evidencegraph.quotas import QuotaStore

_quota_store = QuotaStore(settings.EVIDENCEGRAPH_DATA_DIR)
# Provisional local-demo ceilings; measure worker capacity before public rollout.
POLICIES: dict[str, tuple[int, int, int | None]] = {
    "register": (3600, 5, None),
    "verify": (900, 10, None),
    "login": (900, 20, None),
    "upload": (86400, 10, 20),
    "graph": (86400, 10, 20),
    "chat": (3600, 60, 120),
}


def get_quota_store() -> QuotaStore:
    return _quota_store


def client_ip(request: Request) -> str:
    """Trust X-Real-IP only when the connecting peer is an approved proxy."""
    peer = request.client.host if request.client is not None else "unknown"
    try:
        peer_address = ipaddress.ip_address(peer)
    except ValueError:
        return peer
    for value in settings.EVIDENCEGRAPH_TRUSTED_PROXY_CIDRS.split(","):
        if not value.strip():
            continue
        if peer_address in ipaddress.ip_network(value.strip(), strict=False):
            forwarded = request.headers.get("x-real-ip", "")
            try:
                return str(ipaddress.ip_address(forwarded))
            except ValueError:
                return peer
    return peer


def enforce_quota(
    request: Request,
    store: QuotaStore,
    category: str,
    account_id: str | None = None,
) -> None:
    if not settings.EVIDENCEGRAPH_QUOTAS_ENABLED:
        return
    period, ip_limit, account_limit = POLICIES[category]
    scopes = [(f"ip:{client_ip(request)}", ip_limit)]
    if account_id is not None and account_limit is not None:
        scopes.append((f"account:{account_id}", account_limit))
    global_limits = {
        "upload": settings.EVIDENCEGRAPH_GLOBAL_UPLOADS_PER_DAY,
        "graph": settings.EVIDENCEGRAPH_GLOBAL_GRAPHS_PER_DAY,
        "chat": settings.EVIDENCEGRAPH_GLOBAL_CHATS_PER_HOUR,
    }
    if category in global_limits:
        scopes.append(("site:all", global_limits[category]))
    retry_after = store.consume(category, scopes, period)
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail=f"Too many {category} requests; try again after {retry_after} seconds",
            headers={"Retry-After": str(retry_after)},
        )


def enforce_verification_email_budget(store: QuotaStore) -> None:
    """Bound total provider calls even when registrations originate from many IPs."""
    if not settings.EVIDENCEGRAPH_QUOTAS_ENABLED:
        return
    retry_after = store.consume(
        "verification-email",
        [("site:all", settings.EVIDENCEGRAPH_GLOBAL_VERIFICATION_EMAILS_PER_DAY)],
        86400,
    )
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail=f"Verification email limit reached; try again after {retry_after} seconds",
            headers={"Retry-After": str(retry_after)},
        )
