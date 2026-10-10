"""Owner-only operations summary and reversible account moderation."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from core import settings
from evidencegraph.accounts import AccountStore
from evidencegraph.ops_summary import (
    account_counts,
    current_account_request_counts,
    current_global_request_counts,
)
from service.auth import get_account_store, get_optional_account

router = APIRouter(prefix="/admin", tags=["admin"])


class AdminSummary(BaseModel):
    accounts_total: int
    verified_email_accounts: int
    uploads_allowed_today: int
    graphs_allowed_today: int
    chats_allowed_this_hour: int
    verification_emails_allowed_today: int
    quotas_enabled: bool


class ManagedAccountResponse(BaseModel):
    account_id: str
    username: str
    created_at: str
    email_verified: bool
    banned: bool
    is_self: bool
    uploads_allowed_today: int
    graphs_allowed_today: int
    chats_allowed_this_hour: int


class ManagedAccountsResponse(BaseModel):
    accounts: list[ManagedAccountResponse]


class BanRequest(BaseModel):
    reason: str = Field(default="", max_length=200)


def require_admin(
    account_id: Annotated[str | None, Depends(get_optional_account)],
) -> str:
    if account_id is None:
        raise HTTPException(status_code=401, detail="Sign in to view operations")
    if settings.EVIDENCEGRAPH_ADMIN_ACCOUNT_ID is None or account_id != str(
        settings.EVIDENCEGRAPH_ADMIN_ACCOUNT_ID
    ):
        raise HTTPException(status_code=403, detail="Administrator access required")
    return account_id


@router.get("/summary")
async def summary(
    _admin_id: Annotated[str, Depends(require_admin)],
) -> AdminSummary:
    accounts_total, verified_email_accounts = await run_in_threadpool(
        account_counts, settings.EVIDENCEGRAPH_DATA_DIR
    )
    counts = await run_in_threadpool(
        current_global_request_counts, settings.EVIDENCEGRAPH_DATA_DIR
    )
    return AdminSummary(
        accounts_total=accounts_total,
        verified_email_accounts=verified_email_accounts,
        uploads_allowed_today=counts["upload"],
        graphs_allowed_today=counts["graph"],
        chats_allowed_this_hour=counts["chat"],
        verification_emails_allowed_today=counts["verification-email"],
        quotas_enabled=settings.EVIDENCEGRAPH_QUOTAS_ENABLED,
    )


@router.get("/accounts")
async def managed_accounts(
    _admin_id: Annotated[str, Depends(require_admin)],
    accounts: Annotated[AccountStore, Depends(get_account_store)],
) -> ManagedAccountsResponse:
    listed = await run_in_threadpool(accounts.list_managed_accounts)
    counts = await run_in_threadpool(
        current_account_request_counts,
        settings.EVIDENCEGRAPH_DATA_DIR,
        [item.account_id for item in listed],
    )
    return ManagedAccountsResponse(
        accounts=[
            ManagedAccountResponse(
                account_id=item.account_id,
                username=item.username,
                created_at=item.created_at,
                email_verified=item.email_verified,
                banned=item.banned,
                is_self=item.account_id == _admin_id,
                uploads_allowed_today=counts[item.account_id]["upload"],
                graphs_allowed_today=counts[item.account_id]["graph"],
                chats_allowed_this_hour=counts[item.account_id]["chat"],
            )
            for item in listed
        ]
    )


@router.put("/accounts/{target_id}/ban", status_code=204)
async def ban_account(
    target_id: UUID,
    details: BanRequest,
    admin_id: Annotated[str, Depends(require_admin)],
    accounts: Annotated[AccountStore, Depends(get_account_store)],
) -> None:
    if str(target_id) == admin_id:
        raise HTTPException(status_code=409, detail="Cannot ban the administrator account")
    found = await run_in_threadpool(
        accounts.ban_account, str(target_id), admin_id, details.reason.strip()
    )
    if not found:
        raise HTTPException(status_code=404, detail="Account not found")


@router.delete("/accounts/{target_id}/ban", status_code=204)
async def unban_account(
    target_id: UUID,
    admin_id: Annotated[str, Depends(require_admin)],
    accounts: Annotated[AccountStore, Depends(get_account_store)],
) -> None:
    found = await run_in_threadpool(accounts.unban_account, str(target_id), admin_id)
    if not found:
        raise HTTPException(status_code=404, detail="Account not found")
