"""Opt-in browser account endpoints; graph browsing still uses guest access."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, EmailStr, Field
from starlette.concurrency import run_in_threadpool

from core import settings
from evidencegraph.access import GUEST_COOKIE, PaperAccessStore
from evidencegraph.accounts import ACCOUNT_COOKIE, ACCOUNT_LIFETIME, AccountStore
from evidencegraph.email_delivery import (
    EmailDeliveryError,
    ResendVerificationSender,
    VerificationSender,
)
from evidencegraph.quotas import QuotaStore
from service.quotas import enforce_quota, enforce_verification_email_budget, get_quota_store

router = APIRouter(prefix="/auth", tags=["auth"])
_accounts = AccountStore(settings.EVIDENCEGRAPH_DATA_DIR)
_access_store = PaperAccessStore(settings.EVIDENCEGRAPH_DATA_DIR)


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=32, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=12, max_length=128)


class RegistrationCredentials(Credentials):
    email: EmailStr | None = None


class VerificationCode(BaseModel):
    username: str = Field(min_length=3, max_length=32, pattern=r"^[A-Za-z0-9_.-]+$")
    code: str = Field(min_length=1, max_length=64)


def get_account_store() -> AccountStore:
    return _accounts


def get_paper_access_store() -> PaperAccessStore:
    return _access_store


def get_verification_sender() -> VerificationSender | None:
    if settings.RESEND_API_KEY is None or settings.RESEND_FROM_EMAIL is None:
        return None
    return ResendVerificationSender(
        settings.RESEND_API_KEY.get_secret_value(), settings.RESEND_FROM_EMAIL
    )


def _transfer_guest_grants(request: Request, account_id: str, access: PaperAccessStore) -> None:
    guest_id = access.resolve_guest(request.cookies.get(GUEST_COOKIE, ""))
    if guest_id is not None:
        access.transfer_grants(guest_id, account_id)


def get_optional_account(
    request: Request,
    accounts: Annotated[AccountStore, Depends(get_account_store)],
) -> str | None:
    return accounts.resolve_session(request.cookies.get(ACCOUNT_COOKIE, ""))


def _set_account_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        ACCOUNT_COOKIE,
        token,
        max_age=int(ACCOUNT_LIFETIME.total_seconds()),
        httponly=True,
        secure=settings.EVIDENCEGRAPH_COOKIE_SECURE,
        samesite="lax",
        path="/",
    )


@router.post("/register", status_code=201)
async def register(
    credentials: RegistrationCredentials,
    request: Request,
    response: Response,
    accounts: Annotated[AccountStore, Depends(get_account_store)],
    access: Annotated[PaperAccessStore, Depends(get_paper_access_store)],
    quotas: Annotated[QuotaStore, Depends(get_quota_store)],
    sender: Annotated[VerificationSender | None, Depends(get_verification_sender)],
) -> dict[str, str]:
    await run_in_threadpool(enforce_quota, request, quotas, "register")
    if settings.EVIDENCEGRAPH_EMAIL_VERIFICATION_REQUIRED:
        if credentials.email is None:
            raise HTTPException(status_code=422, detail="Email is required for registration")
        if sender is None:
            raise HTTPException(status_code=503, detail="Email registration is not configured")
        code = await run_in_threadpool(
            accounts.begin_email_registration,
            credentials.username,
            str(credentials.email),
            credentials.password,
        )
        if code is None:
            raise HTTPException(status_code=409, detail="Username or email is unavailable")
        try:
            await run_in_threadpool(enforce_verification_email_budget, quotas)
        except HTTPException:
            await run_in_threadpool(accounts.discard_email_registration, credentials.username, code)
            raise
        try:
            await run_in_threadpool(sender.send_verification, str(credentials.email), code)
        except EmailDeliveryError as exc:
            await run_in_threadpool(accounts.discard_email_registration, credentials.username, code)
            raise HTTPException(
                status_code=503, detail="Verification email could not be sent; try again later"
            ) from exc
        response.status_code = 202
        return {"status": "verification_required"}
    account_id = await run_in_threadpool(accounts.register, credentials.username, credentials.password)
    if account_id is None:
        raise HTTPException(status_code=409, detail="Username is already taken")
    await run_in_threadpool(_transfer_guest_grants, request, account_id, access)
    token = await run_in_threadpool(accounts.issue_session, account_id)
    _set_account_cookie(response, token)
    return {"user_id": account_id}


@router.post("/verify-email")
async def verify_email(
    verification: VerificationCode,
    request: Request,
    response: Response,
    accounts: Annotated[AccountStore, Depends(get_account_store)],
    access: Annotated[PaperAccessStore, Depends(get_paper_access_store)],
    quotas: Annotated[QuotaStore, Depends(get_quota_store)],
) -> dict[str, str]:
    await run_in_threadpool(enforce_quota, request, quotas, "verify")
    account_id = await run_in_threadpool(
        accounts.complete_email_registration, verification.username, verification.code
    )
    if account_id is None:
        raise HTTPException(status_code=400, detail="Verification code is invalid or expired")
    await run_in_threadpool(_transfer_guest_grants, request, account_id, access)
    token = await run_in_threadpool(accounts.issue_session, account_id)
    _set_account_cookie(response, token)
    return {"user_id": account_id}


@router.post("/login")
async def login(
    credentials: Credentials,
    request: Request,
    response: Response,
    accounts: Annotated[AccountStore, Depends(get_account_store)],
    access: Annotated[PaperAccessStore, Depends(get_paper_access_store)],
    quotas: Annotated[QuotaStore, Depends(get_quota_store)],
) -> dict[str, str]:
    await run_in_threadpool(enforce_quota, request, quotas, "login")
    account_id = await run_in_threadpool(accounts.authenticate, credentials.username, credentials.password)
    if account_id is None:
        raise HTTPException(status_code=401, detail="Invalid credentials or too many attempts; try later")
    await run_in_threadpool(_transfer_guest_grants, request, account_id, access)
    token = await run_in_threadpool(accounts.issue_session, account_id)
    _set_account_cookie(response, token)
    return {"user_id": account_id}


@router.post("/logout", status_code=204)
async def logout(
    request: Request,
    response: Response,
    accounts: Annotated[AccountStore, Depends(get_account_store)],
) -> None:
    await run_in_threadpool(accounts.revoke_session, request.cookies.get(ACCOUNT_COOKIE, ""))
    response.delete_cookie(ACCOUNT_COOKIE, path="/")
