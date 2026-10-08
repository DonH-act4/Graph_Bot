"""Opt-in browser account endpoints; graph browsing still uses guest access."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from core import settings
from evidencegraph.accounts import ACCOUNT_COOKIE, ACCOUNT_LIFETIME, AccountStore

router = APIRouter(prefix="/auth", tags=["auth"])
_accounts = AccountStore(settings.EVIDENCEGRAPH_DATA_DIR)


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=32, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=12, max_length=128)


def get_account_store() -> AccountStore:
    return _accounts


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
    credentials: Credentials,
    response: Response,
    accounts: Annotated[AccountStore, Depends(get_account_store)],
) -> dict[str, str]:
    account_id = await run_in_threadpool(accounts.register, credentials.username, credentials.password)
    if account_id is None:
        raise HTTPException(status_code=409, detail="Username is already taken")
    token = await run_in_threadpool(accounts.issue_session, account_id)
    _set_account_cookie(response, token)
    return {"user_id": account_id}


@router.post("/login")
async def login(
    credentials: Credentials,
    response: Response,
    accounts: Annotated[AccountStore, Depends(get_account_store)],
) -> dict[str, str]:
    account_id = await run_in_threadpool(accounts.authenticate, credentials.username, credentials.password)
    if account_id is None:
        raise HTTPException(status_code=401, detail="Invalid credentials or too many attempts; try later")
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
