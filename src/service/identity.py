"""Bind private conversation operations to their server-issued principal."""

from typing import Annotated

from fastapi import Depends, HTTPException

from core import settings
from service.auth import get_optional_account
from service.papers import get_guest_identity


def strict_identity_enabled() -> bool:
    return settings.EVIDENCEGRAPH_ENFORCE_SESSION_IDENTITY or settings.EVIDENCEGRAPH_REQUIRE_LOGIN_FOR_CHAT


def get_chat_identity(
    guest_id: Annotated[str, Depends(get_guest_identity)],
    account_id: Annotated[str | None, Depends(get_optional_account)],
) -> str | None:
    """Keep legacy mode intact; account mode never trusts a caller-supplied user ID."""
    return account_id if settings.EVIDENCEGRAPH_REQUIRE_LOGIN_FOR_CHAT else guest_id


def assert_session_owner(supplied_user_id: str | None, principal_id: str | None) -> None:
    if settings.EVIDENCEGRAPH_REQUIRE_LOGIN_FOR_CHAT and principal_id is None:
        raise HTTPException(status_code=401, detail="Sign in to use research chat")
    if strict_identity_enabled() and supplied_user_id != principal_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
