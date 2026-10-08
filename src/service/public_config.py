"""Fail closed on clearly unsafe public deployment settings."""

import ipaddress

from email_validator import EmailNotValidError, validate_email

from core import settings
from service.csrf import _origin


def validate_public_configuration() -> None:
    if not settings.EVIDENCEGRAPH_PUBLIC_MODE:
        return
    errors: list[str] = []
    if not settings.EVIDENCEGRAPH_COOKIE_SECURE:
        errors.append("EVIDENCEGRAPH_COOKIE_SECURE must be true")
    if not settings.EVIDENCEGRAPH_REQUIRE_LOGIN_FOR_CHAT:
        errors.append("EVIDENCEGRAPH_REQUIRE_LOGIN_FOR_CHAT must be true")
    if not settings.EVIDENCEGRAPH_QUOTAS_ENABLED:
        errors.append("EVIDENCEGRAPH_QUOTAS_ENABLED must be true")
    if not settings.EVIDENCEGRAPH_EMAIL_VERIFICATION_REQUIRED:
        errors.append("EVIDENCEGRAPH_EMAIL_VERIFICATION_REQUIRED must be true")
    if settings.RESEND_API_KEY is None:
        errors.append("RESEND_API_KEY must be configured")
    try:
        if settings.RESEND_FROM_EMAIL is None:
            raise EmailNotValidError("Missing sender")
        validate_email(settings.RESEND_FROM_EMAIL, check_deliverability=False)
    except EmailNotValidError:
        errors.append("RESEND_FROM_EMAIL must be a valid sender address")
    origins = [part.strip() for part in settings.EVIDENCEGRAPH_TRUSTED_ORIGINS.split(",")]
    if not origins or any(
        not part or _origin(part, allow_path=False) is None or not part.startswith("https://")
        for part in origins
    ):
        errors.append("EVIDENCEGRAPH_TRUSTED_ORIGINS must contain only exact HTTPS origins")
    proxies = [part.strip() for part in settings.EVIDENCEGRAPH_TRUSTED_PROXY_CIDRS.split(",")]
    if not any(proxies):
        errors.append("EVIDENCEGRAPH_TRUSTED_PROXY_CIDRS must identify the trusted proxy")
    else:
        try:
            for value in proxies:
                ipaddress.ip_network(value, strict=False)
        except ValueError:
            errors.append("EVIDENCEGRAPH_TRUSTED_PROXY_CIDRS contains an invalid CIDR")
    if settings.AUTH_SECRET is not None:
        errors.append("AUTH_SECRET must be unset for browser public routes")
    if errors:
        raise RuntimeError("Public mode configuration is incomplete: " + "; ".join(errors))
