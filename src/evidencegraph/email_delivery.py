"""Small transactional-email boundary for account verification."""

from __future__ import annotations

from typing import Protocol

import httpx


class VerificationSender(Protocol):
    def send_verification(self, email: str, code: str) -> None: ...


class EmailDeliveryError(RuntimeError):
    """The provider did not accept the verification message."""


class ResendVerificationSender:
    def __init__(
        self, api_key: str, from_email: str, *, client: httpx.Client | None = None
    ) -> None:
        self.api_key = api_key
        self.from_email = from_email
        self.client = client

    def send_verification(self, email: str, code: str) -> None:
        payload = {
            "from": self.from_email,
            "to": [email],
            "subject": "Your EvidenceGraph verification code",
            "text": (
                f"Your EvidenceGraph verification code is {code}. "
                "It expires in 20 minutes. If you did not register, ignore this email."
            ),
        }
        try:
            if self.client is None:
                with httpx.Client(timeout=10) as client:
                    response = client.post(
                        "https://api.resend.com/emails",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json=payload,
                    )
            else:
                response = self.client.post(
                    "https://api.resend.com/emails",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EmailDeliveryError("Verification email was not accepted") from exc
