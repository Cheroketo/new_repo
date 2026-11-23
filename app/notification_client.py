"""HTTP-клиент отдельного сервиса уведомлений."""

import os
from uuid import UUID

import httpx


class NotificationIntegrationError(Exception):
    """Сервис уведомлений не подтвердил запрос."""


def send_notification(message_id: UUID, text: str, recipient: str | None = None) -> None:
    base_url = os.getenv("NOTIFICATIONS_URL", "http://127.0.0.1:8001")
    payload = {"message_id": str(message_id), "text": text, "recipient": recipient}
    try:
        response = httpx.post(
            f"{base_url.rstrip('/')}/notifications",
            json=payload,
            timeout=3.0,
        )
    except httpx.RequestError as exc:
        raise NotificationIntegrationError("Сервис уведомлений недоступен.") from exc

    if response.status_code not in (200, 201):
        raise NotificationIntegrationError(
            f"Сервис уведомлений не принял запрос (HTTP {response.status_code})."
        )
