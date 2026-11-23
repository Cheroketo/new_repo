import re
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_notifications_dir


def create_notification(
    notification_id: str,
    message: str,
    recipient: str | None = None,
) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", notification_id):
        raise ValueError("Недопустимый идентификатор уведомления")

    directory = get_notifications_dir()
    directory.mkdir(parents=True, exist_ok=True)

    file_path = directory / f"{notification_id}.txt"
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    lines = [
        f"Идентификатор: {notification_id}",
        f"Время создания: {created_at}",
    ]
    if recipient is not None:
        lines.append(f"Получатель: {recipient}")
    lines.append(f"Сообщение: {message}")

    with file_path.open("x", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    return file_path