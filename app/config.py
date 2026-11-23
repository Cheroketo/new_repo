import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_notifications_dir() -> Path:
    default_dir = PROJECT_ROOT / "var" / "notifications"
    configured_dir = os.getenv("NOTIFICATIONS_DIR")
    return Path(configured_dir) if configured_dir else default_dir