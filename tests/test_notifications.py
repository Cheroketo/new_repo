import pytest

from app.notifications.service import create_notification


def test_creates_notification_file(tmp_path, monkeypatch):
    directory = tmp_path / "notifications"
    monkeypatch.setenv("NOTIFICATIONS_DIR", str(directory))

    path = create_notification(
        notification_id="iris-001",
        message="Предсказанный вид Iris: setosa.",
        recipient="Студент",
    )

    assert path == directory / "iris-001.txt"
    assert path.is_file()

    text = path.read_text(encoding="utf-8")
    assert "Идентификатор: iris-001" in text
    assert "Время создания: " in text
    assert "Получатель: Студент" in text
    assert "Сообщение: Предсказанный вид Iris: setosa." in text


def test_rejects_notification_id_with_path(tmp_path, monkeypatch):
    directory = tmp_path / "notifications"
    monkeypatch.setenv("NOTIFICATIONS_DIR", str(directory))

    with pytest.raises(ValueError, match="идентификатор"):
        create_notification("../outside", "Проверка")

    assert not directory.exists()