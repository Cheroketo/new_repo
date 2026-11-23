"""Тесты FastAPI-сервиса."""
import json
from uuid import UUID

import httpx
import pytest
from fastapi.testclient import TestClient

from app import notification_client
from app.api import app

client = TestClient(app)


@pytest.fixture
def sent_notifications(monkeypatch):
    outgoing = []
    monkeypatch.setenv("NOTIFICATIONS_URL", "http://notifications.test:8001")

    def receive(request):
        outgoing.append(request)
        return httpx.Response(201, json={"status": "created", "filename": "sample.txt"})

    with httpx.Client(transport=httpx.MockTransport(receive)) as transport_client:
        monkeypatch.setattr(notification_client.httpx, "post", transport_client.post)
        yield outgoing


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_ready" in body


def test_predict_valid_input(sent_notifications):
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 0
    assert body["class_name"] == "setosa"
    request, = sent_notifications
    assert str(request.url) == "http://notifications.test:8001/notifications"
    payload = json.loads(request.content)
    assert UUID(payload["message_id"]).version == 4
    assert payload["text"] == "Предсказанный вид Iris: setosa (класс 0)."
    assert request.extensions["timeout"]["read"] == 3.0



def test_predict_rejects_missing_field():
    payload = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any(err["type"] == "missing" for err in detail)
    assert any(err["loc"][-1] == "petal_width" for err in detail)


def test_predict_rejects_negative_values():
    payload = {"sepal_length": -1, "sepal_width": 3.5,
               "petal_length": 1.4, "petal_width": 0.2}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any(err["type"] == "greater_than" for err in detail)


def test_openapi_available():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()
    assert "/predict" in spec["paths"]
    assert "/health" in spec["paths"]



def test_health_degraded_when_model_missing(monkeypatch):
    """Если модель недоступна — status=degraded, model_ready=false."""
    from src import model_service

    model_service._load_model.cache_clear()

    def raise_fnf(*args, **kwargs):
        raise FileNotFoundError("no model")

    monkeypatch.setattr(model_service, "_load_model", raise_fnf)

    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["model_ready"] is False


def test_integration_real_model_predicts_setosa(sent_notifications):
    """Интеграционный тест это сервис + реальная сохранённая модель.
    Проверяет, что /predict действительно использует models/model.pkl,а не заглушку.
    """
    from src import model_service

    model_service._load_model.cache_clear()

    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()

    # конкретный ожидаемый класс , что модель реально работает
    assert body["prediction"] == 0
    assert body["class_name"] == "setosa"


def test_predict_returns_integration_error_when_notifications_unavailable(monkeypatch):
    def unavailable(request):
        raise httpx.ConnectError("offline", request=request)

    with httpx.Client(transport=httpx.MockTransport(unavailable)) as transport_client:
        monkeypatch.setattr(notification_client.httpx, "post", transport_client.post)
        response = client.post("/predict", json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        })

    assert response.status_code == 502
    assert response.json() == {"detail": "Сервис уведомлений недоступен."}
