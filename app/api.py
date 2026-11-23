"""FastAPI-приложение: HTTP-интерфейс над ML-моделью."""
from fastapi import FastAPI, HTTPException

from app.notification_client import NotificationIntegrationError
from app.prediction.service import predict_and_notify
from app.schemas import HealthResponse, PredictRequest, PredictResponse
from src import model_service

app = FastAPI(
    title="ML Integration API",
    description="HTTP-сервис классификации Iris",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Состояние сервиса и признак готовности модели."""
    try:
        model_service._load_model()
        model_ready = True
    except FileNotFoundError:
        model_ready = False

    return HealthResponse(
        status="ok" if model_ready else "degraded",
        model_ready=model_ready,
    )


@app.post("/predict", response_model=PredictResponse, tags=["ml"])
def predict(request: PredictRequest) -> PredictResponse:
    """Предсказание вида Iris по четырём признакам."""
    features = [[
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width,
    ]]

    try:
        predicted_class, class_name = predict_and_notify(features)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    except NotificationIntegrationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return PredictResponse(
        prediction=predicted_class,
        class_name=class_name,
    )
