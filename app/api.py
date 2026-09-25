"""FastAPI-приложение: HTTP-интерфейс над ML-моделью."""
from fastapi import FastAPI, HTTPException
from src import model_service
from app.schemas import HealthResponse, PredictRequest, PredictResponse
from src.model_service import (
    FEATURE_ORDER,
    predict_class_name,
    predict_one,
)

app = FastAPI(
    title="ML Integration API",
    description="HTTP-сервис классификации Iris",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Состояние сервиса и признак готовности модели."""
    try:
        from src import model_service
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
        predicted_class = predict_one(features)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return PredictResponse(
        prediction=predicted_class,
        class_name=predict_class_name(predicted_class),
    )