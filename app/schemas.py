"""Pydantic-схемы запроса и ответа для API."""
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    sepal_length: float = Field(gt=0, description="Длина чашелистика, см")
    sepal_width: float = Field(gt=0, description="Ширина чашелистика, см")
    petal_length: float = Field(gt=0, description="Длина лепестка, см")
    petal_width: float = Field(gt=0, description="Ширина лепестка, см")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2,
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    prediction: int = Field(description="Номер предсказанного класса (0..2)")
    class_name: str = Field(description="Название вида Iris")


class HealthResponse(BaseModel):
    status: str
    model_ready: bool