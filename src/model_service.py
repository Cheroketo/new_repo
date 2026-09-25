"""Прикладной слой: загрузка модели и предсказание.

API-слой не должен знать детали обучения
"""
import os
from functools import lru_cache

import joblib
import numpy as np
from sklearn.datasets import load_iris

DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")

FEATURE_ORDER = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


@lru_cache(maxsize=1)
def _load_model(model_path: str = DEFAULT_MODEL_PATH):
    """Загружает модель один раз и кэширует."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Модель не найдена: {model_path}. "
            f"Сначала обучите её: python src/train.py --output {model_path}"
        )
    return joblib.load(model_path)


def predict_one(features: list[list[float]]) -> int:
    """Принимает [[sepal_length, sepal_width, petal_length, petal_width]]."""
    model = _load_model()
    arr = np.asarray(features, dtype=float)
    return int(model.predict(arr)[0])


def predict_class_name(predicted_class: int) -> str:
    return str(load_iris().target_names[predicted_class])