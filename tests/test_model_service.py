"""Модульные тесты прикладного слоя src/model_service.py.

Не используют HTTP и FastAPI (только чистые функции)
"""
import os

import pytest

from src import model_service
from src.model_service import FEATURE_ORDER, predict_class_name, predict_one


def test_feature_order_matches_iris_columns():
    """Порядок признаков"""
    assert FEATURE_ORDER == [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]


def test_predict_one_returns_int():
    """predict_one возвращает int, не np.int64 и не список."""
    model_service._load_model.cache_clear()
    result = predict_one([[5.1, 3.5, 1.4, 0.2]])
    assert isinstance(result, int)


def test_predict_one_setosa():
    """Пример setosa предсказывается как класс 0."""
    model_service._load_model.cache_clear()
    result = predict_one([[5.1, 3.5, 1.4, 0.2]])
    assert result == 0


def test_predict_one_virginica():
    """Пример virginica предсказывается как класс 2"""
    model_service._load_model.cache_clear()
    result = predict_one([[6.3, 3.3, 6.0, 2.5]])
    assert result == 2


def test_predict_class_name_returns_string():
    """predict_class_name возвращает читаемое имя класса."""
    assert predict_class_name(0) == "setosa"
    assert predict_class_name(1) == "versicolor"
    assert predict_class_name(2) == "virginica"


def test_load_model_raises_when_file_missing(tmp_path):
    """_load_model с несуществующим путём даёт FileNotFoundError."""
    missing = tmp_path / "nope.pkl"
    with pytest.raises(FileNotFoundError, match="Модель не найдена"):
        model_service._load_model(str(missing))


def test_load_model_caches():
    """Модель загружается один раз и повторный вызов не перечитывает файл."""
    model_service._load_model.cache_clear()
    first = model_service._load_model()
    second = model_service._load_model()
    assert first is second  # тот же объект из кэша