import os
import subprocess
import sys


def test_model_exists():
    """Проверяет, что файл модели существует"""
    assert os.path.exists("models/model.pkl"), "Файл models/model.pkl не найден"


def test_predict_script_runs():
    """Проверяет, что скрипт predict.py запускается без ошибок"""
    result = subprocess.run(
        [sys.executable, "src/predict.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Скрипт завершился с ошибкой: {result.stderr}"
    assert "predicted_class=" in result.stdout, "Ожидаемый вывод не найден"
    assert "predicted_name=" in result.stdout, "Ожидаемый вывод не найден"