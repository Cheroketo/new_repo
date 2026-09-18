import importlib
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_model_exists():
    """Проверяет, что файл модели существует."""
    assert os.path.exists("models/model.pkl"), "Файл models/model.pkl не найден"


def test_predict_script_runs():
    """Проверяет, что скрипт predict.py запускается без ошибок."""
    result = subprocess.run(
        [sys.executable, "src/predict.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Скрипт завершился с ошибкой: {result.stderr}"
    assert "predicted_class=" in result.stdout, "Ожидаемый вывод не найден"
    assert "predicted_name=" in result.stdout, "Ожидаемый вывод не найден"


def test_default_model_path_env_override(monkeypatch):
    """MODEL_PATH из окружения подхватывается как значение по умолчанию."""
    monkeypatch.setenv("MODEL_PATH", "custom/model.pkl")
    import src.predict as predict_module
    importlib.reload(predict_module)
    assert predict_module.DEFAULT_MODEL_PATH == "custom/model.pkl"


def test_default_sample_path_env_override(monkeypatch):
    """SAMPLE_PATH из окружения подхватывается как значение по умолчанию."""
    monkeypatch.setenv("SAMPLE_PATH", "custom/sample.csv")
    import src.predict as predict_module
    importlib.reload(predict_module)
    assert predict_module.DEFAULT_SAMPLE_PATH == "custom/sample.csv"


def test_predict_script_runs_with_custom_paths(tmp_path):
    """CLI: train --output + predict --model + --sample + --format csv."""
    model_path = tmp_path / "model.pkl"
    subprocess.run(
        [sys.executable, "src/train.py", "--output", str(model_path)],
        cwd=PROJECT_ROOT, check=True,
    )
    assert model_path.exists()

    sample_path = PROJECT_ROOT / "data_sample" / "sample.csv"
    result = subprocess.run(
        [
            sys.executable, "src/predict.py",
            "--model", str(model_path),
            "--sample", str(sample_path),
            "--format", "csv",
        ],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
    )
    assert "predicted_class" in result.stdout
    assert "predicted_name" in result.stdout


def test_predict_plain_format_still_default(tmp_path):
    """Без --format вывод остаётся plain (обратная совместимость)."""
    model_path = tmp_path / "model.pkl"
    subprocess.run(
        [sys.executable, "src/train.py", "--output", str(model_path)],
        cwd=PROJECT_ROOT, check=True,
    )
    sample_path = PROJECT_ROOT / "data_sample" / "sample.csv"
    result = subprocess.run(
        [
            sys.executable, "src/predict.py",
            "--model", str(model_path),
            "--sample", str(sample_path),
        ],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
    )
    assert "predicted_class=" in result.stdout
    assert "predicted_name=" in result.stdout


def test_train_creates_output_directory(tmp_path):
    """train.py создаёт вложенную папку под модель."""
    nested = tmp_path / "nested" / "dir" / "model.pkl"
    subprocess.run(
        [sys.executable, "src/train.py", "--output", str(nested)],
        cwd=PROJECT_ROOT, check=True,
    )
    assert nested.exists()