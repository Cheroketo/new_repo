"""Клиент: отправляет POST /predict и сохраняет ответ в JSON-файл.

Адрес API берётся из API_URL — внутри compose это http://api:8000.
Путь сохранения — из RESULT_PATH, по умолчанию /results/prediction.json
(bind mount на ./results хоста).
"""

import json
import os
from pathlib import Path
import requests

API_URL = os.getenv("API_URL", "http://api:8000")
RESULT_PATH = Path(os.getenv("RESULT_PATH", "/results/prediction.json"))

PAYLOAD = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}


def main() -> None:
    url = f"{API_URL}/predict"
    print(f"[client] POST {url}")
    print(f"[client] payload={PAYLOAD}")

    response = requests.post(url, json=PAYLOAD, timeout=10)
    response.raise_for_status()
    result = response.json()

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[client] response saved to {RESULT_PATH}: {result}")


if __name__ == "__main__":
    main()