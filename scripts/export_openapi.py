"""Экспорт OpenAPI-схемы FastAPI-приложения в JSON.

Используется в CI: импортирует app и записывает app.openapi() в файл.
Сервер (uvicorn) не запускается.
"""
import argparse
import json
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path, чтобы работал импорт `app`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api import app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Путь к выходному openapi.json")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(app.openapi(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"OpenAPI schema saved to {out}")


if __name__ == "__main__":
    main()