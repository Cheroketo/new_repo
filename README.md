# ML Integration Lab 1 — Классификация Iris

Учебный проект по инженерной организации ML-кода: от исследовательского notebook до воспроизводимого пайплайна с обучением, инференсом и тестами.

---

## Цель проекта

Отработка практик промышленной разработки ML-кода на примере классической задачи — предсказания вида цветка Iris по четырём признакам (длина/ширина чашелистика и лепестка). Модель решает задачу мультиклассовой классификации на 3 класса.

---

## Что внутри

- **Источник:** исследовательский notebook `starter_ml.ipynb`, очищенная копия — `notebooks/source_experiment.ipynb`.
- **Пайплайн:** отдельные скрипты для обучения (`train.py`) и инференса (`predict.py`).
- **Артефакты:** сохранённая модель `models/model.pkl` и тестовый сэмпл `data_sample/sample.csv`.
- **Проверка:** автотесты на корректность предсказаний в `tests/`.

---

## Структура проекта

```
ml-integration-lab1/
├── README.md                      # Документация проекта
├── .gitignore                     # Исключения для Git
├── requirements.txt               # Зависимости Python
├── notebooks/
│   └── source_experiment.ipynb    # Исходный notebook
├── data_sample/
│   └── sample.csv                 # Тестовые данные
├── models/
│   └── model.pkl                  # Обученная модель
├── src/
│   ├── train.py                   # Скрипт обучения
│   └── predict.py                 # Скрипт предсказания
└── tests/
    └── test_predict.py            # Тесты
```

---

## 🚀 Инструкция по запуску

### 1. Клонирование репозитория

```bash
git clone https://github.com/Cheroketo/new_repo.git
cd ml-integration-lab1
```

### 2. Создание и активация виртуального окружения

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

> После активации в начале строки терминала появится плашка `(.venv)`.

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Запуск и проверка модулей

**Обучение модели** — обучаем классификатор и сохраняем веса:

```bash
python src/train.py
```

**Инференс** — предсказание по данным из `data_sample/sample.csv`:

```bash
python src/predict.py
```

**Тесты** — проверка корректности интеграции:

```bash
pytest tests/test_predict.py
```

---

## Стек

Python · scikit-learn · pandas · numpy · pytest

---

## Ключевые инженерные решения

| Практика | Реализация |
|---|---|
| Воспроизводимость | фиксированный `random_state`, `requirements.txt` |
| Разделение кода | `train.py` / `predict.py` вместо монолитного notebook |
| Изоляция окружения | `.venv` + `.gitignore` |
| Тестируемость | отдельный модуль `tests/` с pytest |
| Персистентность модели | сериализация в `models/model.pkl` |


## Конфигурация

Параметры запуска задаются **аргументами CLI** или **переменными окружения**.

### `src/train.py`

| Аргумент | Env | По умолчанию | Описание |
|---|---|---|---|
| `--output` | `MODEL_PATH` | `models/model.pkl` | Куда сохранить обученную модель |
| `--test-size` | — | `0.25` | Доля тестовой выборки |
| `--random-state` | — | `42` | Seed для воспроизводимости |

Пример:

```bash
python src/train.py --output artifacts/model.pkl --test-size 0.3
```

### `src/predict.py`

| Аргумент | Env | По умолчанию | Описание |
|---|---|---|---|
| `--model` | `MODEL_PATH` | `models/model.pkl` | Путь к обученной модели |
| `--sample` | `SAMPLE_PATH` | `data_sample/sample.csv` | Путь к CSV с признаками |
| `--format` | `OUTPUT_FORMAT` | `plain` | Формат вывода: `plain` или `csv` |

Примеры:

```bash
# через CLI
python src/predict.py --model artifacts/model.pkl --format csv

# через переменную окружения
MODEL_PATH=artifacts/model.pkl python src/predict.py
```

### Проверка

```bash
pytest tests/test_predict.py -v
```
## API-сервис (FastAPI)

### Запуск

```bash
pip install -r requirements.txt
python src/train.py                          # обучить модель
python -m uvicorn app.api:app --reload       # запустить сервис
```

Сервис доступен на `http://127.0.0.1:8000`.

## Уровни тестирования

| Уровень | Файл | Что проверяет |
|---|---|---|
| Модульные тесты | `tests/test_model_service.py` | Логика без HTTP: `predict_one`, `predict_class_name`, кэш модели |
| API-тесты | `tests/test_api.py` | Маршруты `/health`, `/predict` через `TestClient` |
| Негативные тесты | `tests/test_api.py` | Отсутствующее поле → 422, отрицательное значение → 422 |
| Интеграционный тест | `tests/test_api.py::test_integration_real_model_predicts_setosa` | Сервис + реальный `models/model.pkl` |
| Тесты конфигурации | `tests/test_predict.py` | CLI-аргументы и env-переменные `predict.py` / `train.py` |

### Запуск

```bash
pytest                    # все тесты
pytest -v                 # с деталями
pytest tests/test_api.py -v          # только API-тесты
pytest tests/test_model_service.py -v  # только модульные
```


### Маршруты

| Метод | Путь | Описание |
|---|---|---|
| GET | `/health` | Состояние сервиса и готовность модели |
| POST | `/predict` | Предсказание вида Iris |
| GET | `/docs` | Swagger UI |
| GET | `/openapi.json` | OpenAPI-схема |

### Пример запроса

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

### Пример ответа

```json
{"prediction": 0, "class_name": "setosa"}
```

### Ограничения учебной модели

- Обучена на встроенном датасете Iris, применима только к 3 видам.
- Модель загружается один раз при старте и кэшируется — при обновлении `.pkl` нужен перезапуск сервиса.
- Путь к модели задаётся через env `MODEL_PATH`.



## Лицензия

Учебный проект, распространяется свободно.
