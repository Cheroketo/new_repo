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
python src/train.py                         
python -m uvicorn app.api:app --reload     
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

## Docker

### Сборка и запуск

```bash
docker build -t ml-api:practice5 .
docker run -d --name ml-api-p5 -p 8080:8000 -e MODEL_PATH=/app/models/model.pkl ml-api:practice5
```

Порт `8080` хоста → порт `8000` контейнера. Внутри uvicorn слушает `0.0.0.0:8000`.

### Проверка

```bash
curl -i http://localhost:8080/health
curl -i -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

### Диагностика и остановка

```bash
docker ps
docker logs ml-api-p5
docker exec ml-api-p5 ls -la /app
docker stats --no-stream ml-api-p5

docker stop ml-api-p5
docker rm ml-api-p5
```

Модель в образе — снимок `models/model.pkl` на момент сборки. Обучение в Dockerfile не выполняется.

---

## Docker Compose

Стенд из двух сервисов: `api` и `client`, соединённых bridge-сетью `app_net`. Результат сохраняется на хосте через bind mount.

### Архитектура

```
Хост ──► localhost:8080/health ──► ┌──────────────┐
                                    │     api      │ :8000
                                    └───────┬──────┘
                                            │ app_net
                                            │ http://api:8000
                                    ┌───────┴──────┐
                                    │    client    │
                                    └───────┬──────┘
                                            │ bind mount
                                            ▼
                                    ./results/prediction.json
```

Внутри сети Compose сервисы видят друг друга по **именам сервисов** (DNS). Клиент использует `http://api:8000`, хост: `localhost:8080`.

### Запуск и проверка

```bash
docker compose up --build -d
docker compose ps -a 
docker compose logs api
docker compose logs client
curl -i http://localhost:8080/health
cat results/prediction.json
# → {"prediction": 0, "class_name": "setosa"}
```

### Ошибка localhost

Внутри `client` `localhost` — это сам client, а не api-контейнер.

```bash
# ОШИБКА ConnectionError
docker compose run --rm -e API_URL=http://localhost:8000 client

# ПРАВИЛЬНО
docker compose run --rm client
```

### Остановка

```bash
docker compose down
cat results/prediction.json   

### Сервисы

| Сервис | Контекст | Образ | Порты |
|---|---|---|---|
| `api` | `.` | `ml-api:practice6` | `8080:8000` |
| `client` | `./client` | `ml-client:practice6` | — |


##  CI/CD

CI-конвейер работает в **GitHub Actions** — файл `.github/workflows/ci.yaml`.

### События запуска

| Событие | Условие |
|---|---|
| `push` | изменения в `main` (фильтр: `app/`, `src/`, `tests/`, `Dockerfile`, `requirements.txt`, `.github/workflows/`) |
| `pull_request` | создание или обновление PR |
| `workflow_dispatch` | вручную через UI GitHub |

### Jobs

| Job | Что делает | Длительность |
|---|---|---|
| **check** | Python 3.12, установка зависимостей, `python -m compileall app src`, `pytest -q` | ~30s |
| **build-image** | после `check` — сборка Docker-образа через **Kaniko** (без docker.sock), `--no-push` | ~30s |

`build-image` использует `needs: check` — не запускается, если тесты упали.

### Локальная проверка перед push

```bash
pytest -q
docker build -t ml-api:test .
```

### Ссылка на последний успешный запуск

https://github.com/Cheroketo/new_repo/actions

## Delivery (CD) — dry run

Второй workflow — `.github/workflows/delivery.yaml` — моделирует доставку сервиса **без реального сервера**. Все команды **печатаются в лог**, не выполняются.

### Событие запуска

Только вручную через `workflow_dispatch` в GitHub Actions:
Actions → **Test Delivery Dry Run** → **Run workflow** → ветка `main`.

### Jobs

| Job | Что делает | Зависит от |
|---|---|---|
| **build** | Формирует тег `test-<sha>`, публикует через `outputs.image-tag` | — |
| **smoke_api_tests_stub** | Печатает команды `curl /health`, `curl /predict` | build |
| **docs_checks** | Проверяет наличие README | build |
| **deploy_dry_run** | Печатает `docker pull / stop / rm / run`, smoke-команды и команды **rollback** | build + smoke + docs |

`smoke_api_tests_stub` и `docs_checks` идут **параллельно**, `deploy_dry_run` ждёт **все три** через `needs`.

### Заглушки

- `registry.example.local/ml-api` — несуществующий registry.
- `test.example.local:8080` — несуществующий тестовый сервер.
- `PREVIOUS_TAG=previous-stable` — заглушка предыдущего проверенного тега.

### Команды, которые выполнились бы при реальной доставке

```bash
docker pull registry.example.local/ml-api:test-<sha>
docker stop ml-api-test || true
docker rm ml-api-test || true
docker run -d --name ml-api-test -p 8080:8000 registry.example.local/ml-api:test-<sha>
curl -f http://test.example.local:8080/health
curl -X POST http://test.example.local:8080/predict -H "Content-Type: application/json" --data '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

### Откат

При проблемах — вернуть предыдущий проверенный тег:

```bash
docker stop ml-api-test || true
docker rm ml-api-test || true
docker run -d --name ml-api-test -p 8080:8000 registry.example.local/ml-api:previous-stable
curl -f http://test.example.local:8080/health
```

### Переход к реальной доставке

Чтобы заменить dry run на реальный деплой:

1. Развернуть тестовый сервер с Docker.
2. Добавить секреты в GitHub: `DEPLOY_HOST`, `DEPLOY_USER`, `SSH_KEY`, `REGISTRY_TOKEN`.
3. Заменить `echo` на реальные `ssh`-вызовы:
   ```bash
   ssh "$DEPLOY_USER@$DEPLOY_HOST" "docker pull $IMAGE_NAME:$IMAGE_TAG"
   ```
4. Убрать `--no-push` в сборке и настроить публикацию в registry.

## Лицензия

Учебный проект, распространяется свободно.
