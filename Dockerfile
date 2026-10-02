FROM python:3.12-slim

WORKDIR /app

# Сначала зависимости — так слой кэшируется и не пересобирается при правках кода
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Затем код приложения и модель
COPY app ./app
COPY src ./src
COPY models ./models


ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/model.pkl

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.api:app", \
     "--host", "0.0.0.0", "--port", "8000"]