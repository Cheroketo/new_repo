## Цель проекта

Учебный проект для отработки инженерной организации ML-кода. Модель классифицирует цветы Iris по четырем признакам.

## Исходная заготовка

Исходный код взят из файла `starter_ml.ipynb` — исследовательского notebook с датасетом Iris. Очищенная копия сохранена в `notebooks/source_experiment.ipynb`.

## Структура проекта

```text
ml-integration-lab1/
├── README.md              # Документация проекта
├── .gitignore             # Исключения для Git
├── requirements.txt       # Зависимости Python
├── notebooks/
│   └── source_experiment.ipynb  # Исходный notebook
├── data_sample/
│   └── sample.csv         # Тестовые данные
├── models/
│   └── model.pkl          # Обученная модель
├── src/
│   ├── train.py           # Скрипт обучения
│   └── predict.py         # Скрипт предсказания
└── tests/
    └── test_predict.py    # Тесты

# Инструкция по запуску проекта
1. Клонируем репозиторий к себе на компьютер: clone https://github.com
2. Создаем и активируем изолированное виртуальное окружение:Если у вас macOS / Linux:bashpython3 -m venv .venv
source .venv/bin/activate(После этого в начале строки терминала должна загореться плашка (.venv))
3. Устанавливаем все необходимые библиотеки:pip install -r requirements.txt
4. Запуск и проверка модулей проекта:Обучение модели: Обучаем классификатор и сохраняем веса команды:python src/train.py
Запуск предсказания: Делаем инференс по тестовым данным из data_sample/sample.csv:python src/predict.py
Запуск тестов: Проверяем корректность работы интеграции (требуется pytest, который установился из requirements):pytest tests/test_predict.py
