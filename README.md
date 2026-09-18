# ML Integration Lab 1

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