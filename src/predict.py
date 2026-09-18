import csv

import joblib
from sklearn.datasets import load_iris

   # Конфигурация путей
MODEL_PATH = "models/model.pkl"
SAMPLE_PATH = "data_sample/sample.csv"

def main():
    # Загрузка модели
    model = joblib.load(MODEL_PATH)

    # Загрузка тестовых данных
    feature_columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]

    with open(SAMPLE_PATH, encoding="utf-8", newline="") as sample_file:
        row = next(csv.DictReader(sample_file))

    features = [[float(row[column]) for column in feature_columns]]

    # Предсказание
    predicted_class = int(model.predict(features)[0])
    iris = load_iris()
    predicted_name = iris.target_names[predicted_class]

    print(f"predicted_class={predicted_class}")
    print(f"predicted_name={predicted_name}")


if __name__ == "__main__":
    main()