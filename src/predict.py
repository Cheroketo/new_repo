import argparse
import csv
import os

import joblib
from sklearn.datasets import load_iris

DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
DEFAULT_SAMPLE_PATH = os.getenv("SAMPLE_PATH", "data_sample/sample.csv")
DEFAULT_OUTPUT_FORMAT = os.getenv("OUTPUT_FORMAT", "plain") 

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Iris model inference")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Path to trained model (.pkl)",
    )
    parser.add_argument(
        "--sample",
        default=DEFAULT_SAMPLE_PATH,
        help="Path to input CSV with a single row of features",
    )
    parser.add_argument(
        "--format",
        default=DEFAULT_OUTPUT_FORMAT,
        choices=["plain", "csv"],
        help="Output format: plain text or csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Загрузка модели
    model = joblib.load(args.model)

    # Загрузка тестовых данных
    with open(args.sample, encoding="utf-8", newline="") as sample_file:
        row = next(csv.DictReader(sample_file))

    features = [[float(row[column]) for column in FEATURE_COLUMNS]]

    # Предсказание
    predicted_class = int(model.predict(features)[0])
    iris = load_iris()
    predicted_name = iris.target_names[predicted_class]

    # Вывод
    if args.format == "csv":
        writer = csv.writer(__import__("sys").stdout)
        writer.writerow(["predicted_class", "predicted_name"])
        writer.writerow([predicted_class, predicted_name])
    else:
        print(f"predicted_class={predicted_class}")
        print(f"predicted_name={predicted_name}")


if __name__ == "__main__":
    main()