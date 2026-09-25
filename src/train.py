import argparse
import os

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DEFAULT_MODEL_OUTPUT = os.getenv("MODEL_PATH", "models/model.pkl")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Iris classifier")
    parser.add_argument(
        "--output",
        default=DEFAULT_MODEL_OUTPUT,
        help="Where to save trained model (.pkl)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Test split fraction (default: 0.25)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Загрузка данных
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=iris.target,
    )

    # Обучение модели
    model = LogisticRegression(max_iter=300, random_state=args.random_state)
    model.fit(X_train, y_train)

    # Оценка качества
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"accuracy={accuracy:.3f}")

    # Сохранение модели
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    joblib.dump(model, args.output)
    print(f"model_saved={args.output}")


if __name__ == "__main__":
    main()