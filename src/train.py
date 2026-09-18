import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main():
    # Загрузка данных
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.25,
        random_state=42,
        stratify=iris.target,
    )

    # Обучение модели
    model = LogisticRegression(max_iter=300, random_state=42)
    model.fit(X_train, y_train)

    # Оценка качества
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"accuracy={accuracy:.3f}")

    # Сохранение модели
    joblib.dump(model, "models/model.pkl")
    print("model_saved=models/model.pkl")


if __name__ == "__main__":
    main()