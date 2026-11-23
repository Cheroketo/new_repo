from uuid import uuid4

from app.notifications.service import create_notification
from src.model_service import predict_class_name, predict_one


def predict_and_notify(features: list[list[float]]) -> tuple[int, str]:
    predicted_class = predict_one(features)
    class_name = predict_class_name(predicted_class)

    create_notification(
        notification_id=uuid4().hex,
        message=f"Предсказанный вид Iris: {class_name} (класс {predicted_class}).",
    )

    return predicted_class, class_name