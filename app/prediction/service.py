from uuid import uuid4

from app.notification_client import send_notification
from src.model_service import predict_class_name, predict_one


def predict_and_notify(features: list[list[float]]) -> tuple[int, str]:
    predicted_class = predict_one(features)
    class_name = predict_class_name(predicted_class)

    send_notification(
        message_id=uuid4(),
        text=f"Предсказанный вид Iris: {class_name} (класс {predicted_class}).",
    )

    return predicted_class, class_name
