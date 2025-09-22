# notifications.py

import redis
import json

# Настройка Redis-клиента
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

def notify_progress(dataset_id: int, task: str, progress: int):
    message = {
        "dataset_id": dataset_id,
        "task": task,
        "progress": progress
    }
    # Публикуем сообщение в канал
    r.publish("ws_progress", json.dumps(message))
