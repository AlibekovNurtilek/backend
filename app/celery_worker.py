from app.celery_config import celery_app

# Импортируем задачи, чтобы они зарегистрировались в Celery
import app.tasks.initialize_dataset_tasks  
import app.tasks.transcription_tasks
import app.tasks.notify_tasks