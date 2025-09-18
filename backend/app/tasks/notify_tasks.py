from app.notifications import notify_progress
from app.celery_config import celery_app


@celery_app.task
def notify_progress_task(dataset_id: int, task: str, progress: int):
    notify_progress(dataset_id=dataset_id, task=task, progress=progress)
