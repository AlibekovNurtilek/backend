from app.celery_config import celery_app
from app.services.transcription_service import transcribe_dataset
from app.db import SessionLocal
import logging

logger = logging.getLogger(__name__)

@celery_app.task(bind=True)
def transcribe_dataset_task(self, dataset_id: int, transcriber_id: int):
    db = SessionLocal()
    try:
        logger.info(f"Запуск Celery-транскрипции: dataset_id={dataset_id}, transcriber_id={transcriber_id}")
        result = transcribe_dataset(dataset_id, transcriber_id, db)
        return result
    except Exception as e:
        logger.exception(f"Ошибка при выполнении транскрипции: {e}")
        self.retry(exc=e, countdown=10, max_retries=3)
        raise
    finally:
        db.close()
