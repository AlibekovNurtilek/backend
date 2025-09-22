from app.celery_config import celery_app
from app.services.initialize_service import initialize_dataset_service
from app.db import SessionLocal
from app.schemas.dataset import DatasetInitRequest
import logging

logger = logging.getLogger(__name__)

@celery_app.task(bind=True)
def initialize_dataset_task(self, dataset_id: int, data_dict: dict):
    db = SessionLocal()
    try:
        data = DatasetInitRequest(**data_dict)
        logger.info(f"Запуск задачи инициализации для dataset_id={dataset_id}")
        result = initialize_dataset_service(dataset_id, data, db)
        return result
    except Exception as e:
        logger.exception(f"Ошибка при выполнении задачи Celery: {e}")
        self.retry(exc=e, countdown=10, max_retries=3)
        raise
    finally:
        db.close()
