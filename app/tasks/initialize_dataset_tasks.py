from app.celery_config import celery_app
from app.services.initialize_service import initialize_dataset_service, resegment_dataset
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

@celery_app.task(bind=True)
def resegment_dataset_task(self, dataset_id: int, min_duration: float, max_duration: float):
    """
    Celery таск для ресегментации датасета
    
    Args:
        dataset_id: ID датасета для ресегментации
        min_duration: Минимальная длительность сегмента в секундах
        max_duration: Максимальная длительность сегмента в секундах
    """
    db = SessionLocal()
    try:
        logger.info(f"Запуск задачи ресегментации для dataset_id={dataset_id}")
        logger.info(f"Параметры: min_duration={min_duration}, max_duration={max_duration}")
        
        result = resegment_dataset(
            dataset_id=dataset_id, 
            min_duration=min_duration, 
            max_duration=max_duration, 
            db=db
        )
        
        logger.info(f"Ресегментация завершена успешно для dataset_id={dataset_id}")
        return result
        
    except Exception as e:
        logger.exception(f"Ошибка при выполнении задачи ресегментации Celery: {e}")
        # Повторяем задачу с экспоненциальной задержкой
        self.retry(exc=e, countdown=2 ** self.request.retries, max_retries=3)
        raise
    finally:
        db.close()