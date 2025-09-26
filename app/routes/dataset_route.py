from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Annotated, Optional
from datetime import datetime
from fastapi import Query
from datetime import datetime
from app.db import SessionLocal
from pydantic import BaseModel
from app.models.datasets import AudioDataset
from app.services import dataset_service
from app.schemas.dataset import DatasetListResponse, DatasetResegmentRequest
from app.schemas.dataset import (
    DatasetCreate,
    DatasetUpdate,
    DatasetOut,
    DatasetInitRequest,
    DatasetImageUpdate
)
from app.tasks.initialize_dataset_tasks import initialize_dataset_task, resegment_dataset_task
from app.services.initialize_service import create_dataset_entry
import logging
logger = logging.getLogger(__name__)



router = APIRouter(prefix="/datasets", tags=["datasets"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



# ---------------------
# 📍 Роуты
# ---------------------




@router.get("/", response_model=DatasetListResponse)
def get_all_datasets(
    page: int = Query(1, ge=1),   # номер страницы (по умолчанию 1)
    limit: int = Query(10, ge=1), # количество элементов на странице
    status: Optional[str] = None,
    name_search: Optional[str] = None,
    created_from: Optional[datetime] = None,
    created_to: Optional[datetime] = None,
    db: Session = Depends(get_db),
):
    return dataset_service.get_all_datasets(
        db=db,
        limit=limit,
        page=page,
        status=status,
        name_search=name_search,
        created_from=created_from,
        created_to=created_to,
    )

@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset_by_id(dataset_id: int, db: Session = Depends(get_db)):
    return dataset_service.get_dataset_by_id(dataset_id, db)


@router.put("/{dataset_id}", response_model=DatasetOut)
def update_dataset(dataset_id: int, dataset: DatasetUpdate, db: Session = Depends(get_db)):
    return dataset_service.update_dataset(dataset_id, dataset, db)

@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    dataset_service.delete_dataset(dataset_id, db)
    return


@router.post("/initialize")
async def initialize_dataset(data: DatasetInitRequest, db: Session = Depends(get_db)):
    # Шаг 0: проверяем наличие такого же URL в базе
    existing_dataset = db.query(AudioDataset).filter(AudioDataset.url == data.url).first()
    if existing_dataset:
        logger.info(f"Датасет с таким URL уже существует: ID={existing_dataset.id}, name={existing_dataset.name}")
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Видео уже загружено",
                "dataset_id": existing_dataset.id,
                "status": existing_dataset.status
            }
        )

    # Шаг 1: создаём запись в БД со статусом INITIALIZING
    dataset_id = await create_dataset_entry(db, data.url)

    # Шаг 2: запускаем фоновую задачу
    task = initialize_dataset_task.delay(dataset_id, data.dict())

    return {
        "message": "Задача инициализации добавлена",
        "task_id": task.id,
        "dataset_id": dataset_id
    }



@router.post("/{dataset_id}/resegment")
async def resegment_dataset_route(
    dataset_id: int, 
    data: DatasetResegmentRequest, 
    db: Session = Depends(get_db)
):
    """
    Ресегментация существующего датасета с новыми параметрами
    
    - **dataset_id**: ID датасета для ресегментации
    - **min_duration**: Минимальная длительность сегмента в секундах
    - **max_duration**: Максимальная длительность сегмента в секундах
    """
    
    # Проверяем существование датасета
    existing_dataset = db.query(AudioDataset).filter(AudioDataset.id == dataset_id).first()
    if not existing_dataset:
        logger.warning(f"Попытка ресегментации несуществующего датасета: ID={dataset_id}")
        raise HTTPException(
            status_code=404,
            detail=f"Датасет с ID {dataset_id} не найден"
        )
    
    # Проверяем что датасет готов к ресегментации
    from app.models.datasets import DatasetStatus
    if existing_dataset.status not in [DatasetStatus.SAMPLED, DatasetStatus.ERROR]:
        logger.warning(f"Попытка ресегментации датасета в неподходящем статусе: ID={dataset_id}, status={existing_dataset.status}")
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Датасет в статусе '{existing_dataset.status}' не может быть ресегментирован",
                "current_status": existing_dataset.status,
                "allowed_statuses": ["SAMPLED", "ERROR"]
            }
        )
    
    # Проверяем корректность параметров
    if data.min_duration >= data.max_duration:
        raise HTTPException(
            status_code=400,
            detail="Минимальная длительность должна быть меньше максимальной"
        )
    
    # Логируем начало процесса
    logger.info(f"Запуск ресегментации датасета: ID={dataset_id}, min={data.min_duration}s, max={data.max_duration}s")
    
    # Запускаем фоновую задачу ресегментации
    task = resegment_dataset_task.delay(
        dataset_id=dataset_id,
        min_duration=data.min_duration,
        max_duration=data.max_duration
    )
    
    return {
        "message": "Задача ресегментации добавлена в очередь",
        "task_id": task.id,
        "dataset_id": dataset_id,
        "dataset_name": existing_dataset.name,
        "parameters": {
            "min_duration": data.min_duration,
            "max_duration": data.max_duration
        }
    }